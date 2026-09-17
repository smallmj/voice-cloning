"""IndexTTS-2.5 on Windows + CUDA (PyTorch) — the local engine for NVIDIA rigs.

Dependencies come in two explicit steps so the CUDA index can never leak into
the rest of the install:

- ``torch`` / ``torchaudio`` are declared as direct deps of the torch step
  and fetched as EXACT wheel URLs (the explicit-index requirement, taken to
  its strongest form: no index resolution, no extra-index leaks) with a
  mirror fallback list, then installed from the local wheel files;
- the engine package and everything else install from the configured PyPI
  index in a second step.

Weights download through a source list (Hugging Face → hf-mirror → ModelScope)
with resumable, byte-progressed transfers; mirror failure is LOUD: the step
fails with the per-source errors instead of silently degrading. Auxiliary
models (w2v-bert-2.0, MaskGCT semantic codec, CAMPPlus, BigVGAN) are
pre-downloaded into the ``hf_cache`` layout ``ensure_models_available`` expects,
so generation never touches the network (HF_HUB_OFFLINE=1).
"""

from __future__ import annotations

import threading
import urllib.parse
import uuid
from pathlib import Path

from ..capabilities import Capabilities
from ..registry import GenerationRequest, GenerationResult, InstallableEngine
from ..runtime import downloader, installer, paths, uvman
from .worker_supervisor import WorkerDegradedError, WorkerSupervisor

REPO = "IndexTeam/IndexTTS-2.5"
PYTHON_SPEC = "3.11"  # index-tts requires >=3.10,<3.12
TORCH_VERSION = "2.8.0"
CUDA_TAG = "cu128"
PY_TAG = "cp311"
PLATFORM_TAG = "win_amd64"  # this engine only registers on Windows
TORCH_WHEELS = [
    f"torch-{TORCH_VERSION}%2B{CUDA_TAG}-{PY_TAG}-{PY_TAG}-{PLATFORM_TAG}.whl",
    f"torchaudio-{TORCH_VERSION}%2B{CUDA_TAG}-{PY_TAG}-{PY_TAG}-{PLATFORM_TAG}.whl",
]
# download.pytorch.org is the canonical CUDA wheel source; the aliyun
# pytorch-wheels mirror is the fallback that actually answers on many China
# networks (observed 2026-09-17: download.pytorch.org TLS-blackholed while
# aliyun responded fine). Mirror failure is loud, never a silent slowdown.
TORCH_SOURCES = [
    "https://download.pytorch.org/whl/" + CUDA_TAG + "/{path}",
    "https://mirrors.aliyun.com/pytorch-wheels/" + CUDA_TAG + "/{path}",
]
# Measured 2026-09-17 on the Windows rig: download.pytorch.org ~800 KB/s
# (after an initial TLS-blackhole window), aliyun ~240 KB/s per connection,
# so the official source stays first and aliyun is the fallback.
ENGINE_PACKAGE_URL = (
    "https://ghfast.top/https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
)
ENGINE_PACKAGE_URL_FALLBACK = "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"

HF = f"https://huggingface.co/{{repo}}/resolve/main/{{path}}"
HF_MIRROR = f"https://hf-mirror.com/{{repo}}/resolve/main/{{path}}"
MODELSCOPE = f"https://modelscope.cn/models/{{ms_repo}}/resolve/master/{{path}}"

# Downloader source templates interpolate {path}; these carry {repo}/{ms_repo}
# too, so each repo passes its own identifiers alongside the file path.
MAIN_WEIGHTS_FILES = [
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "feat1.pt",
    "feat2.pt",
    "wav2vec2bert_stats.pt",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "qwen0.6bemo4-merge/model.safetensors",
    "qwen0.6bemo4-merge/config.json",
    "qwen0.6bemo4-merge/tokenizer.json",
    "qwen0.6bemo4-merge/tokenizer_config.json",
    "qwen0.6bemo4-merge/vocab.json",
    "qwen0.6bemo4-merge/merges.txt",
    "qwen0.6bemo4-merge/added_tokens.json",
    "qwen0.6bemo4-merge/special_tokens_map.json",
    "qwen0.6bemo4-merge/chat_template.jinja",
    "qwen0.6bemo4-merge/generation_config.json",
]

# repo_id on HF, {path} inside that repo, dest name under weights/hf_cache,
# ModelScope model id (None = no ModelScope source).
AUX_WEIGHTS = [
    ("facebook/w2v-bert-2.0", "config.json", "hf_cache/w2v-bert-2.0/config.json", "AI-ModelScope/w2v-bert-2.0"),
    ("facebook/w2v-bert-2.0", "preprocessor_config.json", "hf_cache/w2v-bert-2.0/preprocessor_config.json", "AI-ModelScope/w2v-bert-2.0"),
    ("facebook/w2v-bert-2.0", "model.safetensors", "hf_cache/w2v-bert-2.0/model.safetensors", "AI-ModelScope/w2v-bert-2.0"),
    ("amphion/MaskGCT", "semantic_codec/model.safetensors", "hf_cache/semantic_codec_model.safetensors", "amphion/MaskGCT"),
    ("funasr/campplus", "campplus_cn_common.bin", "hf_cache/campplus_cn_common.bin", "iic/speech_campplus_sv_zh-cn_16k-common"),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "config.json", "hf_cache/bigvgan/config.json", None),
    ("nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt", "hf_cache/bigvgan/bigvgan_generator.pt", None),
]

IDLE_TIMEOUT_S = 300.0  # release GPU memory after five idle minutes
MAX_REQUESTS_PER_WORKER = 20  # recycle the worker to bound slow VRAM leaks


class IndexTts25CudaEngine(InstallableEngine):
    """Local engine. Also carries the InstallableEngine surface."""

    engine_id = "indextts-25-cuda"
    display_name = "IndexTTS-2.5（本地 · CUDA）"

    def __init__(self, output_dir: Path | None = None, root: Path | None = None,
                 env: dict | None = None,
                 idle_timeout_s: float = IDLE_TIMEOUT_S,
                 max_requests: int = MAX_REQUESTS_PER_WORKER) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        self.env = env if env is not None else dict(_runtime_env())
        self.root = root if root is not None else paths.runtime_root(self.env)
        self.idle_timeout_s = idle_timeout_s
        self.max_requests = max_requests
        self._supervisor: WorkerSupervisor | None = None
        self._supervisor_lock = threading.Lock()

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            languages=("zh", "en", "ja", "es", "ar"),
            voice_cloning=True,
            voice_design=False,
            pronunciation_control=True,  # pinyin / CMU phonemes
            emotion=True,
            commercial_license=False,  # bilibili model license
            cross_device_use=False,
            upload_used_for_training=False,
            api_closed_loop=False,
            requires_reference_text=True,  # the sidecar auto-fills ref_text from the transcript
        )

    # -- install surface ----------------------------------------------------

    def _ctx(self) -> installer.InstallContext:
        return installer.InstallContext(engine_id=self.engine_id, root=self.root, env=self.env)

    def is_installed(self) -> bool:
        return bool(installer.load_state(self._ctx()).get("installed"))

    def install_state(self) -> dict:
        state = installer.load_state(self._ctx())
        return {"installed": bool(state.get("installed")), "steps": state.get("steps", {})}

    def install_steps(self) -> list[installer.InstallStep]:
        weights_dir = paths.engine_weights_dir(self.root, self.engine_id)
        venv = uvman.engine_venv_dir(self.root, self.engine_id)

        def find_uv(log):
            return uvman.find_uv(self.root, env=self.env, log=log)

        def step_python(log, progress):
            uvman.ensure_python(find_uv(log), PYTHON_SPEC, env=self.env, log=log)

        def step_venv(log, progress):
            uvman.create_venv(find_uv(log), self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)

        def step_torch(log, progress):
            uv = find_uv(log)
            venv_ = uvman.create_venv(uv, self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)
            wheels_dir = paths.engine_dir(self.root, self.engine_id) / "wheels"
            local_wheels = []
            for wheel in TORCH_WHEELS:
                log(f"torch: fetching {wheel}")
                local_wheels.append(
                    downloader.download_file(
                        # URL name carries %2B; the local file must carry a
                        # literal "+" or uv rejects the wheel filename.
                        downloader.DownloadSpec(path=wheel, dest_name=urllib.parse.unquote(wheel)),
                        wheels_dir, TORCH_SOURCES,
                        progress=lambda name, done, total, _w=wheel: progress(f"torch:{_w}", done, total),
                        log=log,
                    )
                )
            # Install from the local wheel files — no index is consulted at all.
            uvman.pip_install(uv, venv_, [str(w) for w in local_wheels], env=self.env, log=log)

        def step_engine(log, progress):
            uv = find_uv(log)
            venv_ = uvman.create_venv(uv, self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)
            pkg = self.env.get("VOICECLONE_INDEXTTS_URL", ENGINE_PACKAGE_URL)
            try:
                uvman.pip_install(uv, venv_, [f"indextts @ {pkg}"], env=self.env, log=log)
            except uvman.RuntimeCommandError:
                fallback = self.env.get("VOICECLONE_INDEXTTS_URL_FALLBACK", ENGINE_PACKAGE_URL_FALLBACK)
                log(f"engine package install from {pkg} failed; retrying from {fallback}")
                uvman.pip_install(uv, venv_, [f"indextts @ {fallback}"], env=self.env, log=log)

        def step_weights(log, progress):
            sources = [
                HF.format(repo=REPO, path="{path}"),
                HF_MIRROR.format(repo=REPO, path="{path}"),
                MODELSCOPE.format(ms_repo=REPO, path="{path}"),
            ]
            for f in MAIN_WEIGHTS_FILES:
                log(f"weights: {f}")
                downloader.download_file(
                    downloader.DownloadSpec(path=f, dest_name=f),
                    weights_dir, sources,
                    progress=lambda name, done, total, _n=f: progress(f"weights:{_n}", done, total),
                    log=log,
                )
            log(f"weights: {len(MAIN_WEIGHTS_FILES)} files ready in {weights_dir}")

        def step_aux(log, progress):
            for repo, path, dest, ms_repo in AUX_WEIGHTS:
                sources = [
                    HF.format(repo=repo, path="{path}"),
                    HF_MIRROR.format(repo=repo, path="{path}"),
                ]
                if ms_repo:
                    sources.append(MODELSCOPE.format(ms_repo=ms_repo, path="{path}"))
                log(f"aux: {repo}/{path}")
                downloader.download_file(
                    downloader.DownloadSpec(path=path, dest_name=dest),
                    weights_dir, sources,
                    progress=lambda name, done, total, _n=dest: progress(f"aux:{_n}", done, total),
                    log=log,
                )
            log("aux: w2v-bert-2.0 / semantic codec / campplus / bigvgan ready")

        return [
            installer.InstallStep("python", f"安装 uv 托管的 Python {PYTHON_SPEC}", step_python),
            installer.InstallStep("venv", "创建引擎独立 venv", step_venv, artifact=venv),
            installer.InstallStep("torch", f"安装 torch/torchaudio {TORCH_VERSION}（CUDA {CUDA_TAG}，显式 wheel 源）", step_torch, artifact=venv),
            installer.InstallStep("engine", "安装 index-tts 引擎代码", step_engine, artifact=venv),
            # No artifact wipe on these download steps: the downloader
            # guarantees per-file completeness (skip-if-complete + .part
            # resume), and wiping the whole weights dir on a retry would
            # re-download many GB that were already fine (observed).
            installer.InstallStep("weights", f"下载引擎权重 {REPO}", step_weights),
            installer.InstallStep("aux", "下载辅助模型（w2v-bert / MaskGCT / CAMPPlus / BigVGAN）", step_aux),
        ]

    def install(self, log, progress=None) -> dict:
        return installer.run_install(self._ctx(), self.install_steps(), log, progress)

    # -- worker management ---------------------------------------------------

    def _get_supervisor(self) -> WorkerSupervisor:
        with self._supervisor_lock:
            if self._supervisor is None:
                venv = paths.engine_venv_dir(self.root, self.engine_id)
                python = uvman.venv_python(venv)
                if not python.exists():
                    raise RuntimeError(
                        f"engine venv is broken (no python at {python}); reinstall the engine"
                    )
                worker = Path(__file__).with_name("indextts25_worker.py")
                self._supervisor = WorkerSupervisor(
                    command=[str(python), "-u", str(worker)],
                    cwd=str(self.root),
                    idle_timeout_s=self.idle_timeout_s,
                    max_requests=self.max_requests,
                )
            return self._supervisor

    def unload(self) -> None:
        """Explicit VRAM release: drop the loaded model and exit the worker."""
        with self._supervisor_lock:
            if self._supervisor is not None:
                self._supervisor.unload()

    def worker_stats(self) -> dict:
        with self._supervisor_lock:
            return self._supervisor.stats() if self._supervisor else {"alive": False}

    # -- generation ---------------------------------------------------------

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        if not self.is_installed():
            raise RuntimeError(
                "engine is not installed yet — call POST /engines/indextts-25-cuda/install first"
            )
        out_dir = (self.output_dir or Path.cwd() / "data" / "audio").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav").resolve()

        params = request.params or {}
        payload = {
            "action": "synthesize",
            "model_dir": str(paths.engine_weights_dir(self.root, self.engine_id)),
            "text": request.text,
            "output": str(out_path),
            "lang": params.get("lang", "zh"),
        }
        if params.get("ref_audio"):
            payload["ref_audio"] = params["ref_audio"]

        supervisor = self._get_supervisor()
        try:
            result = supervisor.request(payload, on_log=log)
        except WorkerDegradedError as exc:
            raise RuntimeError(f"CUDA gate refused to start the engine: {exc}") from exc
        if result.get("clipping"):
            log("indextts-2.5: output hit full scale — consider lowering gain on the reference audio")
        return GenerationResult(
            audio_path=result["audio_path"],
            sample_rate=result["sample_rate"],
            model_version=REPO,
            cost=0.0,  # local synthesis has no per-run cost
        )


def _runtime_env() -> dict:
    """China-friendly defaults; users can override any of them."""
    return {
        "UV_PYTHON_INSTALL_MIRROR": "https://ghfast.top/https://github.com/indygreg/python-build-standalone/releases/download",
        # PyPI index for the non-torch packages; overridable per machine.
        "UV_DEFAULT_INDEX": "https://mirrors.aliyun.com/pypi/simple/",
    }
