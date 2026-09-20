"""dots.tts on Windows + CUDA — the effect-tier local engine (issue #25).

dots.tts (studio-dots-ai, Apache-2.0, 2B fully-continuous AR + DiT
flow-matching, 48 kHz) is the Windows 效果档 candidate. Two facts shape
this adapter, both verified against upstream source at tag v0.3.1 and
recorded in docs/research/CORRECTIONS.md C-003:

- **WeTextProcessing is a hard import-time dependency** dragging in pynini,
  which has zero official win_amd64 wheels — the exact reason plan.md §5
  once rejected dots.tts for Windows. Since this app normalizes text in its
  own layer BEFORE the engine (ADR-0008), the install skips pynini entirely
  (``--no-deps`` + an explicit dependency list) and applies an
  anchor-checked patch (dots_tts_patch.py) making the ``tn.*`` imports
  optional. ``normalize_text`` is declared ``breaks-pipeline``, never shown.
- **No MPS path** (runtime selects cuda→cpu) — this engine registers on
  Windows only.

Cloning: ``prompt_audio_path`` + ``prompt_text`` (transcript must match the
audio; best similarity) is the vendor-recommended mode; audio alone runs
the CAM++ x-vector timbre mode. Polyphone control ONLY via tone-marked
pinyin inline (``hào``; digit forms like ``hao4`` are not recognized) — the
canonical pronunciation adapter grammar ``dots`` does exactly this.

Anti-fake-parameter note (verified, README SGLang section): dots.tts has NO
temperature/top_p/top_k (continuous latent flow-matching, no token sampler)
and NO speed parameter — none are declared here.
"""

from __future__ import annotations

import threading
import urllib.parse
import uuid
from pathlib import Path

from .. import pronunciation, sources
from .. import params as params_mod
from ..capabilities import AppliesTo, Capabilities, ParamSpec
from ..engine_config import EngineConfig, resolve_seam
from ..registry import GenerationRequest, GenerationResult, InstallableEngine
from ..runtime import downloader, installer, paths, uvman
from .dots_tts_patch import apply_to_venv
from .worker_supervisor import WorkerDegradedError, WorkerSupervisor

REPO = "dots-studio/dots.tts-soar"  # best-SIM checkpoint (vendor benchmark)
MS_REPO = "dots-studio/dots.tts-soar"  # ModelScope twin verified 2026-09-20
PYTHON_SPEC = "3.11"  # upstream: >=3.10,<3.13
TORCH_VERSION = "2.8.0"  # constraints/recommended.txt pins the same
DOTS_VERSION = "0.3.1"

# Vendor checkpoint layout (verified against the HF/ModelScope file tree).
WEIGHTS_FILES = [
    "model.safetensors",
    "vocoder.safetensors",
    "speaker_encoder.safetensors",
    "latent_stats.pt",
    "config.json",
    "llm_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]

# Headless runtime dependency set (verified against upstream source, pinned
# from constraints/recommended.txt where a pin exists): torch is installed
# by the torch step; WeTextProcessing (pynini) is deliberately ABSENT — see
# dots_tts_patch.py; gradio is declared upstream but never imported by the
# runtime path.
DOTS_DEPENDENCIES = [
    "transformers==4.57.0",
    "huggingface-hub",
    "loguru",
    "langcodes[data]",
    "einops",
    "librosa==0.11.0",
    "soundfile",
    "numpy==2.2.6",
    "pydantic==2.12.5",
    "PyYAML",
    "safetensors==0.8.0rc0",
    "torchdiffeq",
    "tqdm",
    "lingua-language-detector",
]

IDLE_TIMEOUT_S = 300.0
MAX_REQUESTS_PER_WORKER = 20

CUDA_TAG = "cu128"
PY_TAG = "cp311"
PLATFORM_TAG = "win_amd64"  # this engine only registers on Windows
TORCH_WHEELS = [
    f"torch-{TORCH_VERSION}%2B{CUDA_TAG}-{PY_TAG}-{PY_TAG}-{PLATFORM_TAG}.whl",
    f"torchaudio-{TORCH_VERSION}%2B{CUDA_TAG}-{PY_TAG}-{PY_TAG}-{PLATFORM_TAG}.whl",
]


def param_specs_for(engine_id: str) -> list[ParamSpec]:
    return [
        params_mod.canonical_pronunciation(
            engine_id, REPO,
            grammar_help=(
                "每行一条「汉字=拼音」（如 号=hao4），将改写为 dots.tts 的带调拼音 "
                "（号→hào）后送入引擎。注意：dots.tts 只认带声调符号的形式，不认数字声调。"
            ),
        ),
        ParamSpec(
            name="speaker_scale",
            label="音量/说话人强度",
            kind="number",
            default=1.5,
            step=0.05,
            layer="engine",
            help="说话人条件强度（上游默认 1.5）。留空使用默认值。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="num_steps",
            label="流匹配步数",
            kind="number",
            integer=True,
            default=10,
            min=4,
            max=32,
            step=1,
            layer="engine",
            help="ODE 求解步数（NFE）：soar/base 建议 10–32，越高越稳、越慢。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="guidance_scale",
            label="引导强度",
            kind="number",
            default=1.2,
            step=0.05,
            layer="engine",
            help="flow-matching 引导强度（上游默认 1.2）。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        ParamSpec(
            name="seed",
            label="随机种子",
            kind="number",
            integer=True,
            min=0,
            max=2**31 - 1,
            layer="engine",
            help="上游 runtime 无 seed 参数，由本引擎在每次请求前调用官方 seed_everything() 固定；留空不固定。",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
        ),
        # 「不支持是数据」（ADR-0018 decision 3）
        ParamSpec(
            name="normalize_text",
            label="引擎内文本归一化",
            kind="bool",
            exposed=False,
            not_exposed_reason="breaks-pipeline",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="归一化由本仓库自建归一化层在进引擎前完成（ADR-0008）；且本引擎安装形态已去除 WeTextProcessing/pynini（CORRECTIONS.md C-003），此项不可用。",
        ),
        ParamSpec(
            name="temperature",
            label="温度",
            kind="text",
            exposed=False,
            not_exposed_reason="no-op",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="参数在 dots.tts 中不存在：全连续 latent flow-matching，没有 token 采样器（官方 README 明示）。",
        ),
        ParamSpec(
            name="speed",
            label="语速",
            kind="text",
            exposed=False,
            not_exposed_reason="no-op",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="dots.tts 无语速参数（速率控制仅经独立的 dots.tts.edit 模型，未接入）。",
        ),
        ParamSpec(
            name="ref_audio",
            label="参考音频",
            kind="text",
            exposed=False,
            not_exposed_reason="server-injected",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="由音色解析自动注入，不作为用户参数。",
        ),
        ParamSpec(
            name="ref_text",
            label="参考文本",
            kind="text",
            exposed=False,
            not_exposed_reason="server-injected",
            applies_to=AppliesTo(engine=engine_id, model=REPO, mode="cloning"),
            help="可选：有转写走官方推荐的续写复刻（最佳相似度），无转写走 x-vector 纯音色模式；由参考音频转写自动填充。",
        ),
    ]


class DotsTtsCudaEngine(InstallableEngine):
    """dots.tts on Windows + CUDA. Carries the InstallableEngine surface."""

    engine_id = "dots-tts-cuda"
    display_name = "dots.tts（本地 · CUDA）"

    def __init__(self, output_dir: Path | None = None, root: Path | None = None,
                 env: dict | None = None,
                 idle_timeout_s: float = IDLE_TIMEOUT_S,
                 max_requests: int = MAX_REQUESTS_PER_WORKER,
                 config: EngineConfig | None = None) -> None:
        output_dir, env = resolve_seam(config, self.engine_id, output_dir, env)
        self.output_dir = Path(output_dir) if output_dir else None
        self.env = env
        self.root = root if root is not None else paths.runtime_root(self.env)
        self.idle_timeout_s = idle_timeout_s
        self.max_requests = max_requests
        self._supervisor: WorkerSupervisor | None = None
        self._supervisor_lock = threading.Lock()

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            # 24 languages vendor-benchmarked (best zh/en); zh/en are what we
            # declare here — the rest stays a matrix note (vendor level).
            languages=("zh", "en"),
            voice_cloning=True,
            voice_design=False,
            pronunciation_control=True,  # tone-marked pinyin ONLY (verified)
            emotion=False,
            commercial_license=True,  # Apache-2.0 code + weights
            cross_device_use=False,
            upload_used_for_training=False,
            api_closed_loop=False,
            # Transcript optional: with it = continuation cloning (best SIM),
            # without = x-vector timbre mode (verified both upstream).
            requires_reference_text=False,
            max_chars_per_request=None,  # cap is duration (~80 s audio), not chars
        )

    def param_specs(self) -> list[ParamSpec]:
        return param_specs_for(self.engine_id)

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
            torch_chain = sources.torch_wheel_sources(
                CUDA_TAG, preferred=self.env.get(sources.PREFERRED_CUDA_ENV)
            )
            local_wheels = []
            for wheel in TORCH_WHEELS:
                log(f"torch: fetching {wheel}")
                local_wheels.append(
                    downloader.download_file(
                        downloader.DownloadSpec(path=wheel, dest_name=urllib.parse.unquote(wheel)),
                        wheels_dir, torch_chain,
                        progress=lambda name, done, total, _w=wheel: progress(f"torch:{_w}", done, total),
                        log=log,
                    )
                )
            uvman.pip_install(uv, venv_, [str(w) for w in local_wheels], env=self.env, log=log)

        def step_engine(log, progress):
            uv = find_uv(log)
            venv_ = uvman.create_venv(uv, self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)
            # --no-deps + explicit list: the upstream dependency set includes
            # WeTextProcessing (pynini) which has no win_amd64 wheel and is
            # redundant here (ADR-0008). The patch below makes it optional.
            uvman.pip_install(uv, venv_, [f"dots.tts=={DOTS_VERSION}", "--no-deps"], env=self.env, log=log)
            uvman.pip_install(uv, venv_, DOTS_DEPENDENCIES, env=self.env, log=log)
            for line in apply_to_venv(venv_, log):
                log(f"dots.tts patch: {line}")

        def step_weights(log, progress):
            chain = sources.weight_sources(REPO, ms_repo=MS_REPO, preferred=sources.weight_pref(self.env))
            for rel in WEIGHTS_FILES:
                log(f"weights: downloading {rel}")
                downloader.download_file(
                    downloader.DownloadSpec(path=rel, dest_name=rel),
                    weights_dir, chain,
                    progress=lambda name, done, total, _n=rel: progress(f"weights:{_n}", done, total),
                    log=log,
                )
            log(f"weights: {len(WEIGHTS_FILES)} files ready in {weights_dir}")

        return [
            installer.InstallStep("python", f"安装 uv 托管的 Python {PYTHON_SPEC}", step_python),
            installer.InstallStep("venv", "创建引擎独立 venv", step_venv, artifact=venv),
            installer.InstallStep("torch", f"安装 torch/torchaudio {TORCH_VERSION}（CUDA {CUDA_TAG}，显式 wheel 源）", step_torch, artifact=venv),
            installer.InstallStep("engine", f"安装 dots.tts {DOTS_VERSION} 并打补丁（去除 pynini 硬依赖）", step_engine, artifact=venv),
            installer.InstallStep("weights", f"下载引擎权重 {REPO}（约 5.2 GB）", step_weights),
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
                worker = Path(__file__).with_name("dots_tts_worker.py")
                self._supervisor = WorkerSupervisor(
                    command=[str(python), "-u", str(worker)],
                    cwd=str(self.root),
                    idle_timeout_s=self.idle_timeout_s,
                    max_requests=self.max_requests,
                )
            return self._supervisor

    def unload(self) -> None:
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
                f"engine is not installed yet — call POST /engines/{self.engine_id}/install first"
            )
        params = request.params or {}
        if not params.get("ref_audio"):
            raise RuntimeError(
                "dots.tts 是零样本复刻引擎，必须提供参考音频：请选择一个音色（或先创建音色）再生成"
            )
        out_dir = (self.output_dir or Path.cwd() / "data" / "audio").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav").resolve()

        text = request.text
        ann = params.get("pronunciation")
        if ann and str(ann).strip():
            text = pronunciation.rewrite(text, str(ann), "dots", log)

        ref_text = (params.get("ref_text") or "").strip()
        payload: dict = {
            "action": "synthesize",
            "model_dir": str(paths.engine_weights_dir(self.root, self.engine_id)),
            "text": text,
            "ref_audio": str(Path(params["ref_audio"]).resolve()),
            "output": str(out_path),
        }
        for key in ("speaker_scale", "num_steps", "guidance_scale", "seed"):
            if params.get(key) not in (None, ""):
                payload[key] = params[key]
        if ref_text:
            # Vendor-recommended continuation cloning (best similarity).
            payload["prompt_text"] = ref_text
            log("dots.tts: 复刻模式 = 续写复刻（参考音频 + 逐字转写，最佳相似度）")
        else:
            log("dots.tts: 复刻模式 = 纯音色（x-vector，无参考文本；提供转写可获得更高相似度）")

        supervisor = self._get_supervisor()
        try:
            result = supervisor.request(payload, on_log=log)
        except WorkerDegradedError as exc:
            raise RuntimeError(f"CUDA gate refused to start the engine: {exc}") from exc
        if result.get("clipping"):
            log("dots.tts: output hit full scale — consider lowering gain on the reference audio")
        return GenerationResult(
            audio_path=result["audio_path"],
            sample_rate=result["sample_rate"],
            model_version=f"{REPO} (dots.tts {DOTS_VERSION})",
            cost=0.0,
        )
