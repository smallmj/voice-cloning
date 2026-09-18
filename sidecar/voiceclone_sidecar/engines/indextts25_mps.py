"""IndexTTS-2.5 on Apple Silicon (MPS, PyTorch) — the local engine for Macs.

Sibling of the CUDA engine (indextts25.py, issue #4) with three deltas:

- torch comes from the standard PyPI index — the macOS arm64 wheels bundle
  MPS support, so there is no explicit CUDA wheel URL step. Versions are
  still pinned exactly (ADR-0002 rule 1) and stay at 2.8.0, below the
  TorchAudio 2.9 TorchCodec cutover that silently clips WAV output.
- The worker gates on ``torch.backends.mps`` instead of CUDA and pins
  ``device="mps"`` with fp32 (the only precision verified end-to-end on
  Apple Silicon); see indextts25_mps_worker.py.
- The weights step reuses weights that already exist on the machine before
  touching the network: point ``VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS`` at an
  existing IndexTTS-2.5 checkpoint directory (e.g. a prior full clone) and
  every matching file is copied in instead of downloaded. The downloader's
  skip-if-complete logic covers whatever remains.

Everything else — the two-step dependency split, the source list
(Hugging Face → hf-mirror → ModelScope) with loud mirror failure, the
pre-seeded ``hf_cache`` aux layout, HF_HUB_OFFLINE at generation time, and
the worker supervisor (idle recycle / crash restart) — is shared with the
CUDA engine verbatim.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from ..capabilities import Capabilities
from ..registry import GenerationRequest, GenerationResult, InstallableEngine
from ..runtime import downloader, installer, paths, uvman
from .indextts25 import (
    AUX_WEIGHTS,
    ENGINE_PACKAGE_URL,
    ENGINE_PACKAGE_URL_FALLBACK,
    HF,
    HF_MIRROR,
    MAIN_WEIGHTS_FILES,
    MODELSCOPE,
    REPO,
    _runtime_env,
)
from .indextts25_mps_worker import MPS_GATE_EXIT_CODE  # noqa: F401 - re-exported for tests
from .worker_supervisor import WorkerDegradedError, WorkerSupervisor

PYTHON_SPEC = "3.11"  # index-tts requires >=3.10,<3.12
TORCH_VERSION = "2.8.0"  # pinned: macOS arm64 wheels bundle MPS; <2.9 avoids TorchAudio's TorchCodec WAV path

LOCAL_WEIGHTS_ENV = "VOICECLONE_INDEXTTS25_LOCAL_WEIGHTS"

IDLE_TIMEOUT_S = 300.0  # reclaim memory after five idle minutes
MAX_REQUESTS_PER_WORKER = 20  # recycle the worker to bound slow memory growth


def _copy_from_local(local_root: Path, relative: str, dest: Path, log) -> bool:
    """Copy one weight file from a local checkpoint tree. True if done.

    Existing complete destination files are left alone (skip-if-complete,
    same contract as the downloader); a zero-byte local file is treated as
    absent rather than copied.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True
    src = local_root / relative
    if not src.is_file() or src.stat().st_size == 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"weights: reusing local {relative}")
    shutil.copy2(src, dest)
    return True


class IndexTts25MpsEngine(InstallableEngine):
    """Local engine for Apple Silicon. Also carries the InstallableEngine surface."""

    engine_id = "indextts-25-mps"
    display_name = "IndexTTS-2.5（本地 · MPS）"

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
            max_chars_per_request=500,  # unified-memory ceiling; keep requests small (issue #13)
        )

    # -- install surface ----------------------------------------------------

    def _ctx(self) -> installer.InstallContext:
        return installer.InstallContext(engine_id=self.engine_id, root=self.root, env=self.env)

    def is_installed(self) -> bool:
        return bool(installer.load_state(self._ctx()).get("installed"))

    def install_state(self) -> dict:
        state = installer.load_state(self._ctx())
        return {"installed": bool(state.get("installed")), "steps": state.get("steps", {})}

    def _local_weights_root(self) -> Path | None:
        override = self.env.get(LOCAL_WEIGHTS_ENV, "")
        return Path(override).expanduser() if override else None

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
            # macOS arm64 wheels from the configured index bundle MPS — no
            # explicit CUDA wheel URLs, but versions stay pinned exactly.
            uvman.pip_install(
                uv, venv_,
                [f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"],
                env=self.env, log=log,
            )

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

        def _fetch(relative: str, dest_name: str, sources: list[str], log, progress, label: str) -> None:
            local = self._local_weights_root()
            if local is not None:
                # Main weights sit at their repo-relative paths in a clone,
                # but aux models live under the pre-seeded hf_cache/ layout —
                # try the destination-relative path first, then repo-relative.
                for local_relative in dict.fromkeys((dest_name, relative)):
                    if _copy_from_local(local, local_relative, weights_dir / dest_name, log):
                        return
            log(f"{label}: downloading {relative}")
            downloader.download_file(
                downloader.DownloadSpec(path=relative, dest_name=dest_name),
                weights_dir, sources,
                progress=lambda name, done, total, _n=relative: progress(f"{label}:{_n}", done, total),
                log=log,
            )

        def step_weights(log, progress):
            sources = [
                HF.format(repo=REPO, path="{path}"),
                HF_MIRROR.format(repo=REPO, path="{path}"),
                MODELSCOPE.format(ms_repo=REPO, path="{path}"),
            ]
            for f in MAIN_WEIGHTS_FILES:
                _fetch(f, f, sources, log, progress, "weights")
            log(f"weights: {len(MAIN_WEIGHTS_FILES)} files ready in {weights_dir}")

        def step_aux(log, progress):
            for repo, path, dest, ms_repo in AUX_WEIGHTS:
                sources = [
                    HF.format(repo=repo, path="{path}"),
                    HF_MIRROR.format(repo=repo, path="{path}"),
                ]
                if ms_repo:
                    sources.append(MODELSCOPE.format(ms_repo=ms_repo, path="{path}"))
                _fetch(path, dest, sources, log, progress, "aux")
            log("aux: w2v-bert-2.0 / semantic codec / campplus / bigvgan ready")

        return [
            installer.InstallStep("python", f"安装 uv 托管的 Python {PYTHON_SPEC}", step_python),
            installer.InstallStep("venv", "创建引擎独立 venv", step_venv, artifact=venv),
            installer.InstallStep("torch", f"安装 torch/torchaudio {TORCH_VERSION}（PyPI，macOS arm64 自带 MPS）", step_torch, artifact=venv),
            installer.InstallStep("engine", "安装 index-tts 引擎代码", step_engine, artifact=venv),
            # No artifact wipe on these steps: files are reused/verified
            # individually, wiping the dir would discard local weights (observed).
            installer.InstallStep("weights", f"准备引擎权重 {REPO}（本地复用优先）", step_weights),
            installer.InstallStep("aux", "准备辅助模型（w2v-bert / MaskGCT / CAMPPlus / BigVGAN，本地复用优先）", step_aux),
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
                worker = Path(__file__).with_name("indextts25_mps_worker.py")
                self._supervisor = WorkerSupervisor(
                    command=[str(python), "-u", str(worker)],
                    cwd=str(self.root),
                    idle_timeout_s=self.idle_timeout_s,
                    max_requests=self.max_requests,
                )
            return self._supervisor

    def unload(self) -> None:
        """Explicit memory release: drop the loaded model and exit the worker."""
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
                "engine is not installed yet — call POST /engines/indextts-25-mps/install first"
            )

        params = request.params or {}
        if not params.get("ref_audio"):
            # Zero-shot cloning with no reference: upstream fails deep inside
            # soundfile with "Invalid file: None" — refuse here with an
            # actionable message instead.
            raise RuntimeError(
                "IndexTTS-2.5 是零样本复刻引擎，必须提供参考音频：请选择一个音色（或先创建音色）再生成"
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
            raise RuntimeError(f"MPS gate refused to start the engine: {exc}") from exc
        if result.get("clipping"):
            log("indextts-2.5: output hit full scale — consider lowering gain on the reference audio")
        return GenerationResult(
            audio_path=result["audio_path"],
            sample_rate=result["sample_rate"],
            model_version=REPO,
            cost=0.0,  # local synthesis has no per-run cost
        )
