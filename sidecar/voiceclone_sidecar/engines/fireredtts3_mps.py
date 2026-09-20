"""FireRedTTS3-Base on Apple Silicon (MPS) — issue #25, ADR-0019.

The issue-#26 PoC verified this engine end to end on MPS (20/20 incl. the
17-item Chinese regression set, RTF median 2.22 fp32, peak 11.7 GB) with a
four-class patch surface; ADR-0019 adopted it with hard constraints, all of
which shape this adapter:

- macOS MPS ONLY (Windows blocked by ``flash_attn``: zero win_amd64 wheels
  across all 76 PyPI releases — the CUDA variant needs a dependency strip
  that is out of scope);
- the patch table lives in ``fireredtts3_patch.py`` and is applied at
  INSTALL time with anchor checks that fail loudly on upstream drift
  (ADR-0019 decision 2);
- bf16-on-MPS is UNVERIFIED, so the worker stays fp32 (the verified
  precision) and the engine ships marked experimental until the bf16 route
  is measured;
- cloning ONLY: Base mode with reference audio + reference transcript
  (requires_reference_text=True). Instruct/voice-design is out of scope;
- compliance (ADR-0019 decision 4): upstream README declares zero-shot
  cloning "for academic research use only" — recorded in the capability
  matrix and NOTICE.md; the review conclusion is registered there.

Install layout under the engine dir:

- ``weights/``   — Base weights (campplus / fireredtts3_base / redae /
  text_tokenizer), HF → hf-mirror chain (ModelScope twin unverified);
- ``upstream/``  — the extracted upstream source checkout, patched in place.
"""

from __future__ import annotations

import shutil
import uuid
import zipfile
from pathlib import Path

from .. import sources
from ..capabilities import AppliesTo, Capabilities, ParamSpec
from ..engine_config import EngineConfig, resolve_seam
from ..registry import GenerationRequest, GenerationResult, InstallableEngine
from ..runtime import downloader, installer, paths, uvman
from .fireredtts3_patch import apply_patches
from .indextts25_base import copy_from_local

REPO = "FireRedTeam/FireRedTTS3"
PYTHON_SPEC = "3.11"
TORCH_VERSION = "2.8.0"  # ADR-0002 rule 1: pinned below the TorchAudio 2.9 cutover
TRANSFORMERS_VERSION = "5.6.2"  # verified by the issue-#26 PoC
EINOPS_VERSION = "0.8.2"  # verified by the issue-#26 PoC

# The patched upstream checkout (PoC-verified commit window; pinned via env
# override if a pin is ever needed — ADR-0019 expects re-checks on drift).
ENGINE_PACKAGE_URL = (
    "https://ghfast.top/https://github.com/FireRedTeam/FireRedTTS3/archive/refs/heads/{path}"
)
ENGINE_PACKAGE_URL_FALLBACK = "https://github.com/FireRedTeam/FireRedTTS3/archive/refs/heads/{path}"

# Base weights only (Instruct weights are out of scope). Paths verified by
# the issue-#26 PoC download (~12.3 GB total). No ModelScope twin verified.
WEIGHTS = [
    "campp/campplus_voxceleb.bin",
    "fireredtts3_base/config.json",
    "fireredtts3_base/model.safetensors",
    "redae/config.json",
    "redae/model.safetensors",
    "text_tokenizer/tokenizer.json",
    "text_tokenizer/tokenizer_config.json",
    "text_tokenizer/vocab.json",
]

LOCAL_UPSTREAM_ENV = "VOICECLONE_FIREREDTTS3_LOCAL_UPSTREAM"
LOCAL_WEIGHTS_ENV = "VOICECLONE_FIREREDTTS3_LOCAL_WEIGHTS"

# Instruct/voice-design is deliberately NOT exposed: only Base cloning was
# verified (PoC scope), and the upstream weight set installed here is Base-only.
UNEXPOSED_INSTRUCT = ParamSpec(
    name="instruct_mode",
    label="Instruct 指令模式",
    kind="text",
    exposed=False,
    not_exposed_reason="wrong-mode",
    applies_to=AppliesTo(engine="fireredtts3-mps", model=REPO, mode="cloning"),
    help="仅接入 Base 复刻模式（issue #26 PoC 范围）；Instruct 权重未安装、未验证。",
)

UNEXPOSED_REF_AUDIO = ParamSpec(
    name="ref_audio",
    label="参考音频",
    kind="text",
    exposed=False,
    not_exposed_reason="server-injected",
    applies_to=AppliesTo(engine="fireredtts3-mps", model=REPO, mode="cloning"),
    help="由音色解析自动注入，不作为用户参数。",
)

UNEXPOSED_REF_TEXT = ParamSpec(
    name="ref_text",
    label="参考文本",
    kind="text",
    exposed=False,
    not_exposed_reason="server-injected",
    applies_to=AppliesTo(engine="fireredtts3-mps", model=REPO, mode="cloning"),
    help="复刻必需；由参考音频转写自动填充（requires_reference_text），不作为用户参数。",
)


class FireRedTts3MpsEngine(InstallableEngine):
    """FireRedTTS3-Base, macOS MPS only (ADR-0019)."""

    engine_id = "fireredtts3-mps"
    display_name = "FireRedTTS3-Base（本地 · MPS · 实验性）"

    def __init__(self, output_dir: Path | None = None, root: Path | None = None,
                 env: dict | None = None, config: EngineConfig | None = None) -> None:
        output_dir, env = resolve_seam(config, self.engine_id, output_dir, env)
        self.output_dir = Path(output_dir) if output_dir else None
        self.env = env
        self.root = root if root is not None else paths.runtime_root(self.env)

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            # Verified on this machine: Chinese only (the PoC's 17-item set
            # is Chinese). Upstream advertises English + 21 Chinese
            # dialects — UNVERIFIED here, recorded in the capability matrix.
            languages=("zh",),
            voice_cloning=True,
            voice_design=False,  # Instruct mode out of scope (PoC decision)
            pronunciation_control=False,  # no pinyin/phoneme entry point verified
            emotion=False,
            # Apache-2.0 code + weights, BUT upstream README scopes zero-shot
            # cloning to academic research (not a LICENSE term — ADR-0019
            # decision 4). Conservative declaration, reasoning in the matrix.
            commercial_license=False,
            cross_device_use=False,
            upload_used_for_training=False,
            api_closed_loop=False,
            requires_reference_text=True,  # cloning prompt needs the transcript
            max_chars_per_request=None,  # no limit verified yet (unverified)
        )

    def param_specs(self) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="seed",
                label="随机种子",
                kind="number",
                integer=True,
                min=0,
                max=2**31 - 1,
                layer="engine",
                help="留空则不固定种子；同一音色与文本可用相同种子复现结果。",
                applies_to=AppliesTo(engine=self.engine_id, model=REPO, mode="cloning"),
            ),
            UNEXPOSED_INSTRUCT,
            UNEXPOSED_REF_AUDIO,
            UNEXPOSED_REF_TEXT,
        ]

    # -- install surface ----------------------------------------------------

    def _ctx(self) -> installer.InstallContext:
        return installer.InstallContext(engine_id=self.engine_id, root=self.root, env=self.env)

    def is_installed(self) -> bool:
        return bool(installer.load_state(self._ctx()).get("installed"))

    def install_state(self) -> dict:
        state = installer.load_state(self._ctx())
        return {"installed": bool(state.get("installed")), "steps": state.get("steps", {})}

    def _engine_dir(self) -> Path:
        return paths.engine_dir(self.root, self.engine_id)

    def _upstream_dir(self) -> Path:
        """The extracted (and patched) upstream source checkout."""
        return self._engine_dir() / "upstream"

    def _local_upstream_root(self) -> Path | None:
        """Env-named existing FireRedTTS3 checkout to seed the engine step
        from before touching the network (None = always download)."""
        override = self.env.get(LOCAL_UPSTREAM_ENV, "")
        return Path(override).expanduser() if override else None

    def _local_weights_root(self) -> Path | None:
        """Env-named existing Base-weight tree (e.g. the issue-#26 PoC
        download) to copy files from before downloading (None = never)."""
        override = self.env.get(LOCAL_WEIGHTS_ENV, "")
        return Path(override).expanduser() if override else None

    def install_steps(self) -> list[installer.InstallStep]:
        weights_dir = paths.engine_weights_dir(self.root, self.engine_id)
        venv = uvman.engine_venv_dir(self.root, self.engine_id)
        engine_dir = self._engine_dir()
        upstream = self._upstream_dir()

        def find_uv(log):
            return uvman.find_uv(self.root, env=self.env, log=log)

        def step_python(log, progress):
            uvman.ensure_python(find_uv(log), PYTHON_SPEC, env=self.env, log=log)

        def step_venv(log, progress):
            uvman.create_venv(find_uv(log), self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)

        def step_torch(log, progress):
            uv = find_uv(log)
            venv_ = uvman.create_venv(uv, self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)
            # macOS arm64 wheels from the index bundle MPS; pinned exactly
            # (ADR-0002 rule 1). No explicit wheel downloads.
            uvman.pip_install(
                uv, venv_,
                [f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}"],
                env=self.env, log=log,
            )

        def step_engine(log, progress):
            """Upstream source + patches + PoC-verified dependency set."""
            uv = find_uv(log)
            venv_ = uvman.create_venv(uv, self.root, self.engine_id, PYTHON_SPEC, env=self.env, log=log)
            # Dependencies verified as sufficient by the issue-#26 PoC:
            # no flash_attn (zero win_amd64 wheels / MPS-incompatible), no
            # fasttext (optional upstream dep), no torchcodec.
            uvman.pip_install(
                uv, venv_,
                [
                    f"transformers=={TRANSFORMERS_VERSION}",
                    f"einops=={EINOPS_VERSION}",
                    "regex", "wetext", "soundfile",
                ],
                env=self.env, log=log,
            )
            zip_path = engine_dir / "fireredtts3-main.zip"
            local_upstream = self._local_upstream_root()
            if local_upstream is not None:
                # Seed from an existing checkout (PoC tree, prior install)
                # before touching the network — same precedent as the MPS
                # local-weights reuse. The patch still runs, so a stale or
                # drifted local tree fails loudly at the anchors.
                log(f"engine: reusing local upstream checkout {local_upstream}")
                if upstream.exists():
                    shutil.rmtree(upstream)
                upstream.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(local_upstream, upstream)
            else:
                downloader.download_file(
                    downloader.DownloadSpec(path="main.zip", dest_name=zip_path.name),
                    zip_path.parent,
                    [
                        self.env.get("VOICECLONE_FIREREDTTS3_URL", ENGINE_PACKAGE_URL),
                        self.env.get("VOICECLONE_FIREREDTTS3_URL_FALLBACK", ENGINE_PACKAGE_URL_FALLBACK),
                    ],
                    progress=lambda name, done, total: progress(f"engine:{name}", done, total),
                    log=log,
                )
                if upstream.exists():
                    shutil.rmtree(upstream)
                upstream.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(upstream.parent)
                extracted = upstream.parent / "FireRedTTS3-main"
                extracted.rename(upstream)
            # Anchor-checked, idempotent, loud on drift (ADR-0019 decision 2).
            for line in apply_patches(upstream):
                log(f"fireredtts3 patch: {line}")

        def step_weights(log, progress):
            chain = sources.weight_sources(REPO, ms_repo=None, preferred=sources.weight_pref(self.env))
            local = self._local_weights_root()
            for rel in WEIGHTS:
                if local is not None and copy_from_local(local, rel, weights_dir / rel, log):
                    continue
                log(f"weights: downloading {rel}")
                downloader.download_file(
                    downloader.DownloadSpec(path=rel, dest_name=rel),
                    weights_dir, chain,
                    progress=lambda name, done, total, _n=rel: progress(f"weights:{_n}", done, total),
                    log=log,
                )
            log(f"weights: {len(WEIGHTS)} files ready in {weights_dir}")

        return [
            installer.InstallStep("python", f"安装 uv 托管的 Python {PYTHON_SPEC}", step_python),
            installer.InstallStep("venv", "创建引擎独立 venv", step_venv, artifact=venv),
            installer.InstallStep("torch", f"安装 torch/torchaudio {TORCH_VERSION}（PyPI，macOS arm64 自带 MPS）", step_torch, artifact=venv),
            installer.InstallStep("engine", "下载 FireRedTTS3 上游源码并打补丁（SDPA / MPS / 精度 / seed 守卫）", step_engine, artifact=venv),
            installer.InstallStep("weights", "准备 FireRedTTS3-Base 权重（约 12.3 GB）", step_weights),
        ]

    def install(self, log, progress=None) -> dict:
        return installer.run_install(self._ctx(), self.install_steps(), log, progress)

    # -- model dir (issue #17 local model management) -----------------------

    def model_dir(self):
        return self._engine_dir()

    # -- generation ---------------------------------------------------------

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        if not self.is_installed():
            raise RuntimeError(
                f"engine is not installed yet — call POST /engines/{self.engine_id}/install first"
            )
        params = request.params or {}
        if not params.get("ref_audio"):
            raise RuntimeError(
                "FireRedTTS3 是零样本复刻引擎，必须提供参考音频：请选择一个音色（或先创建音色）再生成"
            )
        if not (params.get("ref_text") or "").strip():
            raise RuntimeError(
                "FireRedTTS3 复刻需要参考文本：请为该音色提供参考音频的逐字转写（参考音频诊断可自动转写）"
            )
        out_dir = (self.output_dir or Path.cwd() / "data" / "audio").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav").resolve()

        from .worker_supervisor import WorkerSupervisor

        venv = paths.engine_venv_dir(self.root, self.engine_id)
        python = uvman.venv_python(venv)
        if not python.exists():
            raise RuntimeError(f"engine venv is broken (no python at {python}); reinstall the engine")
        worker = Path(__file__).with_name("fireredtts3_mps_worker.py")
        # The PoC measured 11.7 GB peak on MPS; a long-lived worker pinning
        # that much unified memory is worse than a per-process reload for a
        # cold engine, so the supervisor still applies idle recycle.
        supervisor = WorkerSupervisor(
            command=[str(python), "-u", str(worker)],
            cwd=str(self.root),
            idle_timeout_s=300.0,
            max_requests=20,
        )
        payload = {
            "action": "synthesize",
            "model_dir": str(paths.engine_weights_dir(self.root, self.engine_id)),
            "upstream_dir": str(self._upstream_dir()),
            "text": request.text,
            "ref_audio": str(Path(params["ref_audio"]).resolve()),
            "ref_text": params.get("ref_text"),
            "language": "Chinese",  # verified language; matrix records the rest as unverified
            "output": str(out_path),
        }
        if params.get("seed") not in (None, ""):
            payload["seed"] = params["seed"]
        try:
            result = supervisor.request(payload, on_log=log)
        finally:
            # One-shot supervisor per request: FireRedTTS3 loads in ~1 min
            # but pins ~12 GB; keep behavior predictable until the bf16
            # route (ADR-0019) changes the performance envelope.
            supervisor.unload()
        return GenerationResult(
            audio_path=result["audio_path"],
            sample_rate=result["sample_rate"],
            model_version=REPO,
            cost=0.0,
        )
