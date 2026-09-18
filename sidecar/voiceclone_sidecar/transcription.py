"""Transcription of reference samples (issue #8).

Two provider kinds, deliberately:

- ``local`` — a bundled-runtime tool in its OWN venv (never the sidecar's
  environment, same isolation rule as engines): macOS uses ``mlx-whisper``,
  Windows uses ``faster-whisper``. Weights download once through
  huggingface_hub with an hf-mirror fallback; after that transcription runs
  fully offline.
- cloud — ANY registered engine that exposes ``transcribe(audio_path, log)``.
  The app never adds a vendor of its own (issue #8): cloud transcription
  reuses whatever providers the user already plugged in as engines. Engines
  without a ``transcribe`` method are simply not offered.

The provider choice lives in ``<data>/settings.json`` so it survives restarts.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from .capabilities import Capabilities  # noqa: F401 - re-exported for typing clarity
from .runtime import installer, paths, uvman

LOCAL_TOOL_ID = "transcribe-local"
DEFAULT_PROVIDER = "local"

# macOS → mlx-whisper, Windows → faster-whisper (issue #8 mandate).
PLATFORM_CONFIG: dict[str, dict] = {
    "darwin": {
        "tool": "mlx-whisper",
        "packages": ["mlx-whisper>=0.3"],
        "model_repo": "mlx-community/whisper-small",
        "python_spec": "3.12",
        "label": "mlx-whisper（本地）",
    },
    "win32": {
        "tool": "faster-whisper",
        "packages": ["faster-whisper>=1.1"],
        "model_repo": "Systran/faster-whisper-small",
        "python_spec": "3.12",
        "label": "faster-whisper（本地）",
    },
}


def platform_config() -> dict | None:
    return PLATFORM_CONFIG.get(sys.platform)


class TranscriptionError(RuntimeError):
    """User-facing failure of a transcription request."""


class TranscriptionSettings:
    """Durable provider choice: local by default, cloud engines opt-in."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir) / "settings.json"
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def provider(self) -> str:
        with self._lock:
            return self._load().get("transcription_provider", DEFAULT_PROVIDER)

    def set_provider(self, provider: str) -> None:
        with self._lock:
            data = self._load()
            data["transcription_provider"] = provider
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)


# Snapshot script executed by the TRANSCRIBE venv's python (it has
# huggingface_hub as a dependency of the tool). HF first, hf-mirror fallback
# — same mirror order the engine weight downloads use.
_SNAPSHOT_SCRIPT = """
import json, sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1])
print("RESULT: " + json.dumps({"path": p}))
"""

class LocalTranscriber:
    """Bundled-runtime local ASR tool. Carries the InstallableEngine surface
    (is_installed/install_state/install) so the UI reuses the same install
    progress rendering as engines."""

    def __init__(self, root: Path | None = None, env: dict | None = None) -> None:
        self.env = env if env is not None else dict(_runtime_env())
        self.root = root if root is not None else paths.runtime_root(self.env)
        cfg = platform_config()
        if cfg is None:
            raise TranscriptionError(
                f"当前平台（{sys.platform}）暂不支持本地转写；请在设置中选择云端转写"
            )
        self.cfg = cfg

    def _ctx(self) -> installer.InstallContext:
        return installer.InstallContext(engine_id=LOCAL_TOOL_ID, root=self.root, env=self.env)

    # -- install surface -----------------------------------------------------

    def is_installed(self) -> bool:
        return bool(installer.load_state(self._ctx()).get("installed"))

    def install_state(self) -> dict:
        state = installer.load_state(self._ctx())
        return {"installed": bool(state.get("installed")), "steps": state.get("steps", {})}

    def weights_dir(self) -> Path:
        """Resolve the already-downloaded model directory, offline-only."""
        venv = uvman.engine_venv_dir(self.root, LOCAL_TOOL_ID)
        python = uvman.venv_python(venv)
        if not python.exists():
            raise TranscriptionError("本地转写工具的运行环境缺失；请重新安装")
        proc = subprocess.run(
            [str(python), "-c",
             "import json,sys;from huggingface_hub import snapshot_download;"
             "p=snapshot_download(sys.argv[1], local_files_only=True);"
             "print('RESULT: '+json.dumps({'path': p}))",
             self.cfg["model_repo"]],
            capture_output=True, text=True, timeout=60,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT: "):
                return Path(json.loads(line[len("RESULT: "):])["path"])
        raise TranscriptionError(
            "本地转写模型尚未下载完成；请先在设置中安装本地转写"
        )

    def install_steps(self) -> list[installer.InstallStep]:
        venv = uvman.engine_venv_dir(self.root, LOCAL_TOOL_ID)
        spec = self.cfg["python_spec"]

        def find_uv(log):
            return uvman.find_uv(self.root, env=self.env, log=log)

        def step_python(log, progress):
            uvman.ensure_python(find_uv(log), spec, env=self.env, log=log)

        def step_venv(log, progress):
            uvman.create_venv(find_uv(log), self.root, LOCAL_TOOL_ID, spec, env=self.env, log=log)

        def step_packages(log, progress):
            uv = find_uv(log)
            venv_ = uvman.create_venv(uv, self.root, LOCAL_TOOL_ID, spec, env=self.env, log=log)
            uvman.pip_install(uv, venv_, self.cfg["packages"], env=self.env, log=log)

        def step_weights(log, progress):
            uv = find_uv(log)
            venv_ = uvman.create_venv(uv, self.root, LOCAL_TOOL_ID, spec, env=self.env, log=log)
            python = uvman.venv_python(venv_)
            log(f"weights: downloading {self.cfg['model_repo']} (HF → hf-mirror)")

            import os

            def _snapshot(env_extra: dict | None = None) -> subprocess.CompletedProcess:
                env = {**os.environ, **(env_extra or {})}
                return subprocess.run(
                    [str(python), "-c", _SNAPSHOT_SCRIPT, self.cfg["model_repo"]],
                    capture_output=True, text=True, timeout=3600, env=env,
                )

            # The endpoint must be in the CHILD's environment BEFORE
            # huggingface_hub imports (its constants snapshot at import time),
            # so the mirror fallback is a second subprocess, not a re-set env.
            proc = _snapshot()
            if proc.returncode != 0:
                log("weights: huggingface.co failed; retrying via hf-mirror.com")
                proc = _snapshot({"HF_ENDPOINT": "https://hf-mirror.com"})
            if proc.returncode != 0:
                raise RuntimeError(
                    f"model download failed: {proc.stdout.strip()[-500:]} {proc.stderr.strip()[-500:]}"
                )
            log(f"weights: {self.cfg['model_repo']} ready")

        return [
            installer.InstallStep("python", f"安装 uv 托管的 Python {spec}", step_python),
            installer.InstallStep("venv", "创建转写工具独立环境", step_venv, artifact=venv),
            installer.InstallStep("packages", f"安装 {', '.join(self.cfg['packages'])}", step_packages, artifact=venv),
            installer.InstallStep("weights", f"下载转写模型 {self.cfg['model_repo']}", step_weights),
        ]

    def install(self, log, progress=None) -> dict:
        return installer.run_install(self._ctx(), self.install_steps(), log, progress)

    # -- transcription ---------------------------------------------------------

    def transcribe(self, audio_path: str, log) -> str:
        if not self.is_installed():
            raise TranscriptionError(
                "本地转写工具尚未安装；请在「转写设置」中先安装本地转写，"
                "或切换为云端转写"
            )
        venv = uvman.engine_venv_dir(self.root, LOCAL_TOOL_ID)
        python = uvman.venv_python(venv)
        if not python.exists():
            raise TranscriptionError("本地转写工具的运行环境缺失；请重新安装")
        weights = self.weights_dir()
        worker = Path(__file__).with_name("transcription_worker.py")
        log(f"transcribe: local ({self.cfg['tool']}) on {Path(audio_path).name}")
        proc = subprocess.run(
            [str(python), "-u", str(worker)],
            input=json.dumps({"audio": audio_path, "weights_dir": str(weights)}),
            capture_output=True, text=True, timeout=600,
        )
        text = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("RESULT: "):
                text = json.loads(line[len("RESULT: "):])["text"]
        if text is None:
            err = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise TranscriptionError(f"本地转写失败：{err or f'exit code {proc.returncode}'}")
        log(f"transcribe: {len(text)} chars")
        return text


def _runtime_env() -> dict:
    """China-friendly defaults, mirroring the engine installers."""
    return {
        "UV_PYTHON_INSTALL_MIRROR": "https://ghfast.top/https://github.com/indygreg/python-build-standalone/releases/download",
    }


def engine_transcribers(registry) -> list:
    """Registered engines that can transcribe (cloud providers, no new vendors)."""
    return [e for e in registry.list() if callable(getattr(e, "transcribe", None))]
