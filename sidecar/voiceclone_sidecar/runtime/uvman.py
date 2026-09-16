"""uv-managed Python runtime and per-engine venvs (ADR-0002).

The app never requires a user-installed Python: ``uv`` bootstraps itself (or
ships bundled) and hosts a python-build-standalone interpreter, and each
engine gets its own venv so dependency sets never cross-contaminate.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from . import downloader
from .downloader import download_file
from .paths import engine_venv_dir
DEFAULT_PYTHON = "3.12"


class RuntimeCommandError(RuntimeError):
    """A uv subprocess failed. stderr is kept for the UI log."""


def _run(cmd: list[str], env: dict | None = None, log=None) -> str:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    if log:
        log(f"$ {' '.join(cmd)}")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=full_env, timeout=1800
    )
    if proc.returncode != 0:
        raise RuntimeCommandError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def uv_triple() -> str:
    machine = {"arm64": "aarch64", "x86_64": "x86_64"}.get(
        os.uname().machine, os.uname().machine
    )
    return f"{machine}-apple-darwin" if sys.platform == "darwin" else (
        f"{machine}-pc-windows-msvc" if sys.platform == "win32" else f"{machine}-unknown-linux-gnu"
    )


def find_uv(root: Path, env: dict | None = None, log=None) -> Path:
    """Locate uv: explicit env override > bundled/downloaded copy in root > PATH.

    Downloading hits the GitHub release only when uv is nowhere on the machine
    (first-run on a clean Mac). Mirrors can route it via
    ``VOICECLONE_UV_DOWNLOAD_URL``.
    """
    env = env if env is not None else os.environ
    override = env.get("VOICECLONE_UV_PATH", "")
    if override:
        p = Path(override)
        if p.is_file():
            return p
        raise FileNotFoundError(f"VOICECLONE_UV_PATH points nowhere: {override}")

    exe = "uv.exe" if sys.platform == "win32" else "uv"
    bundled = root / "bin" / exe
    if bundled.is_file():
        return bundled

    on_path = shutil.which("uv")
    if on_path:
        return Path(on_path)

    root.mkdir(parents=True, exist_ok=True)
    archive_name = f"uv-{uv_triple()}.tar.gz"
    url = env.get(
        "VOICECLONE_UV_DOWNLOAD_URL",
        f"https://github.com/astral-sh/uv/releases/latest/download/{archive_name}",
    )
    if log:
        log(f"uv not found on this machine; downloading from {url}")
    # Source templates may omit {path} entirely: a full URL passes through
    # str.format(path="") unchanged.
    archive = download_file(
        downloader.DownloadSpec(path="", dest_name=archive_name),
        root / "tmp",
        sources=[url],
        log=log,
    )
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if member.name.endswith(f"/{exe}"):
                member.name = exe
                tf.extract(member, root / "bin", filter="data")
                break
        else:
            raise RuntimeError(f"uv binary not found inside {archive}")
    (root / "bin" / exe).chmod(0o755)
    archive.unlink()
    return root / "bin" / exe


def ensure_python(uv: Path, python_spec: str = DEFAULT_PYTHON, env: dict | None = None, log=None) -> None:
    """Install the uv-hosted interpreter. Idempotent."""
    _run([str(uv), "python", "install", python_spec], env=env, log=log)


def create_venv(uv: Path, root: Path, engine_id: str, python_spec: str = DEFAULT_PYTHON, env: dict | None = None, log=None) -> Path:
    """Create the engine's isolated venv.

    A half-created venv from a previous failed run is deleted first: a venv
    whose python symlink or site-packages is broken would otherwise wedge
    every later step (ADR-0002's "half-finished .venv makes retries fail
    forever" trap).
    """
    venv = engine_venv_dir(root, engine_id)
    python_bin = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if venv.exists() and not python_bin.exists():
        if log:
            log(f"removing broken half-created venv at {venv}")
        shutil.rmtree(venv)
    if not venv.exists():
        _run([str(uv), "venv", str(venv), "--python", python_spec], env=env, log=log)
    return venv


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def pip_install(uv: Path, venv: Path, packages: list[str], env: dict | None = None, log=None) -> None:
    """Install pinned packages into the engine venv.

    Uses ``uv pip install --python <venv>`` — never ``--extra-index-url``
    anywhere (ADR-0002 rule 1): index/mirror overrides go through
    ``UV_DEFAULT_INDEX`` / ``UV_PYTHON_INSTALL_MIRROR`` env vars.
    """
    _run([str(uv), "pip", "install", "--python", str(venv_python(venv)), *packages], env=env, log=log)
