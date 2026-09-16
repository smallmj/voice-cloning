"""Runtime paths. Everything the bundled runtime installs must live under a
pure-ASCII root (ADR-0002 rule 2): non-ASCII or space-ridden paths have been
observed to corrupt ninja-generated build files and silently degrade CUDA
kernel compilation."""

from __future__ import annotations

import os
import sys
from pathlib import Path


class PathNotAsciiError(RuntimeError):
    """The chosen runtime root is not pure ASCII — refuse to install there."""


def validate_ascii_root(root: Path) -> Path:
    text = str(root)
    if not text.isascii():
        raise PathNotAsciiError(
            f"runtime root must be pure ASCII, got: {text!r}. "
            "Set VOICECLONE_RUNTIME_ROOT to an ASCII-only location."
        )
    return root


def default_runtime_root() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "VoiceClone" / "runtime"


def runtime_root(env: dict | None = None) -> Path:
    """Resolve and validate the runtime installation root.

    ``VOICECLONE_RUNTIME_ROOT`` overrides the platform default. The result is
    validated as pure ASCII before anything is created under it.
    """
    env = env if env is not None else os.environ
    override = env.get("VOICECLONE_RUNTIME_ROOT", "")
    root = Path(override) if override else default_runtime_root()
    return validate_ascii_root(root)


def engine_dir(root: Path, engine_id: str) -> Path:
    """One directory per engine: its venv and weights never mix with others."""
    return root / "engines" / engine_id


def engine_venv_dir(root: Path, engine_id: str) -> Path:
    return engine_dir(root, engine_id) / "venv"


def engine_weights_dir(root: Path, engine_id: str) -> Path:
    return engine_dir(root, engine_id) / "weights"
