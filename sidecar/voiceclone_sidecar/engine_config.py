"""Injected engine config seam (ADR-0015).

Engines never read ``os.environ`` themselves. The registry builds one
:class:`EngineConfig` per engine and injects it at construction time, so a
config value can come from three layers, most specific wins:

1. **built-in defaults** — the China-friendly fallbacks that used to live as
   per-engine ``_runtime_env()`` copies;
2. **process environment** — ``VOICECLONE_*`` and ``UV_*`` variables the user
   exported (the P0-2 fix: the old literal-dict default *reversed* the user's
   ``UV_DEFAULT_INDEX`` because ``uvman._run`` merges ``env`` over
   ``os.environ.copy()``; here the user's value lands inside ``env`` itself);
3. **settings overrides** — per-engine values from the settings storage
   (``<data>/settings.json``, the SAME store the transcription-provider and
   UI-preference settings live in), which win over both so a settings change
   can take effect without touching the environment.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .storage import write_json_atomic

SETTINGS_KEY = "engines"
_SETTINGS_LOCK = threading.Lock()

# Keys the engine config layer concerns itself with. Everything else in the
# process environment is deliberately NOT forwarded: engines get exactly the
# download/runtime knobs they declare, not the caller's whole environment.
CONFIG_ENV_PREFIXES = ("VOICECLONE_", "UV_")

# China-friendly fallbacks (formerly duplicated as _runtime_env() in four
# modules). Lowest precedence: the user's environment and the settings store
# both override these.
BUILTIN_DEFAULTS: dict[str, str] = {
    "UV_PYTHON_INSTALL_MIRROR": (
        "https://ghfast.top/https://github.com/indygreg/python-build-standalone/releases/download"
    ),
    # PyPI index for non-torch engine packages. ADR-0002: the CUDA torch
    # wheels never resolve through this index — they use explicit wheel URLs
    # (see :mod:`voiceclone_sidecar.sources`), so a mirror index here can
    # never downgrade CUDA torch to a CPU build.
    "UV_DEFAULT_INDEX": "https://mirrors.aliyun.com/pypi/simple/",
}


@dataclass(frozen=True)
class EngineConfig:
    """Everything one engine needs besides its own identity.

    ``env`` is the effective environment for install/runtime subprocesses
    (already merged through :func:`effective_env`). ``settings`` carries the
    engine-level non-env overrides from settings storage (future use: the
    download-source picker of ADR-0016).
    """

    engine_id: str = ""
    output_dir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    settings: dict = field(default_factory=dict)


def effective_env(
    engine_id: str,
    process_env: dict | None = None,
    settings: dict | None = None,
) -> dict[str, str]:
    """Merge built-in defaults < process env < per-engine settings.

    Only ``VOICECLONE_*`` / ``UV_*`` keys are taken from the process
    environment; a settings ``{"env": {...}}`` block for this engine wins
    over both.
    """
    process_env = process_env if process_env is not None else os.environ
    env: dict[str, str] = dict(BUILTIN_DEFAULTS)
    for key, value in process_env.items():
        if key.startswith(CONFIG_ENV_PREFIXES) and value:
            env[key] = value
    engine_settings = (settings or {}).get(SETTINGS_KEY, {}).get(engine_id, {})
    for key, value in (engine_settings.get("env") or {}).items():
        if value:
            env[key] = str(value)
    return env


def resolve_seam(
    config: EngineConfig | None,
    engine_id: str,
    output_dir: Path | None,
    env: dict | None,
) -> tuple[Path | None, dict[str, str]]:
    """Resolve the ``(output_dir, env)`` constructor defaults for one engine.

    The single shared implementation of the constructor seam (ADR-0015): an
    explicitly passed argument wins, else the injected config supplies it,
    else — bare construction, for tests and scripts only — the same
    defaults < process env merge the registry performs.
    """
    if output_dir is None and config is not None:
        output_dir = config.output_dir
    if env is None:
        env = dict(config.env) if config is not None else effective_env(engine_id)
    return output_dir, env


def _settings_path(data_dir: Path | None) -> Path | None:
    return Path(data_dir) / "settings.json" if data_dir is not None else None


def load_engine_settings(data_dir: Path | None) -> dict:
    """Read the ``engines`` block from the settings storage.

    Shape: ``{"engines": {"<engine_id>": {"env": {...}}}}`` — one store shared
    with the other app settings. A missing, empty or corrupt file is simply no
    overrides: the seam must never make the app unbootable.
    """
    path = _settings_path(data_dir)
    if path is None:
        return {SETTINGS_KEY: {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {SETTINGS_KEY: {}}
    if not isinstance(raw, dict):
        return {SETTINGS_KEY: {}}
    engines = raw.get(SETTINGS_KEY)
    if not isinstance(engines, dict):
        return {SETTINGS_KEY: {}}
    return {SETTINGS_KEY: {k: v for k, v in engines.items() if isinstance(v, dict)}}


def save_engine_settings(data_dir: Path | None, engines: dict) -> dict:
    """Write the ``engines`` block into the settings storage, preserving the
    other keys in that file. Returns the normalized ``{"engines": {...}}``."""
    path = _settings_path(data_dir)
    if path is None:
        raise ValueError("no data dir configured for settings storage")
    with _SETTINGS_LOCK:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        clean = {
            engine_id: {
                "env": {str(k): str(v) for k, v in (spec.get("env") or {}).items()},
            }
            for engine_id, spec in engines.items()
            if isinstance(spec, dict)
        }
        raw[SETTINGS_KEY] = clean
        write_json_atomic(path, raw)
    return {SETTINGS_KEY: clean}
