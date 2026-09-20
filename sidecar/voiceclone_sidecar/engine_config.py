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
   (the ``settings`` table of ``<data>/library.db``, ADR-0017 — the SAME
   store the transcription-provider and download-source settings live in),
   which win over both so a settings change can take effect without touching
   the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .db import T_SETTINGS, read_kv_block, write_kv_block

SETTINGS_KEY = "engines"
SOURCES_KEY = "sources"

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
    # ADR-0016: the global download-source preferences (settings ``sources``
    # block) sit between the process environment and per-engine overrides —
    # the picker must take effect without touching the environment, while a
    # per-engine ``env`` entry still fine-tunes one engine.
    for key, value in _source_pref_env((settings or {}).get(SOURCES_KEY, {})).items():
        env[key] = value
    engine_settings = (settings or {}).get(SETTINGS_KEY, {}).get(engine_id, {})
    for key, value in (engine_settings.get("env") or {}).items():
        if value:
            env[key] = str(value)
    return env


# The download-source axes (ADR-0016), mapped onto the sources layer's
# catalogs lazily (import cycle avoidance). Single source of truth for the
# prefs validator and the env mapping below.
SOURCE_AXES = ("weights", "pypi", "cuda")


def _axis_choices(axis: str) -> dict:
    from . import sources as _sources

    return {
        "weights": _sources.WEIGHT_SOURCE_CHOICES,
        "pypi": _sources.PYPI_CHOICES,
        "cuda": _sources.CUDA_CHOICES,
    }[axis]


def normalize_source_prefs(prefs: dict) -> dict[str, str]:
    """Keep only valid axis values; unknown/invalid entries are dropped so a
    hand-edited settings file can never poison the install environment."""
    return {
        axis: value
        for axis in SOURCE_AXES
        if (value := prefs.get(axis)) in _axis_choices(axis)
    }


def _source_pref_env(prefs: dict) -> dict[str, str]:
    """Map the validated ``sources`` prefs onto the download-source env knobs.

    Unknown keys/values are ignored (defaults apply) — a hand-edited settings
    file must never poison the install environment. Only VALID choices are
    forwarded; the sources layer's catalogs are the single source of truth.
    """
    from . import sources as _sources

    prefs = normalize_source_prefs(prefs)
    out: dict[str, str] = {}
    if "weights" in prefs:
        out[_sources.PREFERRED_WEIGHT_ENV] = prefs["weights"]
    if "pypi" in prefs:
        out["UV_DEFAULT_INDEX"] = _sources.PYPI_CHOICES[prefs["pypi"]]["index"]
    if "cuda" in prefs:
        out[_sources.PREFERRED_CUDA_ENV] = prefs["cuda"]
    return out


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


def load_full_settings(data_dir: Path | None) -> dict:
    """The FULL settings blocks the registry seam consumes.

    ``{"engines": {...}, "sources": {...}}`` — callers building engine
    configs must use this (not :func:`load_engine_settings`, engines-only) so the global
    download-source preferences (ADR-0016) reach :func:`effective_env`.
    """
    engines = read_kv_block(data_dir, T_SETTINGS, SETTINGS_KEY)
    return {
        SETTINGS_KEY: engines if isinstance(engines, dict) else {},
        SOURCES_KEY: load_source_prefs(data_dir),
    }


def load_engine_settings(data_dir: Path | None) -> dict:
    """Read the ``engines`` block from the settings storage.

    Shape: ``{"engines": {"<engine_id>": {"env": {...}}}}`` — one store shared
    with the other app settings. A missing store is simply no overrides: the
    seam must never make the app unbootable (a corrupt database degrades the
    same way the corrupt JSON file used to).
    """
    if data_dir is None:
        return {SETTINGS_KEY: {}}
    try:
        engines = read_kv_block(data_dir, T_SETTINGS, SETTINGS_KEY)
    except Exception:  # noqa: BLE001 - a corrupt store must never block boot; degrade to no overrides
        return {SETTINGS_KEY: {}}
    if not isinstance(engines, dict):
        return {SETTINGS_KEY: {}}
    return {SETTINGS_KEY: {k: v for k, v in engines.items() if isinstance(v, dict)}}


def save_engine_settings(data_dir: Path | None, engines: dict) -> dict:
    """Write the ``engines`` block into the settings storage. Other settings
    keys are separate rows and untouched. Returns the normalized
    ``{"engines": {...}}``."""
    if data_dir is None:
        raise ValueError("no data dir configured for settings storage")
    clean = {
        engine_id: {
            "env": {str(k): str(v) for k, v in (spec.get("env") or {}).items()},
        }
        for engine_id, spec in engines.items()
        if isinstance(spec, dict)
    }
    write_kv_block(data_dir, T_SETTINGS, SETTINGS_KEY, clean)
    return {SETTINGS_KEY: clean}


def load_source_prefs(data_dir: Path | None) -> dict:
    """Read the download-source preferences (ADR-0016 ``sources`` block).

    Shape: ``{"weights": "hf"|"hf-mirror"|"modelscope",
    "pypi": "aliyun"|"official", "cuda": "official"|"aliyun"}``. Unknown or
    missing values fall back to the defaults from :mod:`.sources` — a corrupt
    store degrades to the built-in China-friendly chains, never to an error.
    """
    from . import sources as _sources

    defaults = {
        "weights": _sources.DEFAULT_WEIGHT_SOURCE,
        "pypi": _sources.DEFAULT_PYPI_SOURCE,
        "cuda": _sources.DEFAULT_CUDA_SOURCE,
    }
    if data_dir is None:
        return defaults
    try:
        block = read_kv_block(data_dir, T_SETTINGS, SOURCES_KEY)
    except Exception:  # noqa: BLE001 - a corrupt store degrades to the built-in source chains
        return defaults
    if not isinstance(block, dict):
        return defaults
    return {**defaults, **normalize_source_prefs(block)}


def save_source_prefs(data_dir: Path | None, prefs: dict) -> dict:
    """Persist the download-source preferences. Callers validate; unknown
    keys are dropped. Returns the normalized prefs."""
    if data_dir is None:
        raise ValueError("no data dir configured for settings storage")
    write_kv_block(data_dir, T_SETTINGS, SOURCES_KEY, normalize_source_prefs(prefs))
    return load_source_prefs(data_dir)
