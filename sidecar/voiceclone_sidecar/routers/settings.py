"""UI preferences (issue #21) and per-engine config (issue #22 / ADR-0015).

ADR-0014 places theme choice in the sidecar's settings storage (a property of
the library, like consent — ADR-0012), and ADR-0013 requires the engine
selection to survive a restart instead of living as renderer component state.
Both ride on one small, validated key/value endpoint. Issue #22 adds the
engine-config block of the SAME store (``engines``): per-engine ``env``
overrides the registry injects into every engine, so the settings storage is
the producer side of the ADR-0015 seam — a PUT rebuilds the registry, which
makes the change take effect without restarting the sidecar.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException

from .. import engine_config, sources
from ..context import AppContext

ALLOWED_THEMES = ("system", "dark", "light")

# Issue #36: the shared right sidebar's persisted geometry. The range matches
# the drag clamp on the renderer side; stored values outside it are clamped.
SIDEBAR_MIN_WIDTH = 260
SIDEBAR_MAX_WIDTH = 640
SIDEBAR_DEFAULT_WIDTH = 360

# Issue #44: software-level display and log-behavior settings. The clamps
# mirror the renderer's controls so a corrupt stored value falls back cleanly.
UI_SCALE_MIN = 0.85
UI_SCALE_MAX = 1.5
FONT_SIZE_MIN = 11
FONT_SIZE_MAX = 24
LOG_BUFFER_MIN = 100
LOG_BUFFER_MAX = 5000
LOG_BUFFER_DEFAULT = 500

# Issue #65 (spec #62 / ADR-0020): 更新渠道 (Update Channel) modes, validated
# against the same set the Electron updater module uses.
ALLOWED_UPDATE_CHANNEL_MODES = ("auto", "official", "mirror")
DEFAULT_UPDATE_MIRROR_PREFIX = "https://gh-proxy.com/"

# PR #63 review fix: the 镜像前缀 is spliced in front of GitHub API/asset URLs
# by the Electron updater, so it must stay `https://` + host (+ optional
# path). file://, http://, scheme-relative or whitespace values are rejected
# (PUT → 422); corrupt stored values degrade to the default on read. The
# same regex lives in app/electron/updater.ts (sanitizeMirrorPrefix).
MIRROR_PREFIX_PATTERN = r"https://[^/\s#?]+(/[^\s#?]*)?"


def _sanitize_mirror_prefix(value) -> str:
    """Validate + normalize a mirror prefix; falls back to the default for
    anything that is not a valid https://host[/path] string."""
    if not isinstance(value, str):
        return DEFAULT_UPDATE_MIRROR_PREFIX
    trimmed = value.strip()
    if not re.fullmatch(MIRROR_PREFIX_PATTERN, trimmed):
        return DEFAULT_UPDATE_MIRROR_PREFIX
    return trimmed.rstrip("/") + "/"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _clamp(value, lo, hi, default):
    if not _is_number(value):
        return default
    return max(lo, min(hi, float(value)))


def _sanitize(raw: dict) -> dict:
    """Coerce stored values into the public shape; unknown/corrupt values
    fall back to defaults instead of leaking internals."""
    theme = raw.get("theme")
    engine_id = raw.get("engine_id")
    return {
        "theme": theme if theme in ALLOWED_THEMES else "system",
        "engine_id": engine_id if isinstance(engine_id, str) and engine_id else None,
        # Issue #23 / ADR-0018 decision 6: per-engine last-used generation
        # parameters. Values are stored as strings (the renderer's state is
        # stringly typed); the engine spec validates/converts at generate.
        "engine_params": _sanitize_engine_params(raw.get("engine_params")),
        # Issue #36: shared right sidebar (logs + job queue) state.
        "sidebar_open": raw.get("sidebar_open", True) is True,
        "sidebar_width": _clamp_sidebar_width(raw.get("sidebar_width")),
        # Issue #44: UI scale / font-size override / log buffer count. A
        # font_size of None means "no override" (the theme default applies).
        "ui_scale": _clamp(raw.get("ui_scale"), UI_SCALE_MIN, UI_SCALE_MAX, 1.0),
        "font_size": _clamp_int_or_none(raw.get("font_size"), FONT_SIZE_MIN, FONT_SIZE_MAX),
        "log_buffer": int(
            _clamp(raw.get("log_buffer"), LOG_BUFFER_MIN, LOG_BUFFER_MAX, LOG_BUFFER_DEFAULT)
        ),
        # Issue #65 (spec #62 / ADR-0020): updater preferences ride the SAME
        # settings store as every other software-level preference, so the
        # main process and the renderer read one source of truth. Validation
        # mirrors electron/updater.ts: corrupt values degrade to defaults.
        "updater_channel_mode": (
            raw.get("updater_channel_mode")
            if raw.get("updater_channel_mode") in ALLOWED_UPDATE_CHANNEL_MODES
            else "auto"
        ),
        "updater_mirror_prefix": _sanitize_mirror_prefix(raw.get("updater_mirror_prefix")),
        # 跳过此版本: a stored tag is kept only when a non-empty string.
        "updater_skipped_tag": (
            raw.get("updater_skipped_tag")
            if isinstance(raw.get("updater_skipped_tag"), str) and raw.get("updater_skipped_tag")
            else None
        ),
    }


def _clamp_int_or_none(value, lo: int, hi: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return max(lo, min(hi, value))


def _clamp_sidebar_width(value) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return SIDEBAR_DEFAULT_WIDTH
    return max(SIDEBAR_MIN_WIDTH, min(SIDEBAR_MAX_WIDTH, value))


def _sanitize_engine_params(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    clean: dict = {}
    for engine_id, params in raw.items():
        if isinstance(engine_id, str) and isinstance(params, dict):
            clean[engine_id] = {
                k: str(v) for k, v in params.items() if isinstance(k, str)
            }
    return clean


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()

    @router.get("/settings/ui", dependencies=[Depends(ctx.require_auth)])
    async def get_ui_prefs() -> dict:
        return _sanitize(ctx.read_ui_prefs())

    @router.put("/settings/ui", dependencies=[Depends(ctx.require_auth)])
    async def put_ui_prefs(body: dict) -> dict:
        update: dict = {}
        if "theme" in body:
            theme = body["theme"]
            if theme not in ALLOWED_THEMES:
                raise HTTPException(
                    status_code=422,
                    detail=f"theme 必须是 {'/'.join(ALLOWED_THEMES)} 之一",
                )
            update["theme"] = theme
        if "engine_id" in body:
            engine_id = body["engine_id"]
            if engine_id is not None and (not isinstance(engine_id, str) or not engine_id):
                raise HTTPException(status_code=422, detail="engine_id 必须为非空字符串或 null")
            update["engine_id"] = engine_id
        if "engine_params" in body:
            raw = body["engine_params"]
            if raw is not None and not isinstance(raw, dict):
                raise HTTPException(
                    status_code=422, detail="engine_params 必须是「引擎 → 参数名 → 值」的对象"
                )
            update["engine_params"] = _sanitize_engine_params(raw)
        if "sidebar_open" in body:
            # Issue #36: explicit boolean — a truthy string is a corrupt
            # payload, not a preference.
            if not isinstance(body["sidebar_open"], bool):
                raise HTTPException(status_code=422, detail="sidebar_open 必须是布尔值")
            update["sidebar_open"] = body["sidebar_open"]
        if "sidebar_width" in body:
            width = body["sidebar_width"]
            if not isinstance(width, int) or isinstance(width, bool):
                raise HTTPException(status_code=422, detail="sidebar_width 必须是整数")
            update["sidebar_width"] = _clamp_sidebar_width(width)
        if "ui_scale" in body:
            # Issue #44: a float scale; booleans are corrupt payloads.
            if not _is_number(body["ui_scale"]):
                raise HTTPException(status_code=422, detail="ui_scale 必须是数字")
            update["ui_scale"] = _clamp(
                body["ui_scale"], UI_SCALE_MIN, UI_SCALE_MAX, 1.0
            )
        if "font_size" in body:
            size = body["font_size"]
            if size is not None and (not isinstance(size, int) or isinstance(size, bool)):
                raise HTTPException(status_code=422, detail="font_size 必须是整数或 null")
            update["font_size"] = (
                None
                if size is None
                else min(FONT_SIZE_MAX, max(FONT_SIZE_MIN, size))
            )
        if "log_buffer" in body:
            buf = body["log_buffer"]
            if not isinstance(buf, int) or isinstance(buf, bool):
                raise HTTPException(status_code=422, detail="log_buffer 必须是整数")
            update["log_buffer"] = int(
                _clamp(buf, LOG_BUFFER_MIN, LOG_BUFFER_MAX, LOG_BUFFER_DEFAULT)
            )
        # Issue #65: updater preferences. channel_mode is one of the three
        # 更新渠道 modes; mirror_prefix is the editable 镜像前缀 (must be a
        # string — empty is allowed and degrades to the auto chain);
        # skipped_tag persists 跳过此版本 (string or null).
        if "updater_channel_mode" in body:
            mode = body["updater_channel_mode"]
            if mode not in ALLOWED_UPDATE_CHANNEL_MODES:
                raise HTTPException(
                    status_code=422,
                    detail=f"updater_channel_mode 必须是 {'/'.join(ALLOWED_UPDATE_CHANNEL_MODES)} 之一",
                )
            update["updater_channel_mode"] = mode
        if "updater_mirror_prefix" in body:
            prefix = body["updater_mirror_prefix"]
            # PR #63 review fix: must be https://host[/path] — the updater
            # splices this value in front of GitHub URLs, so anything else
            # (file://, http://, whitespace, scheme-relative) is refused and
            # never persisted. An emptied field stores the default (the
            # renderer's 「镜像不可编辑为空」 rule); a valid value is
            # normalized to a single trailing slash.
            if not isinstance(prefix, str):
                raise HTTPException(status_code=422, detail="updater_mirror_prefix 必须是字符串")
            trimmed = prefix.strip()
            if trimmed == "":
                update["updater_mirror_prefix"] = DEFAULT_UPDATE_MIRROR_PREFIX
            elif re.fullmatch(MIRROR_PREFIX_PATTERN, trimmed) is None:
                raise HTTPException(
                    status_code=422,
                    detail="updater_mirror_prefix 必须是 https:// 开头的镜像地址（如 https://gh-proxy.com/）",
                )
            else:
                update["updater_mirror_prefix"] = trimmed.rstrip("/") + "/"
        if "updater_skipped_tag" in body:
            tag = body["updater_skipped_tag"]
            if tag is not None and (not isinstance(tag, str) or not tag):
                raise HTTPException(
                    status_code=422, detail="updater_skipped_tag 必须为非空字符串或 null"
                )
            update["updater_skipped_tag"] = tag
        if not update:
            raise HTTPException(
                status_code=422,
                detail="至少提供 theme、engine_id、engine_params、sidebar_open、sidebar_width、ui_scale、font_size、log_buffer 或 updater_* 之一",
            )
        ctx.write_ui_prefs(update)
        return _sanitize(ctx.read_ui_prefs())

    @router.get("/settings/engines", dependencies=[Depends(ctx.require_auth)])
    async def get_engine_config() -> dict:
        return engine_config.load_engine_settings(ctx.data_root)

    @router.put("/settings/engines", dependencies=[Depends(ctx.require_auth)])
    async def put_engine_config(body: dict) -> dict:
        engines = body.get("engines")
        if not isinstance(engines, dict):
            raise HTTPException(status_code=422, detail="body 必须包含 engines 对象")
        for engine_id, spec in engines.items():
            if engine_id not in {e.engine_id for e in ctx.registry.list()}:
                raise HTTPException(status_code=422, detail=f"未知引擎：{engine_id}")
            if not isinstance(spec, dict):
                raise HTTPException(status_code=422, detail=f"{engine_id} 的配置必须是对象")
            env = spec.get("env")
            if env is not None and (
                not isinstance(env, dict)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
            ):
                raise HTTPException(
                    status_code=422, detail=f"{engine_id} 的 env 必须是字符串到字符串的映射"
                )
        saved = engine_config.save_engine_settings(ctx.data_root, engines)
        # ADR-0015 decision 1: config is injected at construction, so the
        # change takes effect by rebuilding the registry — no sidecar restart.
        # ADR-0016: the rebuild carries the full settings (engines + sources)
        # so the download-source preferences stay in effect.
        ctx.rebuild_registry()
        return saved

    @router.get("/settings/sources", dependencies=[Depends(ctx.require_auth)])
    async def get_source_prefs() -> dict:
        """Download-source preferences (ADR-0016): one 首选 per axis, plus
        the choice catalogs the picker renders."""
        return {
            "sources": engine_config.load_source_prefs(ctx.data_root),
            "choices": {
                "weights": [
                    {"id": sid, "label": spec["label"]}
                    for sid, spec in sources.WEIGHT_SOURCE_CHOICES.items()
                ],
                "pypi": [
                    {"id": sid, "label": spec["label"]}
                    for sid, spec in sources.PYPI_CHOICES.items()
                ],
                "cuda": [
                    {"id": sid, "label": spec["label"]}
                    for sid, spec in sources.CUDA_CHOICES.items()
                ],
            },
        }

    @router.put("/settings/sources", dependencies=[Depends(ctx.require_auth)])
    async def put_source_prefs(body: dict) -> dict:
        prefs = body.get("sources")
        if not isinstance(prefs, dict):
            raise HTTPException(status_code=422, detail="body 必须包含 sources 对象")
        labels = {
            "weights": ("权重源", sources.WEIGHT_SOURCE_CHOICES),
            "pypi": ("包索引", sources.PYPI_CHOICES),
            "cuda": ("CUDA 轮子源", sources.CUDA_CHOICES),
        }
        for key, (cn, choices) in labels.items():
            if key in prefs and prefs[key] not in choices:
                raise HTTPException(
                    status_code=422,
                    detail=f"{cn}必须是 {'/'.join(choices)} 之一",
                )
        engine_config.save_source_prefs(ctx.data_root, prefs)
        # Same seam as PUT /settings/engines: prefs map onto env knobs inside
        # effective_env, so a registry rebuild makes them take effect.
        ctx.rebuild_registry()
        return engine_config.load_source_prefs(ctx.data_root)

    return router
