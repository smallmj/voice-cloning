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

from fastapi import APIRouter, Depends, HTTPException

from .. import engine_config, sources
from ..context import AppContext

ALLOWED_THEMES = ("system", "dark", "light")

# Issue #36: the shared right sidebar's persisted geometry. The range matches
# the drag clamp on the renderer side; stored values outside it are clamped.
SIDEBAR_MIN_WIDTH = 260
SIDEBAR_MAX_WIDTH = 640
SIDEBAR_DEFAULT_WIDTH = 360


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
    }


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
        if not update:
            raise HTTPException(status_code=422, detail="至少提供 theme、engine_id、engine_params、sidebar_open 或 sidebar_width 之一")
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
