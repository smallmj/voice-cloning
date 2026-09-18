"""UI preferences (issue #21): theme and engine-selection persistence.

ADR-0014 places theme choice in the sidecar's settings storage (a property of
the library, like consent — ADR-0012), and ADR-0013 requires the engine
selection to survive a restart instead of living as renderer component state.
Both ride on one small, validated key/value endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..context import AppContext

ALLOWED_THEMES = ("system", "dark", "light")


def _sanitize(raw: dict) -> dict:
    """Coerce stored values into the public shape; unknown/corrupt values
    fall back to defaults instead of leaking internals."""
    theme = raw.get("theme")
    engine_id = raw.get("engine_id")
    return {
        "theme": theme if theme in ALLOWED_THEMES else "system",
        "engine_id": engine_id if isinstance(engine_id, str) and engine_id else None,
    }


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
        if not update:
            raise HTTPException(status_code=422, detail="至少提供 theme 或 engine_id 之一")
        ctx.write_ui_prefs(update)
        return _sanitize(ctx.read_ui_prefs())

    return router
