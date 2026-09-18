"""Engine listing + install lifecycle, and BYOK API key management (issue #9).

The BYOK contract never returns key values — only which engines have one
configured. Values go renderer -> sidecar -> OS key store and are then
dropped from memory beyond the read the engines make at call time.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..context import AppContext
from ..registry import InstallableEngine
from ..secrets import KeyStoreError


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    registry = ctx.registry
    keys = ctx.keys

    def _key_engine(engine_id: str):
        engine = ctx.require_engine(engine_id)
        if not engine.requires_key:
            raise HTTPException(
                status_code=422, detail=f"engine {engine.engine_id} does not use an API key"
            )
        return engine

    @router.get("/engines", dependencies=[Depends(ctx.require_auth)])
    async def engines() -> dict:
        return {
            "engines": [
                {
                    "id": e.engine_id,
                    "display_name": e.display_name,
                    "capabilities": e.capabilities().to_dict(),
                    "installed": e.is_installed()
                    if isinstance(e, InstallableEngine)
                    else True,
                    "requires_key": e.requires_key,
                    "key_configured": keys.get(e.engine_id) is not None
                    if e.requires_key
                    else None,
                    "billing_note": e.billing_note,
                    "data_usage_note": e.data_usage_note,
                    "params": [p.to_dict() for p in e.param_specs()],
                }
                for e in registry.list()
            ]
        }

    @router.get("/settings/keys", dependencies=[Depends(ctx.require_auth)])
    async def list_keys() -> dict:
        cloud = [e.engine_id for e in registry.list() if e.requires_key]
        return {"keys": keys.status(cloud)}

    @router.put("/settings/keys", dependencies=[Depends(ctx.require_auth)])
    async def set_key(body: dict) -> dict:
        engine_id = body.get("engine_id")
        key = body.get("key")
        if not isinstance(engine_id, str) or not isinstance(key, str):
            raise HTTPException(status_code=422, detail="engine_id and key are required")
        _key_engine(engine_id)
        if not key.strip():
            raise HTTPException(status_code=422, detail="API Key 不能为空")
        try:
            keys.set(engine_id, key)
        except KeyStoreError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"engine_id": engine_id, "configured": True}

    @router.delete("/settings/keys/{engine_id}", dependencies=[Depends(ctx.require_auth)])
    async def delete_key(engine_id: str) -> dict:
        _key_engine(engine_id)
        try:
            keys.delete(engine_id)
        except KeyStoreError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"engine_id": engine_id, "configured": False}

    @router.get("/engines/{engine_id}/status", dependencies=[Depends(ctx.require_auth)])
    async def engine_status(engine_id: str) -> dict:
        engine = ctx.require_engine(engine_id)
        if not isinstance(engine, InstallableEngine):
            return {"id": engine.engine_id, "installed": True, "installing": False, "steps": {}}
        state = engine.install_state()
        return {
            "id": engine.engine_id,
            "installed": state["installed"],
            "installing": engine_id in ctx.install_jobs,
            "steps": state["steps"],
        }

    @router.post("/engines/{engine_id}/install", dependencies=[Depends(ctx.require_auth)])
    async def install_engine(engine_id: str) -> dict:
        engine = ctx.require_engine(engine_id)
        if not isinstance(engine, InstallableEngine):
            return {"id": engine.engine_id, "status": "installed"}

        def work(log) -> None:
            def progress(name: str, done: int, total: int | None) -> None:
                if total:
                    pct = int(done * 100 / total)
                    log(f"{name}: {done}/{total} bytes ({pct}%)")
                else:
                    log(f"{name}: {done} bytes")

            engine.install(log, progress)

        return ctx.start_install(
            engine_id,
            work,
            conflict_detail="install already running for this engine",
        )

    return router
