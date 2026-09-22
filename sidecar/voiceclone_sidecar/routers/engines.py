"""Engine listing + install lifecycle, and BYOK API key management (issue #9).

The BYOK contract never returns key values — only which engines have one
configured. Values go renderer -> sidecar -> OS key store and are then
dropped from memory beyond the read the engines make at call time.
"""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, HTTPException

from ..context import AppContext
from ..key_store import KeyStoreError
from ..registry import InstallableEngine
from ..transcription import LOCAL_TOOL_ID


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
                    "installed": e.is_installed() if isinstance(e, InstallableEngine) else True,
                    "requires_key": e.requires_key,
                    "key_configured": keys.get(e.engine_id) is not None if e.requires_key else None,
                    # Vendor identity (ADR-0015 decision 3): the key UI groups
                    # engines by vendor so one vendor's key is entered once.
                    "vendor_label": getattr(e, "vendor_label", None) if e.requires_key else None,
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
            return {
                "id": engine.engine_id,
                "installed": True,
                "installing": False,
                "steps": {},
                "progress": None,
            }
        state = engine.install_state()
        installing = engine_id in ctx.install_jobs
        return {
            "id": engine.engine_id,
            "installed": state["installed"],
            "installing": installing,
            "steps": state["steps"],
            # Byte-level download progress (issue #35). Gated on an actually
            # running install so a stale field from a killed run (crash
            # between writes) is never advertised as a live progress bar.
            "progress": state.get("progress") if installing else None,
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

    # -- local model management (issue #17, ADR-0016 「引擎」区) --------------

    def _model_target(engine_id: str):
        """The installable object behind a model-management id."""
        if engine_id == LOCAL_TOOL_ID:
            try:
                return ctx.local_transcriber()
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"当前平台不支持本地转写：{exc}",
                ) from exc
        return ctx.require_engine(engine_id)

    def _model_entry(target, engine_id: str, display_name: str) -> dict:
        d = target.model_dir()
        usage = sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and not f.is_symlink())
        try:
            weights_dir = (
                str(target.weights_dir()) if hasattr(target, "weights_dir") else str(d / "weights")
            )
        except Exception:  # noqa: BLE001 - listing must never fail
            weights_dir = str(d / "weights")
        return {
            "id": engine_id,
            "display_name": display_name,
            "installed": bool(target.is_installed()),
            "model_dir": str(d),
            "weights_dir": weights_dir,
            "disk_usage_bytes": usage,
            "installing": engine_id in ctx.install_jobs,
        }

    @router.get("/engines/models", dependencies=[Depends(ctx.require_auth)])
    async def local_models() -> dict:
        """Installed local models: weight dir, disk usage, install state.

        Reads ``ctx.registry`` (not the captured one) so the listing follows
        a settings-driven registry rebuild.

        The transcription tool is deliberately NOT listed here (issue #42
        follow-up): it has its own card on the engine page with install AND
        uninstall, and surfacing it a second time as an always-gray
        「mlx-whisper（本地）· 未安装 · 卸载」 row in the settings page's
        local-model list only confused users.
        """
        from ..runtime import paths

        items: list[dict] = []
        for engine in ctx.registry.list():
            if not isinstance(engine, InstallableEngine):
                continue
            items.append(_model_entry(engine, engine.engine_id, engine.display_name))
        return {"models": items, "runtime_root": str(ctx.runtime_root or paths.runtime_root())}

    @router.delete("/engines/{engine_id}/model", dependencies=[Depends(ctx.require_auth)])
    async def uninstall_model(engine_id: str) -> dict:
        """Uninstall one local model: remove its engine dir (weights + venv
        + install state) and start from a clean install lifecycle."""
        if engine_id in ctx.install_jobs:
            raise HTTPException(status_code=409, detail="该引擎正在安装中，无法卸载")
        target = _model_target(engine_id)
        if not hasattr(target, "model_dir"):
            raise HTTPException(status_code=422, detail=f"engine {engine_id} 没有可卸载的本地模型")
        # Re-check right before the destructive step: an install started
        # between the earlier check and here must not be deleted underneath.
        if engine_id in ctx.install_jobs:
            raise HTTPException(status_code=409, detail="该引擎刚开始安装，无法卸载；请稍后重试")
        d = target.model_dir()
        if d.exists():
            shutil.rmtree(d)
        return {"id": engine_id, "uninstalled": True, "model_dir": str(d)}

    return router
