"""Generation entry + history + audio artifacts + log WebSocket.

``POST /generations`` is a thin shell over the shared pipeline
(:func:`voiceclone_sidecar.pipeline.run_generation`), which is importable
and directly unit-testable without booting the app (issue #20).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..context import AppContext
from ..normalization import normalize_with_flag
from ..pipeline import run_generation


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    generation_store = ctx.generation_store
    audio_dir = ctx.audio_dir

    @router.post("/normalize", dependencies=[Depends(ctx.require_auth)])
    async def normalize(body: dict) -> dict:
        """Normalization preview: same layer every generation path uses."""
        text = body.get("text")
        if not isinstance(text, str):
            raise HTTPException(status_code=422, detail="text must be a string")
        return normalize_with_flag(text)

    @router.post("/generations", dependencies=[Depends(ctx.require_auth)])
    async def create_generation(body: dict) -> dict:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=422, detail="text is required")
        return await run_generation(
            ctx,
            body.get("engine_id") or "",
            text,
            body.get("voice_id"),
            body.get("params"),
        )

    @router.get("/generations", dependencies=[Depends(ctx.require_auth)])
    async def list_generations(
        q: str = "",
        engine_id: str | None = None,
        voice_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
        if offset < 0:
            raise HTTPException(status_code=422, detail="offset must be non-negative")
        return generation_store.list(
            q=q, engine_id=engine_id, voice_id=voice_id, status=status,
            limit=limit, offset=offset,
        )

    @router.get("/generations/{generation_id}", dependencies=[Depends(ctx.require_auth)])
    async def get_generation(generation_id: str) -> dict:
        record = generation_store.get(generation_id)
        if record is None:
            raise HTTPException(status_code=404, detail="generation not found")
        return record

    @router.delete("/generations/{generation_id}", dependencies=[Depends(ctx.require_auth)])
    async def delete_generation(generation_id: str) -> dict:
        # Removing a record also removes its audio artifact from disk.
        try:
            generation_store.delete(generation_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="generation not found") from None
        return {"deleted": generation_id}

    @router.get("/audio/{filename}", dependencies=[Depends(ctx.require_media_auth)])
    async def audio(filename: str) -> FileResponse:
        path = (audio_dir / filename).resolve()
        if path.parent != audio_dir.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="audio not found")
        return FileResponse(path, media_type="audio/wav")

    @router.websocket("/ws/logs")
    async def ws_logs(websocket: WebSocket) -> None:
        await ctx.require_auth_ws(websocket)
        await websocket.accept()
        queue = ctx.log_bus.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            ctx.log_bus.unsubscribe(queue)

    return router
