"""Sidecar entry: FastAPI app factory + WebSocket log bus + server bootstrap.

The sidecar binds 127.0.0.1 on a dynamic port (port 0) and enforces bearer
auth on every route. Once listening it prints a single JSON handshake line
on stdout, e.g.:

    {"event": "ready", "port": 54321, "pid": 1234}

The desktop shell reads that line to learn where to connect.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .registry import Registry, default_registry

SIDECAR_VERSION = "0.1.0"


class LogBus:
    """Broadcasts generation log lines to every connected WebSocket client."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, generation_id: str, message: str) -> None:
        event = {
            "type": "log",
            "generation_id": generation_id,
            "message": message,
            "ts": asyncio.get_running_loop().time(),
        }
        for q in list(self._subscribers):
            q.put_nowait(event)


def verify_token(expected: str):
    compare = secrets.compare_digest

    async def _verify(request: Request) -> None:
        auth = request.headers.get("Authorization", "")
        if not compare(auth, f"Bearer {expected}"):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    return _verify


def verify_token_ws(expected: str):
    async def _verify(websocket: WebSocket) -> None:
        # Browsers cannot set headers on WebSocket, so also accept the token
        # as a query parameter. Header form stays the primary contract.
        auth = websocket.headers.get("Authorization", "")
        query_token = websocket.query_params.get("token", "")
        if not (
            secrets.compare_digest(auth, f"Bearer {expected}")
            or secrets.compare_digest(query_token, expected)
        ):
            await websocket.accept()
            await websocket.close(code=4401, reason="invalid or missing bearer token")
            raise WebSocketDisconnect(code=4401)

    return _verify


def create_app(registry: Registry, token: str, audio_dir: Path) -> FastAPI:
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="voiceclone-sidecar", version=SIDECAR_VERSION)
    # Renderer runs on a dev origin in development; the sidecar is a local
    # loopback service guarded by bearer auth, so origins stay permissive.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    log_bus = LogBus()
    require_auth = verify_token(token)
    require_auth_ws = verify_token_ws(token)
    generations: dict[str, dict] = {}

    @app.get("/health", dependencies=[Depends(require_auth)])
    async def health() -> dict:
        return {"status": "ok", "version": SIDECAR_VERSION}

    @app.get("/engines", dependencies=[Depends(require_auth)])
    async def engines() -> dict:
        return {
            "engines": [
                {
                    "id": e.engine_id,
                    "display_name": e.display_name,
                    "capabilities": e.capabilities().to_dict(),
                }
                for e in registry.list()
            ]
        }

    @app.post("/generations", dependencies=[Depends(require_auth)])
    async def create_generation(body: dict) -> dict:
        text = body.get("text")
        engine_id = body.get("engine_id")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=422, detail="text is required")
        engine = registry.get(engine_id or "")
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")

        generation_id = uuid.uuid4().hex
        record = {
            "id": generation_id,
            "engine_id": engine.engine_id,
            "text": text,
            "params": body.get("params") or {},
            "status": "running",
        }
        generations[generation_id] = record

        # Synthesis runs in a worker thread: it can take seconds and must not
        # block the event loop (WS log streaming + other requests keep working).
        loop = asyncio.get_running_loop()

        def log(message: str) -> None:
            # Called from the worker thread — hop back onto the loop.
            loop.call_soon_threadsafe(log_bus.publish, generation_id, message)

        from .registry import GenerationRequest

        def run_synthesis():
            return engine.synthesize(
                GenerationRequest(generation_id=generation_id, text=text, params=record["params"]),
                log,
            )

        try:
            result = await loop.run_in_executor(None, run_synthesis)
        except Exception as exc:  # noqa: BLE001 - surfaced on the contract, not swallowed
            record["status"] = "failed"
            record["error"] = str(exc)
            log_bus.publish(generation_id, f"generation failed: {exc}")
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

        record.update(
            status="succeeded",
            audio_url=f"/audio/{Path(result.audio_path).name}",
            sample_rate=result.sample_rate,
        )
        return record

    @app.get("/generations/{generation_id}", dependencies=[Depends(require_auth)])
    async def get_generation(generation_id: str) -> dict:
        record = generations.get(generation_id)
        if record is None:
            raise HTTPException(status_code=404, detail="generation not found")
        return record

    @app.get("/audio/{filename}", dependencies=[Depends(require_auth)])
    async def audio(filename: str) -> FileResponse:
        path = (audio_dir / filename).resolve()
        if path.parent != audio_dir.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="audio not found")
        return FileResponse(path, media_type="audio/wav")

    @app.websocket("/ws/logs")
    async def ws_logs(websocket: WebSocket) -> None:
        await require_auth_ws(websocket)
        await websocket.accept()
        queue = log_bus.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            log_bus.unsubscribe(queue)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="voiceclone-sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 = dynamic port")
    parser.add_argument("--token", default=os.environ.get("SIDECAR_TOKEN") or secrets.token_urlsafe(32))
    parser.add_argument("--audio-dir", default=os.environ.get("SIDECAR_AUDIO_DIR") or "data/audio")
    args = parser.parse_args(argv)
    audio_dir = Path(args.audio_dir)

    import uvicorn

    app = create_app(default_registry(output_dir=audio_dir), token=args.token, audio_dir=audio_dir)

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    server = uvicorn.Server(config)

    async def serve_and_announce() -> None:
        serve_task = asyncio.create_task(server.serve())
        # Announce only once the socket is actually listening — the shell
        # connects right after reading this line.
        while not server.started:
            if serve_task.done():
                break
            await asyncio.sleep(0.01)
        if server.started:
            port = None
            for s in server.servers:
                if s.sockets:
                    port = s.sockets[0].getsockname()[1]
                    break
            print(json.dumps({"event": "ready", "port": port, "pid": os.getpid()}), flush=True)
            sys.stdout.flush()
        await serve_task

    asyncio.run(serve_and_announce())


if __name__ == "__main__":
    main()
