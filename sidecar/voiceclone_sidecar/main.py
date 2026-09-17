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
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .generations import GenerationStore, now_iso
from .normalization import normalize_text, normalize_with_flag
from .registry import InstallableEngine, Registry, default_registry
from .voices import AUDIO_MEDIA_TYPES, AVATAR_MEDIA_TYPES, VoiceStore, VoiceValidationError

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
        query_token = request.query_params.get("token", "")
        # The query-parameter form exists for media elements (<img>/<audio>),
        # which cannot set Authorization headers — same rationale as WS auth.
        if not (
            compare(auth, f"Bearer {expected}") or compare(query_token, expected)
        ):
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


def create_app(
    registry: Registry, token: str, audio_dir: Path, data_dir: Path | None = None
) -> FastAPI:
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
    # Generation records are durable history (search/filter/paginate/rerun);
    # each one snapshots enough lineage to reproduce its run.
    generation_store = GenerationStore(
        data_dir if data_dir is not None else audio_dir.parent, audio_dir
    )
    # One generation at a time per engine: concurrent requests queue on this
    # lock instead of racing the worker, and a crashed worker restarts under
    # the lock — the queued requests behind it are never lost.
    generation_locks: dict[str, threading.Lock] = {}
    # Engine installs run in background threads; one at a time, keyed by id.
    install_lock = threading.Lock()
    install_jobs: dict[str, dict] = {}
    # Voices live next to the audio dir (data root). Reference samples are the
    # source of truth; engine bindings derived from them are a cache.
    voice_store = VoiceStore(data_dir if data_dir is not None else audio_dir.parent)

    def _require_voice(voice_id: str) -> dict:
        record = voice_store.get(voice_id)
        if record is None:
            raise HTTPException(status_code=404, detail="voice not found")
        return record

    async def _read_upload(upload: UploadFile | None) -> Path | None:
        if upload is None or not upload.filename:
            return None
        tmp = audio_dir / f"upload-{uuid.uuid4().hex}{Path(upload.filename).suffix}"
        with open(tmp, "wb") as f:
            while chunk := await upload.read(1 << 20):
                f.write(chunk)
        return tmp

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
                    "installed": e.is_installed()
                    if isinstance(e, InstallableEngine)
                    else True,
                }
                for e in registry.list()
            ]
        }

    @app.get("/engines/{engine_id}/status", dependencies=[Depends(require_auth)])
    async def engine_status(engine_id: str) -> dict:
        engine = registry.get(engine_id)
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")
        if not isinstance(engine, InstallableEngine):
            return {"id": engine.engine_id, "installed": True, "installing": False, "steps": {}}
        state = engine.install_state()
        return {
            "id": engine.engine_id,
            "installed": state["installed"],
            "installing": engine_id in install_jobs,
            "steps": state["steps"],
        }

    @app.post("/engines/{engine_id}/install", dependencies=[Depends(require_auth)])
    async def install_engine(engine_id: str) -> dict:
        engine = registry.get(engine_id)
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")
        if not isinstance(engine, InstallableEngine):
            return {"id": engine.engine_id, "status": "installed"}
        if engine_id in install_jobs:
            raise HTTPException(status_code=409, detail="install already running for this engine")
        with install_lock:
            if engine_id in install_jobs:
                raise HTTPException(status_code=409, detail="install already running for this engine")

            job_id = f"install:{engine_id}"
            job = {"id": job_id, "status": "running", "error": None}
            install_jobs[engine_id] = job
            loop = asyncio.get_running_loop()

            def log(message: str) -> None:
                loop.call_soon_threadsafe(log_bus.publish, job_id, message)

            def progress(name: str, done: int, total: int | None) -> None:
                if total:
                    pct = int(done * 100 / total)
                    log(f"{name}: {done}/{total} bytes ({pct}%)")
                else:
                    log(f"{name}: {done} bytes")

            def run():
                try:
                    engine.install(log, progress)
                    job["status"] = "succeeded"
                except Exception as exc:  # noqa: BLE001 - recorded in the job + logs
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    log(f"install failed: {exc}")
                finally:
                    install_jobs.pop(engine_id, None)

            threading.Thread(target=run, daemon=True).start()
            return {"id": job_id, "status": "running"}


    # -- voices ---------------------------------------------------------------

    @app.post("/voices", dependencies=[Depends(require_auth)])
    async def create_voice(
        file: UploadFile,
        name: str = Form(...),
        description: str = Form(""),
        avatar: UploadFile | None = None,
    ) -> dict:
        audio_tmp = avatar_tmp = None
        try:
            audio_tmp = await _read_upload(file)
            avatar_tmp = await _read_upload(avatar)
            if audio_tmp is None:
                raise VoiceValidationError("参考音频不能为空")
            return voice_store.create(name, description, audio_tmp, avatar_tmp)
        except VoiceValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            for tmp in (audio_tmp, avatar_tmp):
                if tmp is not None:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass

    @app.get("/voices", dependencies=[Depends(require_auth)])
    async def list_voices() -> dict:
        return {"voices": voice_store.list()}

    @app.get("/voices/{voice_id}", dependencies=[Depends(require_auth)])
    async def get_voice(voice_id: str) -> dict:
        return _require_voice(voice_id)

    @app.patch("/voices/{voice_id}", dependencies=[Depends(require_auth)])
    async def update_voice(voice_id: str, body: dict) -> dict:
        _require_voice(voice_id)
        try:
            return voice_store.update(
                voice_id,
                name=body.get("name") if "name" in body else None,
                description=body.get("description") if "description" in body else None,
            )
        except VoiceValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/voices/{voice_id}/avatar", dependencies=[Depends(require_auth)])
    async def set_voice_avatar(voice_id: str, avatar: UploadFile) -> dict:
        _require_voice(voice_id)
        avatar_tmp = None
        try:
            avatar_tmp = await _read_upload(avatar)
            if avatar_tmp is None:
                raise VoiceValidationError("头像不能为空")
            return voice_store.set_avatar(voice_id, avatar_tmp)
        except VoiceValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if avatar_tmp is not None:
                try:
                    avatar_tmp.unlink()
                except OSError:
                    pass

    @app.delete("/voices/{voice_id}", dependencies=[Depends(require_auth)])
    async def delete_voice(voice_id: str) -> dict:
        _require_voice(voice_id)
        voice_store.delete(voice_id)
        return {"deleted": voice_id}

    @app.get("/voices/{voice_id}/reference", dependencies=[Depends(require_auth)])
    async def voice_reference(voice_id: str) -> FileResponse:
        _require_voice(voice_id)
        path = voice_store.reference_path(voice_id)
        media = AUDIO_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media, filename=path.name)

    @app.get("/voices/{voice_id}/avatar", dependencies=[Depends(require_auth)])
    async def voice_avatar(voice_id: str) -> FileResponse:
        _require_voice(voice_id)
        path = voice_store.avatar_path(voice_id)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="voice has no avatar")
        media = AVATAR_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media)

    @app.post("/voices/{voice_id}/bindings/{engine_id}", dependencies=[Depends(require_auth)])
    async def bind_voice(voice_id: str, engine_id: str) -> dict:
        """Create or refresh this voice's binding on one engine.

        For local engines a binding records that the engine can be handed the
        reference sample at generation time; it is rebuilt from the reference
        at any time. The engine must be installed — a binding never claims a
        state the engine cannot actually serve.
        """
        voice = _require_voice(voice_id)
        engine = registry.get(engine_id)
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")
        if not engine.capabilities().voice_cloning:
            raise HTTPException(
                status_code=422,
                detail=f"engine {engine.engine_id} does not support voice cloning",
            )
        if isinstance(engine, InstallableEngine) and not engine.is_installed():
            raise HTTPException(
                status_code=409,
                detail=f"engine {engine.engine_id} is not installed yet; "
                "install it before binding voices to it",
            )
        binding = voice_store.bind(voice_id, engine.engine_id, status="ready")
        return {"voice_id": voice["id"], "engine_id": engine.engine_id, **binding}

    @app.post("/normalize", dependencies=[Depends(require_auth)])
    async def normalize(body: dict) -> dict:
        """Normalization preview: same layer every generation path uses."""
        text = body.get("text")
        if not isinstance(text, str):
            raise HTTPException(status_code=422, detail="text must be a string")
        return normalize_with_flag(text)

    @app.post("/generations", dependencies=[Depends(require_auth)])
    async def create_generation(body: dict) -> dict:
        text = body.get("text")
        engine_id = body.get("engine_id")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=422, detail="text is required")
        engine = registry.get(engine_id or "")
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")

        params = dict(body.get("params") or {})
        voice_id = body.get("voice_id")
        voice = None
        if voice_id:
            voice = _require_voice(voice_id)
            ref_path = voice_store.reference_path(voice_id)
            if not ref_path.is_file():
                raise HTTPException(
                    status_code=409,
                    detail="voice reference sample is missing; delete and recreate the voice",
                )
            if engine.capabilities().voice_cloning and not params.get("ref_audio"):
                # The reference sample is the source of truth: hand it to the
                # engine on every generation. The binding below only records
                # that this happened.
                params["ref_audio"] = str(ref_path)

        # The normalization layer is applied centrally, here, before any
        # engine adapter sees the text — adapters cannot bypass it.
        normalized_text = normalize_text(text)

        generation_id = uuid.uuid4().hex
        started_monotonic = time.monotonic()
        record = {
            "id": generation_id,
            "engine_id": engine.engine_id,
            "model_version": None,
            "text": text,
            "normalized_text": normalized_text,
            "params": params,
            "voice_id": voice["id"] if voice else None,
            "voice_name": voice["name"] if voice else None,
            "status": "running",
            "error": None,
            "logs": [],
            "duration_seconds": None,
            "cost": None,
            "audio_url": None,
            "audio_file": None,
            "sample_rate": None,
            "created_at": now_iso(),
            "finished_at": None,
        }
        # Persist BEFORE synthesis starts: even a sidecar crash mid-run leaves
        # a record (status "running") instead of vanishing from history.
        # (The log callback below only appends while the worker thread runs,
        # strictly before any later persist call — no concurrent mutation.)
        generation_store.persist(record)

        # Synthesis runs in a worker thread: it can take seconds and must not
        # block the event loop (WS log streaming + other requests keep working).
        loop = asyncio.get_running_loop()

        def log(message: str) -> None:
            # Called from the worker thread — hop back onto the loop, and keep
            # the line in the record so history can replay it later.
            record["logs"].append({"ts": now_iso(), "message": message})
            loop.call_soon_threadsafe(log_bus.publish, generation_id, message)

        from .registry import GenerationRequest

        def run_synthesis():
            # setdefault stores the per-engine lock so later generations of the
            # same engine serialize behind it, even after a worker crash.
            with generation_locks.setdefault(engine.engine_id, threading.Lock()):
                return engine.synthesize(
                    GenerationRequest(generation_id=generation_id, text=normalized_text, params=record["params"]),
                    log,
                )

        try:
            result = await loop.run_in_executor(None, run_synthesis)
        except Exception as exc:  # noqa: BLE001 - recorded in history, surfaced on the contract
            record.update(
                status="failed",
                error=str(exc),
                finished_at=now_iso(),
                duration_seconds=round(time.monotonic() - started_monotonic, 3),
            )
            log_bus.publish(generation_id, f"generation failed: {exc}")
            generation_store.persist(record)
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

        audio_name = Path(result.audio_path).name
        record.update(
            status="succeeded",
            audio_url=f"/audio/{audio_name}",
            audio_file=audio_name,
            sample_rate=result.sample_rate,
            model_version=result.model_version,
            cost=result.cost,
            finished_at=now_iso(),
            duration_seconds=round(time.monotonic() - started_monotonic, 3),
        )
        if voice is not None:
            # Persist the binding only after a real synthesis succeeded —
            # the binding claims the reference works on this engine.
            voice_store.bind(voice["id"], engine.engine_id, status="ready")
            record["binding"] = voice_store.get(voice["id"])["bindings"].get(engine.engine_id)
        return generation_store.persist(record)

    @app.get("/generations", dependencies=[Depends(require_auth)])
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

    @app.get("/generations/{generation_id}", dependencies=[Depends(require_auth)])
    async def get_generation(generation_id: str) -> dict:
        record = generation_store.get(generation_id)
        if record is None:
            raise HTTPException(status_code=404, detail="generation not found")
        return record

    @app.delete("/generations/{generation_id}", dependencies=[Depends(require_auth)])
    async def delete_generation(generation_id: str) -> dict:
        # Removing a record also removes its audio artifact from disk.
        try:
            generation_store.delete(generation_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="generation not found") from None
        return {"deleted": generation_id}

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
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("SIDECAR_DATA_DIR") or None,
        help="voice library root; defaults to the audio dir's parent",
    )
    args = parser.parse_args(argv)
    audio_dir = Path(args.audio_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None

    import uvicorn

    app = create_app(
        default_registry(output_dir=audio_dir),
        token=args.token,
        audio_dir=audio_dir,
        data_dir=data_dir,
    )

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
