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
import random
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .diagnostics import DiagnosticError, analyze_audio
from .compare import (
    ALL_LABELS,
    CompareError,
    CompareStore,
    DEFAULT_TEXT_TYPE,
    TEXT_TYPES,
    detect_language,
)
from .loudness import TARGET_LUFS, LoudnessError, normalize_wav_lufs
from .generations import GenerationStore, now_iso
from .transcription import (
    LOCAL_TOOL_ID,
    DEFAULT_PROVIDER,
    LocalTranscriber,
    TranscriptionError,
    TranscriptionSettings,
    engine_transcribers,
)
from .normalization import normalize_text, normalize_with_flag
from .engines.dashscope_base import CloudEngineError
from .registry import InstallableEngine, Registry, default_registry
from .runtime import paths
from .secrets import KeyStore, KeyStoreError
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
    registry: Registry,
    token: str,
    audio_dir: Path,
    data_dir: Path | None = None,
    runtime_root: Path | None = None,
    key_store: KeyStore | None = None,
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
    # BYOK API keys (ADR-0003): the sidecar reads them from the OS key store
    # on demand and keeps nothing on disk. Tests inject an in-memory backend.
    keys = key_store if key_store is not None else KeyStore()
    # Generation records are durable history (search/filter/paginate/rerun);
    # each one snapshots enough lineage to reproduce its run.
    generation_store = GenerationStore(
        data_dir if data_dir is not None else audio_dir.parent, audio_dir
    )
    # Blind-comparison sessions + the preference profile (issue #10): a
    # separate durable index next to the generation history.
    compare_store = CompareStore(data_dir if data_dir is not None else audio_dir.parent)
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
    # Transcription (issue #8): provider choice is durable; the local tool
    # is built lazily so unsupported platforms degrade to cloud-only.
    transcription_settings = TranscriptionSettings(
        data_dir if data_dir is not None else audio_dir.parent
    )
    _transcriber: LocalTranscriber | None = None

    def _local_transcriber() -> LocalTranscriber:
        nonlocal _transcriber
        if _transcriber is None:
            _transcriber = LocalTranscriber(root=runtime_root)
        return _transcriber

    def _require_voice(voice_id: str) -> dict:
        record = voice_store.get(voice_id)
        if record is None:
            raise HTTPException(status_code=404, detail="voice not found")
        return record

    def _transcribe_voice_sync(voice_id: str, provider: str | None) -> str:
        """Transcribe a voice's reference with the chosen provider.

        Runs blocking (worker subprocess / engine API call) — callers on the
        event loop must wrap it in an executor.
        """
        record = voice_store.get(voice_id)
        if record is None:
            raise KeyError(voice_id)
        ref_path = voice_store.reference_path(voice_id)
        if not ref_path.is_file():
            raise TranscriptionError("参考音频文件缺失，无法转写")
        provider = provider or transcription_settings.provider()
        if provider == "local":
            return _local_transcriber().transcribe(str(ref_path), lambda m: None)
        engine = registry.get(provider)
        if engine is None or not callable(getattr(engine, "transcribe", None)):
            raise TranscriptionError(f"引擎 {provider!r} 不提供转写能力")
        return engine.transcribe(str(ref_path), lambda m: None)

    def _transcription_providers() -> dict:
        engines = [
            {"id": e.engine_id, "display_name": e.display_name}
            for e in engine_transcribers(registry)
        ]
        local: dict = {"supported": True, "installed": False, "label": None, "status": None}
        try:
            tr = _local_transcriber()
            local.update(
                installed=tr.is_installed(),
                label=tr.cfg["label"],
                status=tr.install_state(),
            )
        except TranscriptionError as exc:
            local.update(supported=False, label=None, reason=str(exc))
        return {
            "provider": transcription_settings.provider(),
            "default": DEFAULT_PROVIDER,
            "local": local,
            "engines": engines,
        }

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

    # -- BYOK keys (issue #9) ---------------------------------------------------
    #
    # The contract never returns key values — only which engines have one
    # configured. Values go renderer -> sidecar -> OS key store and are then
    # dropped from memory beyond the read the engines make at call time.

    def _key_engine(engine_id: str):
        engine = registry.get(engine_id)
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")
        if not engine.requires_key:
            raise HTTPException(
                status_code=422, detail=f"engine {engine.engine_id} does not use an API key"
            )
        return engine

    @app.get("/settings/keys", dependencies=[Depends(require_auth)])
    async def list_keys() -> dict:
        cloud = [e.engine_id for e in registry.list() if e.requires_key]
        return {"keys": keys.status(cloud)}

    @app.put("/settings/keys", dependencies=[Depends(require_auth)])
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

    @app.delete("/settings/keys/{engine_id}", dependencies=[Depends(require_auth)])
    async def delete_key(engine_id: str) -> dict:
        _key_engine(engine_id)
        try:
            keys.delete(engine_id)
        except KeyStoreError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"engine_id": engine_id, "configured": False}

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

    # -- diagnostics + transcription (issue #8) -------------------------------

    def _diagnose_sync(path: Path) -> dict:
        return analyze_audio(path)

    @app.post("/diagnose", dependencies=[Depends(require_auth)])
    async def diagnose_upload(file: UploadFile) -> dict:
        """Diagnose an audio file BEFORE creating a voice. Read-only: the
        upload lands in a temp file that is deleted after analysis."""
        tmp = None
        try:
            tmp = await _read_upload(file)
            if tmp is None:
                raise VoiceValidationError("参考音频不能为空")
            return await asyncio.get_running_loop().run_in_executor(
                None, _diagnose_sync, tmp
            )
        except (DiagnosticError, VoiceValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    @app.get("/voices/{voice_id}/diagnose", dependencies=[Depends(require_auth)])
    async def diagnose_voice(voice_id: str) -> dict:
        _require_voice(voice_id)
        path = voice_store.reference_path(voice_id)
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, _diagnose_sync, path
            )
        except DiagnosticError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/transcription/providers", dependencies=[Depends(require_auth)])
    async def transcription_providers() -> dict:
        return _transcription_providers()

    @app.put("/transcription/provider", dependencies=[Depends(require_auth)])
    async def set_transcription_provider(body: dict) -> dict:
        provider = body.get("provider")
        # Validate against the cheap engine-id set — no install-state disk
        # reads just to check a name.
        if provider not in {"local"} | {e.engine_id for e in engine_transcribers(registry)}:
            raise HTTPException(
                status_code=422,
                detail=f"未知转写提供方：{provider!r}；可选 local 或已注册的转写引擎",
            )
        transcription_settings.set_provider(provider)
        return {"provider": transcription_settings.provider()}

    @app.get("/transcription/local/status", dependencies=[Depends(require_auth)])
    async def transcription_local_status() -> dict:
        try:
            tr = _local_transcriber()
        except TranscriptionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        state = tr.install_state()
        return {
            "id": LOCAL_TOOL_ID,
            "installed": state["installed"],
            "installing": LOCAL_TOOL_ID in install_jobs,
            "steps": state["steps"],
        }

    @app.post("/transcription/local/install", dependencies=[Depends(require_auth)])
    async def transcription_local_install() -> dict:
        """Install the local ASR tool in the background — same lifecycle as
        engine installs (per-step durable state, WS log lines, polling)."""
        try:
            tr = _local_transcriber()
        except TranscriptionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if LOCAL_TOOL_ID in install_jobs:
            raise HTTPException(status_code=409, detail="install already running")
        with install_lock:
            if LOCAL_TOOL_ID in install_jobs:
                raise HTTPException(status_code=409, detail="install already running")
            job_id = f"install:{LOCAL_TOOL_ID}"
            job = {"id": job_id, "status": "running", "error": None}
            install_jobs[LOCAL_TOOL_ID] = job
            loop = asyncio.get_running_loop()

            def log(message: str) -> None:
                loop.call_soon_threadsafe(log_bus.publish, job_id, message)

            def run():
                try:
                    tr.install(log)
                    job["status"] = "succeeded"
                except Exception as exc:  # noqa: BLE001 - recorded in the job + logs
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    log(f"install failed: {exc}")
                finally:
                    install_jobs.pop(LOCAL_TOOL_ID, None)

            threading.Thread(target=run, daemon=True).start()
            return {"id": job_id, "status": "running"}

    @app.post("/voices/{voice_id}/transcribe", dependencies=[Depends(require_auth)])
    async def transcribe_voice(voice_id: str, body: dict | None = None) -> dict:
        """Transcribe the reference and store the transcript on the voice.

        The user never types reference text for engines that need it (issue
        #8); the transcript is metadata — the audio itself is never touched.
        """
        _require_voice(voice_id)
        provider = (body or {}).get("provider")
        try:
            text = await asyncio.get_running_loop().run_in_executor(
                None, _transcribe_voice_sync, voice_id, provider
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="voice not found") from None
        except TranscriptionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return voice_store.set_transcript(voice_id, text)

    @app.post("/voices/{voice_id}/bindings/{engine_id}", dependencies=[Depends(require_auth)])
    async def bind_voice(voice_id: str, engine_id: str) -> dict:
        """Create or refresh this voice's binding on one engine.

        For local engines a binding records that the engine can be handed the
        reference sample at generation time; it is rebuilt from the reference
        at any time. For BYOK cloud engines the binding carries the vendor's
        voice_id minted by enrolling the reference (issue #9). The engine
        must be usable — installed (local) and keyed (cloud) — a binding
        never claims a state the engine cannot actually serve.
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
        if engine.requires_key and keys.get(engine.engine_id) is None:
            raise HTTPException(
                status_code=409,
                detail=f"engine {engine.engine_id} needs an API key; "
                "set it on the settings page first",
            )
        if isinstance(engine, InstallableEngine) and not engine.is_installed():
            raise HTTPException(
                status_code=409,
                detail=f"engine {engine.engine_id} is not installed yet; "
                "install it before binding voices to it",
            )

        binding_extra: dict | None = None
        if engine.requires_key:
            ref_path = voice_store.reference_path(voice_id)
            transcript = (voice.get("reference") or {}).get("transcript")
            loop = asyncio.get_running_loop()

            def _bind_log(message: str) -> None:
                # Runs in the executor thread — hop back onto the loop.
                loop.call_soon_threadsafe(log_bus.publish, f"bind:{engine_id}", message)

            try:
                binding_extra = await loop.run_in_executor(
                    None,
                    lambda: engine.bind_reference(ref_path, transcript, _bind_log),
                )
            except CloudEngineError as exc:
                raise HTTPException(
                    status_code=502, detail=f"云端绑定失败：{exc}"
                ) from exc

        binding = voice_store.bind(voice_id, engine.engine_id, status="ready", extra=binding_extra)
        return {"voice_id": voice["id"], "engine_id": engine.engine_id, **binding}

    # -- voice design (issue #11) -----------------------------------------------
    #
    # A voice can be created from a text description alone. The design
    # engine's preview sample becomes the voice's reference (the source of
    # truth every other engine binds from, ADR-0001), so a designed voice is
    # a first-class voice: multi-engine bindings, generation and deletion
    # all go through the same paths as a cloned one.

    @app.post("/voices/design", dependencies=[Depends(require_auth)])
    async def design_voice(body: dict) -> dict:
        engine_id = body.get("engine_id")
        name = body.get("name")
        description = body.get("description", "")
        voice_prompt = body.get("voice_prompt")
        preview_text = body.get("preview_text")

        engine = registry.get(engine_id if isinstance(engine_id, str) else "")
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")
        if not engine.capabilities().voice_design:
            raise HTTPException(
                status_code=422,
                detail=f"引擎 {engine.engine_id} 未声明音色设计能力：不支持用文字描述创建音色",
            )
        if engine.requires_key and keys.get(engine.engine_id) is None:
            raise HTTPException(
                status_code=409,
                detail=f"引擎 {engine.engine_id} 需要 API Key；请在「设置」页配置后再设计音色",
            )
        if not isinstance(voice_prompt, str) or not voice_prompt.strip():
            raise HTTPException(status_code=422, detail="声音描述不能为空")
        if not isinstance(preview_text, str) or not preview_text.strip():
            raise HTTPException(status_code=422, detail="试听文本不能为空")
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(status_code=422, detail="音色名称不能为空")

        voice = voice_store.create_designed(
            name, description if isinstance(description, str) else "",
            engine_id=engine.engine_id,
            voice_prompt=voice_prompt,
            preview_text=preview_text,
        )

        loop = asyncio.get_running_loop()

        def _design_log(message: str) -> None:
            # Runs in the executor thread — hop back onto the loop.
            loop.call_soon_threadsafe(log_bus.publish, f"design:{engine.engine_id}", message)

        try:
            extra = await loop.run_in_executor(
                None,
                lambda: engine.design_voice(voice_prompt, preview_text, _design_log),
            )
            voice = voice_store.attach_reference(
                voice["id"], Path(extra["sample_audio_path"]), extra.get("transcript")
            )
            voice_store.bind(voice["id"], engine.engine_id, status="ready",
                             extra={"voice_id": extra["voice_id"]})
            return voice_store.get(voice["id"])
        except CloudEngineError as exc:
            voice_store.delete(voice["id"])
            raise HTTPException(status_code=502, detail=f"设计音色失败：{exc}") from exc
        except (Exception, OSError) as exc:  # noqa: BLE001
            # ANY failure (vendor or local IO) leaves no orphaned,
            # reference-less voice behind — ADR-0010's hard guarantee.
            voice_store.delete(voice["id"])
            raise HTTPException(
                status_code=500, detail=f"设计音色失败：{exc}"
            ) from exc

    @app.post("/normalize", dependencies=[Depends(require_auth)])
    async def normalize(body: dict) -> dict:
        """Normalization preview: same layer every generation path uses."""
        text = body.get("text")
        if not isinstance(text, str):
            raise HTTPException(status_code=422, detail="text must be a string")
        return normalize_with_flag(text)

    async def run_generation(
        engine_id: str,
        text: str,
        voice_id: str | None = None,
        params: dict | None = None,
        generation_id: str | None = None,
    ) -> dict:
        """One complete synthesis run through the shared pipeline.

        Both single generation (POST /generations) and every engine leg of a
        blind comparison (issue #10) execute EXACTLY this code — a compared
        candidate is never produced by a different route than a plain one.
        """
        engine = registry.get(engine_id or "")
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")

        params = dict(params or {})
        voice = None
        if voice_id:
            voice = _require_voice(voice_id)
            if not voice.get("reference"):
                # Designed voices always carry their preview sample after a
                # successful design run; a record without one is unusable.
                raise HTTPException(
                    status_code=409,
                    detail="该音色没有参考样本，无法参与生成；请删除后重新创建",
                )
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
            if engine.capabilities().requires_reference_text and not params.get("ref_text"):
                # Engines that need reference text get it automatically
                # (issue #8): stored transcript first, on-demand transcription
                # second — the user never types it by hand.
                transcript = (voice.get("reference") or {}).get("transcript")
                if not transcript:
                    try:
                        transcript = await asyncio.get_running_loop().run_in_executor(
                            None, _transcribe_voice_sync, voice["id"], None
                        )
                    except TranscriptionError as exc:
                        raise HTTPException(
                            status_code=409,
                            detail=f"该引擎需要参考文本，自动转写失败：{exc}",
                        ) from exc
                    voice_store.set_transcript(voice["id"], transcript)
                params["ref_text"] = transcript
            elif not params.get("ref_text"):
                # Engines that don't require reference text still benefit from
                # one when it exists (cloud enrollment quality) — pass it if
                # it is already known, never transcribe on demand for them.
                transcript = (voice.get("reference") or {}).get("transcript")
                if transcript:
                    params["ref_text"] = transcript

        # The normalization layer is applied centrally, here, before any
        # engine adapter sees the text — adapters cannot bypass it.
        normalized_text = normalize_text(text)

        generation_id = generation_id or uuid.uuid4().hex

        # BYOK cloud engines: the key must exist, and the voice needs its
        # cloud binding. Missing bindings are minted right here (issue #9):
        # the reference is enrolled once, the returned voice_id becomes part
        # of the binding persisted after the run succeeds.
        binding_extra: dict | None = None
        if engine.requires_key:
            if keys.get(engine.engine_id) is None:
                raise HTTPException(
                    status_code=409,
                    detail=f"引擎 {engine.engine_id} 需要阿里百炼 API Key；"
                    "请在「设置」页配置后再生成",
                )
            if voice is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"云端引擎 {engine.engine_id} 必须使用音色生成"
                    "（复刻模型需要一个已绑定的云端音色）",
                )
            cloud_voice_id = (voice["bindings"].get(engine.engine_id) or {}).get("voice_id")
            if not cloud_voice_id:
                ref_path = voice_store.reference_path(voice["id"])
                transcript = params.get("ref_text") or (voice.get("reference") or {}).get("transcript")

                enroll_loop = asyncio.get_running_loop()

                def _log(message: str) -> None:
                    # Runs in the executor thread — hop back onto the loop.
                    enroll_loop.call_soon_threadsafe(log_bus.publish, generation_id, message)

                try:
                    binding_extra = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: engine.bind_reference(ref_path, transcript, _log),
                    )
                except CloudEngineError as exc:
                    raise HTTPException(
                        status_code=502, detail=f"云端音色创建失败：{exc}"
                    ) from exc
                cloud_voice_id = binding_extra["voice_id"]
                # Persist the cloud binding the moment enrollment succeeds —
                # the vendor-side voice now exists regardless of what happens
                # to this synthesis run. Waiting for synthesis would leak an
                # orphaned cloud voice on every failed run.
                voice_store.bind(voice["id"], engine.engine_id, status="ready", extra=binding_extra)
            params["voice_id"] = cloud_voice_id

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
            # the binding claims the reference works on this engine. A cloud
            # engine's voice_id (minted during this run or an earlier one)
            # rides along as binding extras.
            voice_store.bind(voice["id"], engine.engine_id, status="ready", extra=binding_extra)
            record["binding"] = voice_store.get(voice["id"])["bindings"].get(engine.engine_id)
        return generation_store.persist(record)

    @app.post("/generations", dependencies=[Depends(require_auth)])
    async def create_generation(body: dict) -> dict:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=422, detail="text is required")
        return await run_generation(
            body.get("engine_id") or "",
            text,
            body.get("voice_id"),
            body.get("params"),
        )

    # -- blind comparison + preference profile (issue #10, ADR-0009) ----------
    #
    # One voice + one text -> several engines side by side. Every artifact is
    # gain-normalized to the same LUFS target so loudness cannot bias the
    # listener; entries carry shuffled blind labels until revealed. Ratings
    # aggregate into a read-only preference profile — it never drives any
    # default-engine selection or routing decision.

    async def _normalize_entries(entries: list[dict], session_id: str) -> list[dict]:
        """Gain-normalize each leg's artifact to the shared LUFS target.

        Normalized copies are named by SESSION + blind label — never by the
        generation id — so even the served URL cannot correlate a blind entry
        back to its engine through the generation history.
        """
        loop = asyncio.get_running_loop()
        labeled: list[dict] = []
        for entry, label in zip(entries, list(ALL_LABELS)[: len(entries)]):
            norm_name = f"norm-{session_id}-{label}.wav"
            try:
                measurement = await loop.run_in_executor(
                    None,
                    lambda: normalize_wav_lufs(
                        audio_dir / entry["audio_file"], audio_dir / norm_name
                    ),
                )
            except LoudnessError as exc:
                entry.update(status="failed", error=f"响度归一化失败：{exc}")
                continue
            entry.update(
                label=label,
                normalized_file=norm_name,
                normalized_audio_url=f"/audio/{norm_name}",
                **measurement,
            )
            labeled.append(entry)
        return labeled

    @app.post("/compare", dependencies=[Depends(require_auth)])
    async def create_compare(body: dict) -> dict:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=422, detail="text is required")
        voice_id = body.get("voice_id")
        if not isinstance(voice_id, str) or not voice_id:
            raise HTTPException(status_code=422, detail="对比必须选择一个音色")
        _require_voice(voice_id)

        engine_ids = body.get("engine_ids")
        if (
            not isinstance(engine_ids, list)
            or len(engine_ids) < 2
            or len(engine_ids) > len(ALL_LABELS)
            or not all(isinstance(e, str) for e in engine_ids)
            or len(set(engine_ids)) != len(engine_ids)
        ):
            raise HTTPException(
                status_code=422,
                detail=f"engine_ids 需要 2–{len(ALL_LABELS)} 个不重复的引擎 ID",
            )
        for engine_id in engine_ids:
            if registry.get(engine_id) is None:
                raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")

        text_type = body.get("text_type") or DEFAULT_TEXT_TYPE
        if text_type not in TEXT_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"text_type 必须是 {', '.join(TEXT_TYPES)} 之一",
            )

        # Every engine leg runs the shared pipeline; one engine failing (not
        # installed, no key, vendor error) fails ITS entry, never the session.
        # Engine order is shuffled up front so blind label A/B/C never maps
        # to a predictable engine position.
        engine_ids = random.sample(engine_ids, len(engine_ids))
        entries: list[dict] = []
        failed: list[dict] = []
        for engine_id in engine_ids:
            try:
                record = await run_generation(engine_id, text, voice_id, None)
            except HTTPException as exc:
                failed.append(
                    {"engine_id": engine_id, "error": exc.detail}
                )
                continue
            entries.append(
                {
                    "engine_id": engine_id,
                    "generation_id": record["id"],
                    "audio_file": record.get("audio_file"),
                    "status": "ok",
                    "error": None,
                }
            )

        # Normalize each artifact to the shared LUFS target (pure gain, peak
        # safe). Comparison only makes sense over the entries that produced
        # audio; failed legs stay listed so the user knows what to fix.
        session_id = uuid.uuid4().hex
        present = await _normalize_entries([e for e in entries if e["audio_file"]], session_id)
        # Stable display order: by blind label so A always comes first.
        present.sort(key=lambda e: e["label"])

        session = {
            "id": session_id,
            "created_at": now_iso(),
            "voice_id": voice_id,
            "text": text,
            "text_type": text_type,
            "language": detect_language(text),
            "target_lufs": TARGET_LUFS,
            "entries": present,
            "failed": failed,
            "scored": bool(present) and all("score" in e for e in present),
        }
        return _blind_view(compare_store.create(session), reveal=False)

    def _blind_view(session: dict, reveal: bool) -> dict:
        """Strip engine identity from a session unless revealed.

        Identity leaks through more than the engine_id field: the per-entry
        loudness numbers fingerprint the engines (with level-skewed engines
        the original LUFS ordering IS the answer), failed legs name the
        engine, and generation ids correlate with /generations history. All
        of it stays behind until reveal.
        """
        if reveal:
            return session
        hidden = ("engine_id", "generation_id", "audio_file", "normalized_file",
                  "original_lufs", "gain_db", "achieved_lufs", "peak_limited")
        out = dict(session)
        out["entries"] = [
            {k: v for k, v in e.items() if k not in hidden}
            for e in out.get("entries", [])
        ]
        out["failed_count"] = len(out.get("failed", []))
        out["failed"] = []
        return out

    @app.get("/compare", dependencies=[Depends(require_auth)])
    async def list_compares(reveal: bool = False, limit: int = 50) -> dict:
        sessions = compare_store.list(limit)
        return {
            "sessions": [
                _blind_view(s, reveal or s.get("scored", False)) for s in sessions
            ]
        }

    @app.get("/compare/{session_id}", dependencies=[Depends(require_auth)])
    async def get_compare(session_id: str, reveal: bool = False) -> dict:
        session = compare_store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="comparison session not found")
        return _blind_view(session, reveal or session.get("scored", False))

    @app.post("/compare/{session_id}/scores", dependencies=[Depends(require_auth)])
    async def score_compare(session_id: str, body: dict) -> dict:
        session = compare_store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="comparison session not found")
        scores = body.get("scores")
        if not isinstance(scores, dict) or not scores:
            raise HTTPException(status_code=422, detail="scores must be a {label: rating} object")
        try:
            session = CompareStore.set_scores(session, scores)
        except CompareError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _blind_view(compare_store.update(session), reveal=session["scored"])

    @app.delete("/compare/{session_id}", dependencies=[Depends(require_auth)])
    async def delete_compare(session_id: str) -> dict:
        try:
            session = compare_store.delete(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="comparison session not found") from None
        # Normalized copies are comparison-internal artifacts; the original
        # generations (and their history) survive the session.
        for entry in session.get("entries", []):
            name = entry.get("normalized_file")
            if name:
                try:
                    (audio_dir / name).unlink()
                except OSError:
                    pass
        return {"deleted": session_id}

    @app.get("/preferences", dependencies=[Depends(require_auth)])
    async def preference_profile() -> dict:
        """The local preference profile — insight only, never routing input."""
        return compare_store.preference_profile()

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
        default_registry(output_dir=audio_dir, key_store=KeyStore()),
        token=args.token,
        audio_dir=audio_dir,
        data_dir=data_dir,
        runtime_root=paths.runtime_root(),
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
