"""Diagnostics (issue #7/#8) + transcription + engine bindings."""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from ..context import AppContext
from ..diagnostics import (
    DiagnosticError,
    analyze_audio,
)
from ..engines.cloud_base import (
    CloudEngineError,
)
from ..registry import (
    InstallableEngine,
)
from ..transcription import (
    LOCAL_TOOL_ID,
    TranscriptionError,
    engine_transcribers,
)
from ..voices import VoiceValidationError

# Issue #28: diagnostics only ever reads (never stores) the upload; cap it
# like the other upload endpoints instead of trusting the client.
ANALYSIS_UPLOAD_MAX_BYTES = 200 * 1024 * 1024


def _read_all_kv(ctx: AppContext, table: str) -> dict:
    """Read one kv table (settings / app_state) through a short-lived
    connection — the same fresh-read path every settings consumer uses."""
    from ..db import Database, library_db_path

    db = Database(library_db_path(ctx.data_root), migrate=False)
    try:
        return db.all_kv(table)
    finally:
        db.close()


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    registry = ctx.registry
    voice_store = ctx.voice_store
    _require_voice = ctx.require_voice
    _flag_placeholder = ctx.flag_placeholder
    _read_upload = ctx.read_upload
    _transcribe_voice_sync = ctx.transcribe_voice_sync

    def _diagnose_sync(path) -> dict:
        return analyze_audio(path)

    @router.post("/diagnose", dependencies=[Depends(ctx.require_auth)])
    async def diagnose_upload(file: UploadFile) -> dict:
        """Diagnose an audio file BEFORE creating a voice. Read-only: the
        upload lands in a temp file that is deleted after analysis."""
        tmp = None
        try:
            tmp = await _read_upload(file, max_bytes=ANALYSIS_UPLOAD_MAX_BYTES)
            if tmp is None:
                raise VoiceValidationError("参考音频不能为空")
            return await asyncio.get_running_loop().run_in_executor(
                None, _diagnose_sync, tmp
            )
        except (DiagnosticError, VoiceValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    tmp.unlink()

    @router.get("/voices/{voice_id}/diagnose", dependencies=[Depends(ctx.require_auth)])
    async def diagnose_voice(voice_id: str) -> dict:
        _require_voice(voice_id)
        path = voice_store.reference_path(voice_id)
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, _diagnose_sync, path
            )
        except DiagnosticError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/diagnostics/export", dependencies=[Depends(ctx.require_auth)])
    async def export_index() -> dict:
        """Export the whole SQLite index as readable JSON (ADR-0017).

        SQLite replaces the five hand-written JSON indexes, which weakens
        the "the library is a few readable files" intuition; this endpoint
        restores that inspectability — the full index, one GET away.
        """
        from ..db import T_APP_STATE, T_SETTINGS

        return {
            "voices": ctx.voice_store.list(),
            "generations": ctx.generation_store.list(limit=10**9)["records"],
            "compare_sessions": ctx.compare_store.list(limit=10**9),
            "regression_sessions": ctx.regression_store.list(limit=10**9),
            "settings": _read_all_kv(ctx, T_SETTINGS),
            "app_state": _read_all_kv(ctx, T_APP_STATE),
        }

    @router.get("/transcription/providers", dependencies=[Depends(ctx.require_auth)])
    async def transcription_providers() -> dict:
        return ctx.transcription_providers()

    @router.put("/transcription/provider", dependencies=[Depends(ctx.require_auth)])
    async def set_transcription_provider(body: dict) -> dict:
        provider = body.get("provider")
        # Validate against the cheap engine-id set — no install-state disk
        # reads just to check a name.
        if provider not in {"local"} | {e.engine_id for e in engine_transcribers(registry)}:
            raise HTTPException(
                status_code=422,
                detail=f"未知转写提供方：{provider!r}；可选 local 或已注册的转写引擎",
            )
        ctx.transcription_settings.set_provider(provider)
        return {"provider": ctx.transcription_settings.provider()}

    @router.get("/transcription/local/status", dependencies=[Depends(ctx.require_auth)])
    async def transcription_local_status() -> dict:
        try:
            tr = ctx.local_transcriber()
        except TranscriptionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        state = tr.install_state()
        installing = LOCAL_TOOL_ID in ctx.install_jobs
        # Issue #42: the SAME byte-level progress structure engine installs
        # expose (issue #35) — the engine-page card renders both with one
        # shared helper. Gated on an actually running install so a stale
        # field from a killed run is never advertised as a live progress bar.
        return {
            "id": LOCAL_TOOL_ID,
            "installed": state["installed"],
            "installing": installing,
            "steps": state["steps"],
            "progress": state.get("progress") if installing else None,
        }

    @router.post("/transcription/local/install", dependencies=[Depends(ctx.require_auth)])
    async def transcription_local_install() -> dict:
        """Install the local ASR tool in the background — same lifecycle as
        engine installs (per-step durable state, WS log lines, polling)."""
        try:
            tr = ctx.local_transcriber()
        except TranscriptionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ctx.start_install(LOCAL_TOOL_ID, tr.install)

    @router.post("/voices/{voice_id}/transcribe", dependencies=[Depends(ctx.require_auth)])
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
        # Issue #33: a cloud provider's failure (missing key, vendor error)
        # reaches the user verbatim — never a silent fallback to another
        # provider and never a bare 500; same 409 contract as local ASR.
        except (TranscriptionError, CloudEngineError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _flag_placeholder(voice_store.set_transcript(voice_id, text))

    @router.post("/voices/{voice_id}/bindings/{engine_id}", dependencies=[Depends(ctx.require_auth)])
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
        engine = ctx.require_engine(engine_id)
        if not engine.capabilities().voice_cloning:
            raise HTTPException(
                status_code=422,
                detail=f"engine {engine.engine_id} does not support voice cloning",
            )
        if engine.requires_key and ctx.keys.get(engine.engine_id) is None:
            raise HTTPException(
                status_code=409,
                detail=f"engine {engine.engine_id} needs an API key; "
                "set it in the vendor API-key area on the engines page first",
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
            _bind_log = ctx.log_from_thread(f"bind:{engine_id}")

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

    return router
