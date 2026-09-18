"""Shared application context passed to every router (issue #20).

``create_app`` used to be a single 1700-line closure: every store, lock and
helper lived in one scope, so no handler could be imported — let alone unit
tested — without booting the whole app. This module extracts the genuinely
shared services and helpers into one importable object. Routers receive an
``AppContext`` and declare their routes as plain functions over it.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any

from fastapi import HTTPException, UploadFile

from .aigc import embed_aigc_marker
from .compare import CompareStore
from .generations import GenerationStore
from .jobs import JobStore
from .logbus import LogBus
from .regression import RegressionStore
from .registry import Registry
from .secrets import KeyStore
from .transcription import (
    DEFAULT_PROVIDER,
    LocalTranscriber,
    TranscriptionError,
    TranscriptionSettings,
    engine_transcribers,
    is_placeholder_transcript,
)
from .voices import VoiceStore


@dataclass
class AppContext:
    """Everything the routers share: stores, locks and cross-cutting helpers."""

    registry: Registry
    token: str
    audio_dir: Path
    data_root: Path
    runtime_root: Path | None = None
    rate_limit_attempts: int = 4
    rate_limit_backoff: float = 1.0

    log_bus: LogBus = field(default_factory=LogBus)
    keys: KeyStore = field(default_factory=KeyStore)
    generation_store: GenerationStore = field(init=False)
    compare_store: CompareStore = field(init=False)
    regression_store: RegressionStore = field(init=False)
    voice_store: VoiceStore = field(init=False)
    job_store: JobStore = field(default_factory=JobStore)
    transcription_settings: TranscriptionSettings = field(init=False)

    # One generation at a time per engine: concurrent requests queue on this
    # lock instead of racing the worker, and a crashed worker restarts under
    # the lock — the queued requests behind it are never lost.
    generation_locks: dict = field(default_factory=dict)
    # Engine/tool installs run in background threads; one at a time, keyed by id.
    install_lock: threading.Lock = field(default_factory=threading.Lock)
    install_jobs: dict = field(default_factory=dict)

    # Issue #15: first-use voice consent. Durable in the portable data root
    # (issue #14's layout), so a library backup/restore carries it along.
    consent_version: str = "voice-consent-v1"
    consent_path: Path = field(init=False)
    consent_lock: threading.Lock = field(default_factory=threading.Lock)

    # Issue #21 / ADR-0014: UI preferences (theme, last selected engine) are
    # properties of the library, not of one machine's browser profile — so
    # they persist in the portable data root (same file as consent), not in
    # localStorage. A library backup/restore carries them along.
    ui_prefs_path: Path = field(init=False)
    ui_prefs_lock: threading.Lock = field(default_factory=threading.Lock)

    _transcriber: LocalTranscriber | None = field(default=None, repr=False)

    # Auth gates, built once from the process token (issue #19 policy).
    require_auth: Any = field(init=False, repr=False)
    require_media_auth: Any = field(init=False, repr=False)
    require_auth_ws: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from .auth import verify_media_query_token, verify_token, verify_token_ws

        self.generation_store = GenerationStore(self.data_root, self.audio_dir)
        self.compare_store = CompareStore(self.data_root)
        self.regression_store = RegressionStore(self.data_root)
        self.voice_store = VoiceStore(self.data_root)
        self.transcription_settings = TranscriptionSettings(self.data_root)
        self.consent_path = self.data_root / "app_state.json"
        self.ui_prefs_path = self.data_root / "app_state.json"
        self.require_auth = verify_token(self.token)
        self.require_media_auth = verify_media_query_token(self.token)
        self.require_auth_ws = verify_token_ws(self.token)

    # -- voice helpers -------------------------------------------------------

    def require_voice(self, voice_id: str) -> dict:
        record = self.voice_store.get(voice_id)
        if record is None:
            raise HTTPException(status_code=404, detail="voice not found")
        return record

    def require_engine(self, engine_id: str):
        """Look up an engine or raise the shared 404 contract."""
        engine = self.registry.get(engine_id)
        if engine is None:
            raise HTTPException(status_code=404, detail=f"unknown engine: {engine_id!r}")
        return engine

    def log_from_thread(self, stream_id: str):
        """Callback factory for executor/worker threads: hops the message
        back onto the running loop and publishes it on the log bus."""
        loop = asyncio.get_running_loop()

        def publish(message: str) -> None:
            loop.call_soon_threadsafe(self.log_bus.publish, stream_id, message)

        return publish

    def start_install(self, key: str, work, conflict_detail: str = "install already running") -> dict:
        """Run ``work(log)`` in a daemon thread as install job ``key``.

        Double-checked registration under ``install_lock``: a second request
        for a running install gets the 409 ``conflict_detail``. The job dict
        stays in ``install_jobs`` until ``work`` settles, so status polls
        report "installing" throughout.
        """
        if key in self.install_jobs:
            raise HTTPException(status_code=409, detail=conflict_detail)
        with self.install_lock:
            if key in self.install_jobs:
                raise HTTPException(status_code=409, detail=conflict_detail)
            job_id = f"install:{key}"
            job = {"id": job_id, "status": "running", "error": None}
            self.install_jobs[key] = job
            log = self.log_from_thread(job_id)

            def run():
                try:
                    work(log)
                    job["status"] = "succeeded"
                except Exception as exc:  # noqa: BLE001 - recorded in the job + logs
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    log(f"install failed: {exc}")
                finally:
                    self.install_jobs.pop(key, None)

            threading.Thread(target=run, daemon=True).start()
            return {"id": job_id, "status": "running"}

    def flag_placeholder(self, record: dict) -> dict:
        """Annotate a voice record for the UI (issue #18): reference
        transcripts that are the old fake engine's fixed test text are
        flagged, so the user is told to re-transcribe instead of the
        placeholder silently riding along as ref_text. Returns a copy —
        the flag is derived, never persisted."""
        ref = record.get("reference")
        if not isinstance(ref, dict) or "transcript_placeholder" in ref:
            return record
        annotated = {**record, "reference": {**ref}}
        annotated["reference"]["transcript_placeholder"] = is_placeholder_transcript(
            ref.get("transcript")
        )
        return annotated

    # -- uploads -------------------------------------------------------------

    async def read_upload(self, upload: UploadFile | None) -> Path | None:
        if upload is None or not upload.filename:
            return None
        tmp = self.audio_dir / f"upload-{uuid.uuid4().hex}{Path(upload.filename).suffix}"
        with open(tmp, "wb") as f:
            while chunk := await upload.read(1 << 20):
                f.write(chunk)
        return tmp

    # -- AIGC marker ---------------------------------------------------------

    def mark_artifact(self, target: dict, path: Path, fields: dict[str, str]) -> None:
        """Stamp an artifact with the AIGC marker (issue #15).

        Records the outcome on ``target`` ("aigc_marked") instead of failing
        the generation: a marker write must never turn a finished synthesis
        into an error, but a silent skip would hide a compliance gap.
        """
        try:
            target["aigc_marked"] = embed_aigc_marker(path, fields)
        except OSError:
            target["aigc_marked"] = False

    # -- transcription helpers (issue #8) --------------------------------------

    def local_transcriber(self) -> LocalTranscriber:
        if self._transcriber is None:
            self._transcriber = LocalTranscriber(root=self.runtime_root)
        return self._transcriber

    def transcribe_voice_sync(self, voice_id: str, provider: str | None) -> str:
        """Transcribe a voice's reference with the chosen provider.

        Runs blocking (worker subprocess / engine API call) — callers on the
        event loop must wrap it in an executor.
        """
        record = self.voice_store.get(voice_id)
        if record is None:
            raise KeyError(voice_id)
        ref_path = self.voice_store.reference_path(voice_id)
        if not ref_path.is_file():
            raise TranscriptionError("参考音频文件缺失，无法转写")
        provider = provider or self.transcription_settings.provider()
        if provider == "local":
            return self.local_transcriber().transcribe(str(ref_path), lambda m: None)
        engine = self.registry.get(provider)
        if engine is None or not callable(getattr(engine, "transcribe", None)):
            raise TranscriptionError(f"引擎 {provider!r} 不提供转写能力")
        return engine.transcribe(str(ref_path), lambda m: None)

    def transcription_providers(self) -> dict:
        engines = [
            {"id": e.engine_id, "display_name": e.display_name}
            for e in engine_transcribers(self.registry)
        ]
        local: dict = {"supported": True, "installed": False, "label": None, "status": None}
        try:
            tr = self.local_transcriber()
            local.update(
                installed=tr.is_installed(),
                label=tr.cfg["label"],
                status=tr.install_state(),
            )
        except TranscriptionError as exc:
            local.update(supported=False, label=None, reason=str(exc))
        return {
            "provider": self.transcription_settings.provider(),
            "default": DEFAULT_PROVIDER,
            "local": local,
            "engines": engines,
        }

    # -- consent (issue #15) ---------------------------------------------------

    def read_consent(self) -> dict:
        try:
            raw = json.loads(self.consent_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        state = raw.get("voice_consent")
        return state if isinstance(state, dict) else {}

    def write_consent(self, state: dict) -> None:
        from .storage import write_json_atomic

        with self.consent_lock:
            try:
                raw = json.loads(self.consent_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
            raw["voice_consent"] = state
            write_json_atomic(self.consent_path, raw)

    # -- UI preferences (issue #21 / ADR-0014) ---------------------------------

    def read_ui_prefs(self) -> dict:
        try:
            raw = json.loads(self.ui_prefs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        prefs = raw.get("ui_prefs")
        return prefs if isinstance(prefs, dict) else {}

    def write_ui_prefs(self, state: dict) -> None:
        from .storage import write_json_atomic

        with self.ui_prefs_lock:
            try:
                raw = json.loads(self.ui_prefs_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                raw = {}
            existing = raw.get("ui_prefs")
            merged = existing if isinstance(existing, dict) else {}
            merged.update(state)
            raw["ui_prefs"] = merged
            write_json_atomic(self.ui_prefs_path, raw)

    # -- portability -----------------------------------------------------------

    def new_temp_zip(self) -> Path:
        # mkstemp hands back an open fd; close it (the file is unlinked and
        # recreated atomically by the export/backup writer).
        fd, name = tempfile.mkstemp(suffix=".zip", prefix="voice-")
        os.close(fd)
        return Path(name)


def build_context(
    registry: Registry,
    token: str,
    audio_dir: Path,
    data_dir: Path | None = None,
    runtime_root: Path | None = None,
    key_store: KeyStore | None = None,
    rate_limit_attempts: int = 4,
    rate_limit_backoff: float = 1.0,
) -> AppContext:
    """Create the shared context exactly the way ``create_app`` used to."""
    # BYOK API keys (ADR-0003): the sidecar reads them from the OS key store
    # on demand and keeps nothing on disk. Tests inject an in-memory backend.
    keys = key_store if key_store is not None else KeyStore()
    # The data root is the portable library directory (issue #14): every
    # index and audio artifact lives under it, so it can be backed up,
    # restored and copied as one unit. BYOK keys are the exception — they
    # stay in the OS key store (ADR-0003).
    data_root = data_dir if data_dir is not None else audio_dir.parent
    return AppContext(
        registry=registry,
        token=token,
        audio_dir=audio_dir,
        data_root=data_root,
        runtime_root=runtime_root,
        rate_limit_attempts=rate_limit_attempts,
        rate_limit_backoff=rate_limit_backoff,
        keys=keys,
    )
