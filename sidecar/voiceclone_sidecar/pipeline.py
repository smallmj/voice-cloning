"""The shared generation pipeline (issue #20's main extraction).

One complete synthesis run — voice/reference checks, reference text,
normalization, BYOK key/binding handling with cloud-voice health checks and
auto re-enrollment, the synthesis worker with rate-limit retry, AIGC marking
and history persistence.

Single generation (POST /generations), every engine leg of a blind
comparison (issue #10), every segment of a long-text job (issue #13) and
every engine × item pair of a regression run (issue #16) execute EXACTLY
this function — a compared or queued candidate is never produced by a
different route than a plain one.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from pathlib import Path

from fastapi import HTTPException

from .context import AppContext
from .engines.cloud_base import CloudEngineError
from .generations import now_iso
from .normalization import normalize_text
from .registry import Engine, GenerationRequest
from .transcription import TranscriptionError


async def run_generation(
    ctx: AppContext,
    engine_id: str,
    text: str,
    voice_id: str | None = None,
    params: dict | None = None,
    generation_id: str | None = None,
) -> dict:
    """One complete synthesis run through the shared pipeline."""
    registry = ctx.registry
    engine: Engine | None = ctx.require_engine(engine_id or "")

    params = dict(params or {})
    voice = None
    if voice_id:
        voice = ctx.require_voice(voice_id)
        if not voice.get("reference"):
            # Designed voices always carry their preview sample after a
            # successful design run; a record without one is unusable.
            raise HTTPException(
                status_code=409,
                detail="该音色没有参考样本，无法参与生成；请删除后重新创建",
            )
        ref_path = ctx.voice_store.reference_path(voice_id)
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
                        None, ctx.transcribe_voice_sync, voice["id"], None
                    )
                except TranscriptionError as exc:
                    raise HTTPException(
                        status_code=409,
                        detail=f"该引擎需要参考文本，自动转写失败：{exc}",
                    ) from exc
                ctx.voice_store.set_transcript(voice["id"], transcript)
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
        if ctx.keys.get(engine.engine_id) is None:
            # Vendor copy comes from the engine itself (ADR-0015 decision 4):
            # a MiniMax engine must never be told it needs 阿里百炼.
            raise HTTPException(status_code=409, detail=engine.key_missing_hint())
        if voice is None:
            raise HTTPException(
                status_code=422,
                detail=f"云端引擎 {engine.engine_id} 必须使用音色生成"
                "（复刻模型需要一个已绑定的云端音色）",
            )
        cloud_voice_id = (voice["bindings"].get(engine.engine_id) or {}).get("voice_id")

        # Issue #12: cloud vendors silently recycle enrolled voices. A
        # binding that says "ready" is only a cache — probe the vendor
        # BEFORE relying on it, and rebuild from the local reference the
        # moment the voice is gone. The probe only runs on engines that
        # actually implement a health check.
        rebuilding = cloud_voice_id is not None and (
            type(engine).check_voice is not Engine.check_voice
        )
        if rebuilding:
            _health_log = ctx.log_from_thread(generation_id)

            try:
                alive = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: engine.check_voice(cloud_voice_id, _health_log)
                )
            except CloudEngineError as exc:
                # Not a verdict about the voice (bad key, outage) — the
                # existing binding stays untouched and the run fails.
                raise HTTPException(
                    status_code=502, detail=f"云端音色健康检查失败：{exc}"
                ) from exc
            if not alive:
                ctx.log_bus.publish(
                    generation_id,
                    f"云端音色已失效（{engine.engine_id}），正在用本地参考音频自动重建绑定…",
                )
                cloud_voice_id = None  # fall through to re-enrollment

        if not cloud_voice_id:
            ref_path = ctx.voice_store.reference_path(voice["id"])
            transcript = params.get("ref_text") or (voice.get("reference") or {}).get("transcript")

            _log = ctx.log_from_thread(generation_id)

            try:
                binding_extra = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: engine.bind_reference(ref_path, transcript, _log),
                )
            except CloudEngineError as exc:
                if rebuilding:
                    # The dead binding could not be rebuilt: mark THIS
                    # binding unavailable with the vendor's reason. Other
                    # engines' bindings are separate records and stay
                    # ready (issue #12).
                    ctx.voice_store.bind(
                        voice["id"],
                        engine.engine_id,
                        status="unavailable",
                        extra={"error": str(exc)},
                    )
                    raise HTTPException(
                        status_code=502,
                        detail=f"云端音色重建失败（绑定已标记不可用）：{exc}",
                    ) from exc
                raise HTTPException(
                    status_code=502, detail=f"云端音色创建失败：{exc}"
                ) from exc
            cloud_voice_id = binding_extra["voice_id"]
            # Persist the cloud binding the moment enrollment succeeds —
            # the vendor-side voice now exists regardless of what happens
            # to this synthesis run. Waiting for synthesis would leak an
            # orphaned cloud voice on every failed run.
            ctx.voice_store.bind(voice["id"], engine.engine_id, status="ready", extra=binding_extra)
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
    ctx.generation_store.persist(record)

    # Synthesis runs in a worker thread: it can take seconds and must not
    # block the event loop (WS log streaming + other requests keep working).
    loop = asyncio.get_running_loop()
    publish = ctx.log_from_thread(generation_id)

    def log(message: str) -> None:
        # Called from the worker thread — hop back onto the loop, and keep
        # the line in the record so history can replay it later.
        record["logs"].append({"ts": now_iso(), "message": message})
        publish(message)

    def _looks_like_rate_limit(exc: Exception) -> bool:
        # Only vendor (cloud) errors count — a local crash that happens
        # to mention "429" must not be retried (issue #13 review).
        if not isinstance(exc, CloudEngineError):
            return False
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in ("429", "限流", "throttl", "rate limit", "ratelimit", "toomanyrequests")
        )

    def run_synthesis():
        # setdefault stores the per-engine lock so later generations of the
        # same engine serialize behind it, even after a worker crash.
        request = GenerationRequest(
            generation_id=generation_id, text=normalized_text, params=record["params"]
        )

        def attempt() -> object:
            with ctx.generation_locks.setdefault(engine.engine_id, threading.Lock()):
                return engine.synthesize(request, log)

        # Issue #13: a vendor-side throttle is queue pressure, not a
        # failure the user should ever see. Retry with backoff — while
        # holding the per-engine lock each time, which is the throttle's
        # natural cure (one request at a time). Any other error, or the
        # attempts exhausted, propagates unchanged.
        backoff = ctx.rate_limit_backoff
        for attempt_no in range(1, ctx.rate_limit_attempts + 1):
            try:
                return attempt()
            except Exception as exc:  # noqa: BLE001 - re-raised below
                if (
                    not engine.requires_key
                    or attempt_no >= ctx.rate_limit_attempts
                    or not _looks_like_rate_limit(exc)
                ):
                    raise
                wait = min(backoff, 10.0)
                log(
                    f"云端限流，{wait:.1f}s 后自动重试"
                    f"（第 {attempt_no}/{ctx.rate_limit_attempts - 1} 次）…"
                )
                time.sleep(wait)
                backoff *= 2

    try:
        result = await loop.run_in_executor(None, run_synthesis)
    except Exception as exc:  # noqa: BLE001 - recorded in history, surfaced on the contract
        record.update(
            status="failed",
            error=str(exc),
            finished_at=now_iso(),
            duration_seconds=round(time.monotonic() - started_monotonic, 3),
        )
        ctx.log_bus.publish(generation_id, f"generation failed: {exc}")
        ctx.generation_store.persist(record)
        raise HTTPException(status_code=500, detail="generation failed") from exc

    audio_name = Path(result.audio_path).name
    # Issue #15: stamp the artifact itself with the AI-generated-content
    # marker so the disclosure travels with the file, not just the record.
    ctx.mark_artifact(
        record,
        ctx.audio_dir / audio_name,
        {
            "generation": generation_id,
            "voice": (voice or {}).get("name", ""),
            "engine": engine.engine_id,
            "model": result.model_version or "",
            "date": record.get("created_at", ""),
        },
    )
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
        # rides along as binding extras. Extras minted by EARLIER runs
        # must survive: re-binding with extra=None would silently drop
        # the cloud voice_id and force a re-enroll (an orphaned cloud
        # voice) on every subsequent run. Only status/created_at are
        # refreshed; a stale "error" from a marked-unavailable binding
        # is cleared now that the run succeeded.
        previous = voice["bindings"].get(engine.engine_id) or {}
        carried = {
            k: v for k, v in previous.items()
            if k not in {"status", "created_at", "reference_sha256", "error"}
        }
        ctx.voice_store.bind(
            voice["id"],
            engine.engine_id,
            status="ready",
            extra={**carried, **(binding_extra or {})},
        )
        record["binding"] = ctx.voice_store.get(voice["id"])["bindings"].get(engine.engine_id)
    return ctx.generation_store.persist(record)
