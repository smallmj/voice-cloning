"""Blind comparison + preference profile (issue #10, ADR-0009).

One voice + one text -> several engines side by side. Every artifact is
gain-normalized to the same LUFS target so loudness cannot bias the
listener; entries carry shuffled blind labels until revealed. Ratings
aggregate into a read-only preference profile — it never drives any
default-engine selection or routing decision.
"""

from __future__ import annotations

import asyncio
import random
import uuid

from fastapi import APIRouter, Depends, HTTPException

from ..compare import (
    ALL_LABELS,
    CompareError,
    DEFAULT_TEXT_TYPE,
    TEXT_TYPES,
    detect_language,
)
from ..context import AppContext
from ..generations import now_iso
from ..loudness import TARGET_LUFS, LoudnessError, normalize_wav_lufs
from ..pipeline import run_generation


def build_router(ctx: AppContext) -> APIRouter:
    router = APIRouter()
    compare_store = ctx.compare_store
    audio_dir = ctx.audio_dir
    _require_voice = ctx.require_voice

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

    @router.post("/compare", dependencies=[Depends(ctx.require_auth)])
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
            ctx.require_engine(engine_id)

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
                record = await run_generation(ctx, engine_id, text, voice_id, None)
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

    @router.get("/compare", dependencies=[Depends(ctx.require_auth)])
    async def list_compares(reveal: bool = False, limit: int = 50) -> dict:
        sessions = compare_store.list(limit)
        return {
            "sessions": [
                _blind_view(s, reveal or s.get("scored", False)) for s in sessions
            ]
        }

    @router.get("/compare/{session_id}", dependencies=[Depends(ctx.require_auth)])
    async def get_compare(session_id: str, reveal: bool = False) -> dict:
        session = compare_store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="comparison session not found")
        return _blind_view(session, reveal or session.get("scored", False))

    @router.post("/compare/{session_id}/scores", dependencies=[Depends(ctx.require_auth)])
    async def score_compare(session_id: str, body: dict) -> dict:
        session = compare_store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="comparison session not found")
        scores = body.get("scores")
        if not isinstance(scores, dict) or not scores:
            raise HTTPException(status_code=422, detail="scores must be a {label: rating} object")
        try:
            session = compare_store.set_scores(session, scores)
        except CompareError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _blind_view(compare_store.update(session), reveal=session["scored"])

    @router.delete("/compare/{session_id}", dependencies=[Depends(ctx.require_auth)])
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

    @router.get("/preferences", dependencies=[Depends(ctx.require_auth)])
    async def preference_profile() -> dict:
        """The local preference profile — insight only, never routing input."""
        return compare_store.preference_profile()

    return router
