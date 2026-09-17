"""Blind comparison sessions + the local preference profile (issue #10).

A compare session is one voice, one text, several engines: each engine
produces a generation through the same generation pipeline, every artifact is
then gain-normalized to a shared LUFS target (ADR-0009: without equal
loudness "louder" always wins), and entries get shuffled blind labels
(A/B/C…) so the scorer cannot tell engines apart.

Scoring collects star ratings per blind label. Aggregated across sessions,
ratings form the preference profile: per language and per text type, which
engine the user actually prefers. The profile is a READ-ONLY insight —
nothing here feeds back into default-engine selection, capability-gated
routing or any automatic choice (ADR-0009: preferences are for humans, not
for the scheduler).

Storage layout:

    <data>/compare.json     session index (atomic writes)
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from .storage import write_json_atomic

# Fixed, closed text-type vocabulary: free-form tags would fragment the
# profile into singletons it can never aggregate over.
TEXT_TYPES = ("general", "narration", "dialogue", "news", "poetry")
DEFAULT_TEXT_TYPE = "general"

MIN_RATING = 1
MAX_RATING = 5

ALL_LABELS = "ABCDEFGH"


class CompareError(ValueError):
    """Raised on invalid comparison input; message is user-facing."""


def detect_language(text: str) -> str:
    """Coarse text language: 'zh' when CJK dominates the lettered content.

    Deliberately a two-way zh/en split (the engines' language axis); CJK here
    includes kana/hangul so those scripts count toward the CJK bucket.
    """
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    words = len(re.findall(r"[A-Za-z]+", text))
    if cjk + words == 0:
        return "zh"
    # Each CJK character carries a word's worth of meaning, so character
    # count competes against LATIN WORD count, not letter count ("你好 world"
    # is a Chinese sentence with a borrowed word, not an English one).
    return "zh" if cjk >= words else "en"


class CompareStore:
    """Persistence for compare sessions; ratings live inside each session."""

    def __init__(self, data_dir: Path) -> None:
        self.index_path = Path(data_dir) / "compare.json"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for session in raw.get("sessions", []):
            if isinstance(session, dict) and session.get("id"):
                self._sessions[session["id"]] = session

    def _save(self) -> None:
        ordered = sorted(
            self._sessions.values(), key=lambda s: s.get("created_at", ""), reverse=True
        )
        write_json_atomic(self.index_path, {"sessions": ordered})

    # -- CRUD ----------------------------------------------------------------

    def create(self, session: dict) -> dict:
        with self._lock:
            self._sessions[session["id"]] = session
            self._save()
        return dict(session)

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return dict(session) if session else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            sessions = sorted(
                self._sessions.values(),
                key=lambda s: s.get("created_at", ""),
                reverse=True,
            )
        return [dict(s) for s in sessions[:limit]]

    def update(self, session: dict) -> dict:
        with self._lock:
            if session["id"] not in self._sessions:
                raise KeyError(session["id"])
            self._sessions[session["id"]] = session
            self._save()
        return dict(session)

    def delete(self, session_id: str) -> dict:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                raise KeyError(session_id)
            self._save()
        return session

    # -- scoring --------------------------------------------------------------

    @staticmethod
    def set_scores(session: dict, scores: dict[str, int]) -> dict:
        """Attach ratings keyed by blind label; validates every value."""
        labels = {e["label"] for e in session["entries"]}
        for label, rating in scores.items():
            if label not in labels:
                raise CompareError(f"未知对比项：{label!r}")
            if not isinstance(rating, int) or not MIN_RATING <= rating <= MAX_RATING:
                raise CompareError(f"评分必须是 {MIN_RATING}–{MAX_RATING} 的整数")
        for entry in session["entries"]:
            if entry["label"] in scores:
                entry["score"] = scores[entry["label"]]
        session["scored"] = all("score" in e for e in session["entries"])
        return session

    # -- the preference profile -----------------------------------------------

    def preference_profile(self) -> dict:
        """Aggregate scores by (language, text type) × engine.

        Pure read-side statistics over stored sessions. Nothing here writes
        back or influences any routing/default decision.
        """
        with self._lock:
            sessions = [dict(s) for s in self._sessions.values()]

        buckets: dict[tuple[str, str], dict[str, list[int]]] = {}
        for session in sessions:
            for entry in session.get("entries", []):
                score = entry.get("score")
                if not isinstance(score, int):
                    continue
                engine_id = entry.get("engine_id")
                if not engine_id:
                    continue  # entry predating a reveal-eligible state
                key = (session.get("language") or "zh", session.get("text_type") or DEFAULT_TEXT_TYPE)
                buckets.setdefault(key, {}).setdefault(engine_id, []).append(score)

        cells = []
        for (language, text_type), engines in sorted(buckets.items()):
            rows = []
            for engine_id, scores in engines.items():
                mean = sum(scores) / len(scores)
                rows.append(
                    {
                        "engine_id": engine_id,
                        "average_score": round(mean, 2),
                        "score_count": len(scores),
                        "wins": sum(1 for s in scores if s == MAX_RATING),
                    }
                )
            rows.sort(key=lambda r: (-r["average_score"], -r["score_count"]))
            cells.append({"language": language, "text_type": text_type, "engines": rows})
        return {"cells": cells}

