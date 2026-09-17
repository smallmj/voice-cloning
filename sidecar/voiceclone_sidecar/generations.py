"""Generation records: persistent lineage for every synthesis run.

Every generation — success or failure — is written to a durable index with
enough lineage (text, voice, engine + model version, every parameter, logs,
timing, cost, artifact) to reproduce the run later (ADR-0001: the record is
the source of truth for history; the audio file is its artifact).

Storage layout:

    <data>/generations.json              the index (atomic writes)
    <audio_dir>/<generation_id>.wav      the artifact, served via /audio/

Deleting a record deletes its artifact file; deleting a voice does NOT touch
records — history must keep describing what actually happened, and each
record snapshots the voice's name so it stays readable after the voice is
renamed or deleted.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


class GenerationStore:
    """Persistence for generation records."""

    def __init__(self, data_dir: Path, audio_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.audio_dir = Path(audio_dir)
        self.index_path = self.data_dir / "generations.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict] = {}
        # Writes come from request handlers and synthesis worker threads.
        self._lock = threading.Lock()
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for record in raw.get("generations", []):
            if isinstance(record, dict) and record.get("id"):
                self._records[record["id"]] = record

    def _save(self) -> None:
        # Newest first, so a restarted sidecar reads history in display order.
        ordered = sorted(
            self._records.values(), key=lambda r: r.get("created_at", ""), reverse=True
        )
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"generations": ordered}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.index_path)

    # -- CRUD ----------------------------------------------------------------

    def persist(self, record: dict) -> dict:
        """Insert or update one record and flush it to disk."""
        with self._lock:
            self._records[record["id"]] = record
            self._save()
        return dict(record)

    def get(self, generation_id: str) -> dict | None:
        with self._lock:
            record = self._records.get(generation_id)
            return dict(record) if record else None

    def list(
        self,
        q: str = "",
        engine_id: str | None = None,
        voice_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Search + filter + paginate. ``q`` matches text, voice name or id."""
        with self._lock:
            records = list(self._records.values())
        needle = q.strip().lower()
        if needle:
            records = [
                r
                for r in records
                if needle in (r.get("text") or "").lower()
                or needle in (r.get("voice_name") or "").lower()
                or needle in r["id"].lower()
            ]
        if engine_id:
            records = [r for r in records if r.get("engine_id") == engine_id]
        if voice_id:
            records = [r for r in records if r.get("voice_id") == voice_id]
        if status:
            records = [r for r in records if r.get("status") == status]
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        total = len(records)
        page = records[offset : offset + limit]
        return {"records": [dict(r) for r in page], "total": total}

    def delete(self, generation_id: str) -> None:
        """Remove the record and its audio artifact."""
        with self._lock:
            record = self._records.pop(generation_id, None)
            if record is None:
                raise KeyError(generation_id)
            self._save()
        audio_name = record.get("audio_file")
        if audio_name:
            try:
                (self.audio_dir / audio_name).unlink()
            except OSError:
                pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
