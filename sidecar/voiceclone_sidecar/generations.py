"""Generation records: persistent lineage for every synthesis run.

Every generation — success or failure — is written to a durable index with
enough lineage (text, voice, engine + model version, every parameter, logs,
timing, cost, artifact) to reproduce the run later (ADR-0001: the record is
the source of truth for history; the audio file is its artifact).

Storage layout:

    <data>/library.db                    the SQLite index (generations table)
    <audio_dir>/<generation_id>.wav      the artifact, served via /audio/

Deleting a record deletes its artifact file; deleting a voice does NOT touch
records — history must keep describing what actually happened, and each
record snapshots the voice's name so it stays readable after the voice is
renamed or deleted.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

from .db import T_GENERATIONS, Database, library_db_path


class GenerationStore:
    """Persistence for generation records.

    Filtering, search and pagination run in SQL over indexed columns
    (ADR-0017: never a full in-lock linear scan per keystroke). Reads always
    return freshly parsed objects — a record handed to the pipeline is never
    aliased into the store, so later in-place mutations (log appends from
    worker threads) cannot race a serialization pass.
    """

    def __init__(self, data_dir: Path, audio_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.audio_dir = Path(audio_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db = Database(library_db_path(self.data_dir))

    # -- persistence ---------------------------------------------------------

    def reload(self) -> None:
        """Reconnect to the index database (used by library restore #14)."""
        self._db.reconnect()

    # -- CRUD ----------------------------------------------------------------

    def persist(self, record: dict) -> dict:
        """Insert or update one record; the payload is snapshotted at write
        time, so subsequent caller-side mutations never reach the index."""
        self._db.put_payload(
            T_GENERATIONS,
            record["id"],
            record.get("created_at", ""),
            record,
            {
                "engine_id": record.get("engine_id") or "",
                "voice_id": record.get("voice_id") or "",
                "status": record.get("status") or "",
                "text": record.get("text") or "",
                "voice_name": record.get("voice_name") or "",
            },
        )
        return dict(record)

    def get(self, generation_id: str) -> dict | None:
        return self._db.get_payload(T_GENERATIONS, generation_id)

    def list(
        self,
        q: str = "",
        engine_id: str | None = None,
        voice_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Search + filter + paginate in SQL. ``q`` matches text, voice name or id."""
        where: list[str] = []
        params: list = []
        needle = q.strip().lower()
        if needle:
            like = f"%{needle}%"
            where.append(
                "(lower(text) LIKE ? OR lower(voice_name) LIKE ? OR lower(id) LIKE ?)"
            )
            params += [like, like, like]
        if engine_id:
            where.append("engine_id = ?")
            params.append(engine_id)
        if voice_id:
            where.append("voice_id = ?")
            params.append(voice_id)
        if status:
            where.append("status = ?")
            params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        total = self._db.execute(
            f"SELECT COUNT(*) AS n FROM {T_GENERATIONS} {clause}", tuple(params)
        ).fetchone()["n"]
        rows = self._db.execute(
            f"SELECT payload FROM {T_GENERATIONS} {clause} "
            "ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            (*tuple(params), limit, offset),
        ).fetchall()
        return {"records": [json.loads(r["payload"]) for r in rows], "total": total}

    def delete(self, generation_id: str) -> None:
        """Remove the record and its audio artifact."""
        record = self.get(generation_id)
        if record is None or not self._db.delete_payload(T_GENERATIONS, generation_id):
            raise KeyError(generation_id)
        audio_name = record.get("audio_file")
        if audio_name:
            with contextlib.suppress(OSError):
                (self.audio_dir / audio_name).unlink()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
