"""Long-text generation jobs: queue state, visibility, cancellation (#13).

The queue is runtime state held in memory on purpose: a queued job is a plan
over text the user still has in front of them, and each executed segment is
a REAL generation record in the durable history — restart the sidecar and
what survives is the per-segment lineage, not a half-dead job row. The
worker itself lives in main.py (it calls back into the shared pipeline).
"""

from __future__ import annotations

import threading
import uuid

# Terminal statuses: cancel is refused, no worker will ever touch them again.
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


def new_job(
    engine_id: str,
    voice_id: str | None,
    voice_name: str | None,
    text: str,
    segments: list[str],
) -> dict:
    """Build one queued job. Private keys (``_``-prefixed) never leave the
    sidecar — ``public_view`` strips them from every HTTP response."""
    return {
        "id": uuid.uuid4().hex,
        "engine_id": engine_id,
        "voice_id": voice_id,
        "voice_name": voice_name,
        "text": text,
        "status": "queued",
        "segment_count": len(segments),
        "segments": [
            {"index": i, "text": seg, "status": "pending", "generation_id": None, "error": None}
            for i, seg in enumerate(segments)
        ],
        "audio_url": None,
        "audio_file": None,
        "sample_rate": None,
        "error": None,
        "created_at": None,  # stamped by the caller (now_iso)
        "started_at": None,
        "finished_at": None,
        # Runtime-only, never serialized.
        "_cancel_requested": False,
    }


class JobStore:
    """Thread-safe in-memory job table + FIFO order."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def add(self, job: dict) -> dict:
        with self._lock:
            self._jobs[job["id"]] = job
            self._order.append(job["id"])
        return job

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._jobs[i] for i in ids]

    def next_queued(self) -> dict | None:
        """Pop the oldest non-terminal job for the worker."""
        with self._lock:
            for job_id in self._order:
                job = self._jobs[job_id]
                if job["status"] == "queued":
                    return job
        return None

    def request_cancel(self, job_id: str) -> dict | None:
        """Set the cancel flag. Returns the job, or None when unknown or
        already terminal (a terminal job can no longer react to cancel)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] in TERMINAL_STATUSES:
                return None
            job["_cancel_requested"] = True
            return job


def public_view(job: dict) -> dict:
    """Strip runtime-only fields from a job before it crosses HTTP."""
    return {k: v for k, v in job.items() if not k.startswith("_")}


def is_cancel_requested(job: dict) -> bool:
    return bool(job.get("_cancel_requested"))
