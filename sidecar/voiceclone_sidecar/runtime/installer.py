"""Per-step install orchestrator with durable state and clean retries.

Each engine install is a list of named steps, each declaring the artifact
directory it owns. State is persisted to ``install_state.json`` inside the
engine directory, so:

- a completed step is never redone (weights already on disk stay there);
- a failed step's artifacts are DELETED before the retry — a half-written
  venv or a truncated weights file must never be trusted (ADR-0002);
- the UI can render per-step status from the same JSON;
- byte-level download progress is persisted to the same JSON as
  ``progress: {file, done_bytes, total_bytes}`` (issue #35), so the status
  endpoint the frontend polls once a second shows a real progress bar
  without any new endpoint. The field is written throttled (the poll
  cadence makes sub-second writes pointless) and CLEARED when a step
  completes or fails — a finished or dead install must never advertise a
  stale byte count.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .paths import engine_dir

STATE_FILE = "install_state.json"

LogFn = Callable[[str], None]
ProgressFn = Callable[[str, int, int | None], None]


@dataclass
class InstallStep:
    id: str
    description: str
    run: Callable[[LogFn, ProgressFn], None]
    artifact: Path | None = None  # owned dir; wiped before a retry


@dataclass
class InstallContext:
    engine_id: str
    root: Path
    env: dict = field(default_factory=dict)  # mirror overrides etc.

    @property
    def dir(self) -> Path:
        d = engine_dir(self.root, self.engine_id)
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_state(ctx: InstallContext) -> dict:
    path = ctx.dir / STATE_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass  # corrupt state is treated as no state
    return {"steps": {}}


def save_state(ctx: InstallContext, state: dict) -> None:
    path = ctx.dir / STATE_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    tmp.replace(path)  # atomic: a crash never leaves a half-written state


def _persist_progress(
    ctx: InstallContext,
    state: dict,
    last_save: dict,
    name: str,
    done: int,
    total: int | None,
    force: bool,
) -> None:
    """Write the current download position into the durable install state.

    Throttled to one write a second (the frontend polls at exactly that
    cadence, so faster writes are pure disk churn); ``force`` flushes
    immediately, used for the final byte count and the clear.
    """
    now = time.time()
    if not force and now - last_save.get("t", 0.0) < 1.0:
        return
    last_save["t"] = now
    state["progress"] = {"file": name, "done_bytes": done, "total_bytes": total}
    save_state(ctx, state)


def run_install(
    ctx: InstallContext, steps: list[InstallStep], log: LogFn, progress: ProgressFn | None = None
) -> dict:
    progress = progress or (lambda *a: None)
    state = load_state(ctx)
    # Stale progress from a killed previous run must never be advertised.
    state.pop("progress", None)
    save_state(ctx, state)
    record = state.setdefault("steps", {})
    last_save: dict = {"t": 0.0}

    def tracked(name: str, done: int, total: int | None) -> None:
        # Unknown total size: every report is flushed, so the UI still sees
        # movement (a 1s throttle on the only writes would freeze the bar).
        force = total is None or done >= total
        _persist_progress(ctx, state, last_save, name, done, total, force=force)
        progress(name, done, total)

    # Pre-pass: a failed step's artifact is DELETED before the retry, and
    # every completed step sharing that artifact is reset to pending — its
    # artifact was just wiped (observed, issue #25: the torch step shares
    # the venv artifact with the engine step; a failed engine step wiped
    # the venv while torch stayed "completed", so the retry produced a venv
    # without torch). This must happen in a PRE-PASS: within the main loop
    # the shared earlier step is reached (and skipped) before the failed
    # step's cleanup runs.
    for step in steps:
        entry = record.setdefault(step.id, {"status": "pending"})
        if entry["status"] == "failed" and step.artifact and step.artifact.exists():
            log(f"[{step.id}] cleaning artifacts of the failed attempt: {step.artifact}")
            shutil.rmtree(step.artifact, ignore_errors=True)
            entry["status"] = "pending"
            for other in steps:
                if (
                    other.id != step.id
                    and other.artifact is not None
                    and other.artifact == step.artifact
                    and record[other.id].get("status") == "completed"
                ):
                    record[other.id] = {"status": "pending"}
                    log(f"[{other.id}] artifact was wiped — rerunning this step on retry")

    for step in steps:
        entry = record.setdefault(step.id, {"status": "pending"})
        if entry["status"] == "completed":
            log(f"[{step.id}] already installed, skipping")
            continue
        entry.update(status="running", error=None, started_at=time.time())
        save_state(ctx, state)
        log(f"[{step.id}] {step.description}")
        try:
            step.run(log, tracked)
        except Exception as exc:
            state.pop("progress", None)  # no fake progress after a failure
            entry.update(status="failed", error=str(exc), finished_at=time.time())
            save_state(ctx, state)
            log(f"[{step.id}] FAILED: {exc}")
            raise
        state.pop("progress", None)  # a completed step has no live download
        entry.update(status="completed", finished_at=time.time())
        save_state(ctx, state)

    state["installed"] = True
    state["installed_at"] = time.time()
    save_state(ctx, state)
    log("install complete")
    return state


def reset_failed(ctx: InstallContext) -> None:
    """Drop state so a fresh install can be attempted from scratch."""
    save_state(ctx, {"steps": {}})
