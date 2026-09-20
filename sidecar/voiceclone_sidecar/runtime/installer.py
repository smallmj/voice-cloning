"""Per-step install orchestrator with durable state and clean retries.

Each engine install is a list of named steps, each declaring the artifact
directory it owns. State is persisted to ``install_state.json`` inside the
engine directory, so:

- a completed step is never redone (weights already on disk stay there);
- a failed step's artifacts are DELETED before the retry — a half-written
  venv or a truncated weights file must never be trusted (ADR-0002);
- the UI can render per-step status from the same JSON.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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


def run_install(ctx: InstallContext, steps: list[InstallStep], log: LogFn, progress: ProgressFn | None = None) -> dict:
    progress = progress or (lambda *a: None)
    state = load_state(ctx)
    record = state.setdefault("steps", {})

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
            step.run(log, progress)
        except Exception as exc:  # noqa: BLE001 - recorded, surfaced, retriable
            entry.update(status="failed", error=str(exc), finished_at=time.time())
            save_state(ctx, state)
            log(f"[{step.id}] FAILED: {exc}")
            raise
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
