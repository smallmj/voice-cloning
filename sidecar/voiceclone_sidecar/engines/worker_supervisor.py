"""Persistent worker-process supervisor: crash restart, idle VRAM release.

Several local engines (IndexTTS-2.5 among them) take tens of seconds to load
their model, so respawning a fresh process per generation would be wasteful —
but a long-lived worker must never pin GPU memory forever. The supervisor owns
one worker subprocess and enforces the three reclamation levers the spec
requires:

- **explicit unload** — ``unload()`` asks the worker to free its model and
  exits the process;
- **idle timeout** — no request for ``idle_timeout_s`` terminates the worker
  (process exit is the only reliable VRAM release);
- **request threshold** — after ``max_requests`` generations the worker is
  recycled, bounding any slow leak.

Crash handling: a worker that dies between requests is restarted on demand; a
worker that dies *during* a request fails that request once and the very next
request transparently restarts the worker — queued requests are never lost.
A worker that refuses to boot because the CUDA gate rejects the machine is
marked degraded: restarts are pointless until the user fixes the environment,
so the error is raised verbatim instead of being retried into noise.

Protocol (over the worker's stdin/stdout, one JSON per line):

- stdin: ``{"id": "<request id>", "action": "synthesize"|"unload"|"shutdown", ...}``
- stdout: log lines verbatim; per request exactly one of
  ``RESULT: {json}`` or ``WORKER_ERROR: {json}`` where the JSON carries the
  request ``id``.

Transport: a dedicated reader thread drains the worker's stdout into a queue.
select() is not an option — it cannot see bytes already sitting in Python's
buffered-reader prefetch (a race that silently eats replies), and it does not
work on Windows pipes at all.
"""

from __future__ import annotations

import contextlib
import json
import queue
import subprocess
import threading
import time
import uuid


class WorkerDegradedError(RuntimeError):
    """The worker refused to start (CUDA gate rejection). Retrying is futile."""


class WorkerSupervisor:
    def __init__(
        self,
        command: list[str],
        cwd: str | None = None,
        env: dict | None = None,
        log=None,
        idle_timeout_s: float = 300.0,
        max_requests: int = 20,
        request_timeout_s: float = 3600.0,
    ) -> None:
        self._command = command
        self._cwd = cwd
        self._env = env
        self._log = log or (lambda msg: None)
        self._idle_timeout_s = idle_timeout_s
        self._max_requests = max_requests
        self._request_timeout_s = request_timeout_s

        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue | None = None
        self._degraded_reason: str | None = None
        self._requests_served = 0
        self._last_used = time.monotonic()
        self._reaper: threading.Thread | None = None
        self._closed = False

    # -- public API ---------------------------------------------------------

    def request(self, payload: dict, on_log=None, timeout_s: float | None = None) -> dict:
        """Send one request, return the worker's RESULT payload.

        Serializes all requests through one worker. If the worker died since
        the last request it is restarted first; if it dies mid-request the
        request is retried exactly once on a fresh worker before failing.
        """
        req_id = str(payload.get("id") or uuid.uuid4().hex)
        payload = {**payload, "id": req_id}
        timeout_s = timeout_s if timeout_s is not None else self._request_timeout_s
        deadline = time.monotonic() + timeout_s

        with self._lock:
            self._raise_if_degraded()
            self._ensure_alive()
            try:
                return self._exchange(payload, on_log, deadline)
            except _WorkerDied:
                self._log(f"worker died mid-request; restarting and retrying {req_id}")
                self._raise_if_degraded()
                self._ensure_alive()
                return self._exchange(payload, on_log, deadline)

    def unload(self) -> None:
        """Explicit VRAM release: ask the worker to free its model, then exit."""
        with self._lock:
            if self._proc is None:
                return
            # Unload failure is fine: process exit is the real release.
            with contextlib.suppress(Exception):
                self._exchange({"action": "unload"}, None, time.monotonic() + 60)
            self._terminate()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            self._terminate()

    def stats(self) -> dict:
        with self._lock:
            return {
                "alive": self._proc is not None and self._proc.poll() is None,
                "degraded": self._degraded_reason is not None,
                "requests_served": self._requests_served,
                "idle_seconds": round(time.monotonic() - self._last_used, 1),
            }

    # -- internals ----------------------------------------------------------

    def _raise_if_degraded(self) -> None:
        if self._degraded_reason is not None:
            raise WorkerDegradedError(self._degraded_reason)

    def _ensure_alive(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._closed:
            raise RuntimeError("worker supervisor is shut down")
        if self._degraded_reason is not None:
            raise WorkerDegradedError(self._degraded_reason)
        self._spawn()

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=self._cwd,
            env=self._env,
        )
        self._lines = queue.Queue()
        threading.Thread(
            target=self._read_stdout, args=(self._proc, self._lines),
            daemon=True, name="worker-stdout-reader",
        ).start()
        self._requests_served = 0
        self._last_used = time.monotonic()
        self._start_reaper()

    @staticmethod
    def _read_stdout(proc: subprocess.Popen, lines: queue.Queue) -> None:
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)  # EOF marker

    def _exchange(self, payload: dict, on_log, deadline: float) -> dict:
        """One request/response round trip against the live worker."""
        proc = self._proc
        assert proc is not None and proc.stdin is not None and self._lines is not None
        req_id = payload.get("id")
        try:
            proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise _WorkerDied(f"could not write to worker: {exc}") from exc

        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"worker did not answer request {req_id} within the timeout")
            try:
                raw = self._lines.get(timeout=min(deadline - time.monotonic(), 0.5))
            except queue.Empty:
                continue
            if raw is None:  # EOF: the worker is gone
                raise _WorkerDied(f"worker exited with code {proc.returncode} before answering {req_id}")
            line = raw.rstrip()
            if not line:
                continue
            if line.startswith("RESULT: "):
                result = json.loads(line[len("RESULT: "):])
                if result.get("id") != req_id:
                    continue
                self._requests_served += 1
                self._last_used = time.monotonic()
                if self._requests_served >= self._max_requests:
                    self._log("worker reached the request threshold; recycling it")
                    self._terminate()
                return result
            if line.startswith("WORKER_ERROR: "):
                error = json.loads(line[len("WORKER_ERROR: "):])
                if error.get("id") not in (None, req_id):
                    continue
                if error.get("fatal"):
                    self._degraded_reason = error.get("error", "worker refused to start")
                    self._terminate()
                    raise WorkerDegradedError(self._degraded_reason)
                raise RuntimeError(error.get("error", "worker request failed"))
            if on_log:
                on_log(line)
            else:
                self._log(line)

    def _terminate(self) -> None:
        proc, self._proc, self._lines = self._proc, None, None
        self._reaper = None  # the reaper exits on its next tick; respawn lazily
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        for stream in (proc.stdout, proc.stdin):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass

    def _start_reaper(self) -> None:
        if self._idle_timeout_s <= 0 or self._reaper is not None:
            return

        def reap():
            while True:
                time.sleep(min(self._idle_timeout_s, 5.0))
                with self._lock:
                    if self._closed or self._proc is None:
                        return
                    if time.monotonic() - self._last_used >= self._idle_timeout_s:
                        self._log("worker idle past the timeout; releasing its GPU memory")
                        self._terminate()
                        return  # respawned lazily by the next request

        self._reaper = threading.Thread(target=reap, daemon=True, name="worker-idle-reaper")
        self._reaper.start()


class _WorkerDied(RuntimeError):
    """Internal: the worker process vanished mid-exchange."""
