"""WorkerSupervisor tests against a real subprocess running a stub worker.

The stub answers the supervisor protocol; mode flags let each test stage a
different failure (boot refusal, mid-request crash, idle exit).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from voiceclone_sidecar.engines.worker_supervisor import WorkerDegradedError, WorkerSupervisor

STUB = textwrap.dedent(
    """
    import json, os, sys
    mode = sys.argv[1]
    marker = sys.argv[2]
    if mode == "gate":
        print("WORKER_ERROR: " + json.dumps({"id": None, "fatal": True, "error": "CUDA gate: no usable GPU"}), flush=True)
        sys.exit(3)
    if mode == "bootfail":
        sys.exit(1)
    print("ready", flush=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        req = json.loads(line)
        if mode == "crash_once" and req.get("text") == "crash" and not os.path.exists(marker):
            open(marker, "w").close()
            os._exit(1)
        if req.get("action") == "die":
            os._exit(1)
        if req.get("action") == "die_on_command":
            os._exit(1)
        if req.get("action") == "unload":
            print("RESULT: " + json.dumps({"id": req["id"], "unloaded": True}), flush=True)
            continue
        print("RESULT: " + json.dumps({"id": req["id"], "echo": req.get("text", "")}), flush=True)
    """
)


@pytest.fixture()
def stub(tmp_path: Path) -> Path:
    path = tmp_path / "stub_worker.py"
    path.write_text(STUB)
    return path


def make(tmp_path: Path, stub: Path, mode: str, **kwargs) -> WorkerSupervisor:
    return WorkerSupervisor(
        command=[sys.executable, "-u", str(stub), mode, str(tmp_path / "marker")],
        cwd=str(tmp_path),
        **kwargs,
    )


def test_request_roundtrip(tmp_path, stub):
    sup = make(tmp_path, stub, "echo")
    result = sup.request({"action": "synthesize", "text": "hello"})
    assert result["echo"] == "hello"
    assert sup.stats()["requests_served"] == 1
    sup.shutdown()


def test_mid_request_crash_restarts_and_retries(tmp_path, stub):
    sup = make(tmp_path, stub, "crash_once")
    # The worker dies mid-request; the supervisor restarts it and the SAME
    # request succeeds on the fresh worker.
    result = sup.request({"action": "synthesize", "text": "crash"})
    assert result["echo"] == "crash"
    sup.shutdown()


def test_crash_between_requests_is_recovered(tmp_path, stub):
    sup = make(tmp_path, stub, "die_on_command")
    assert sup.request({"action": "synthesize", "text": "a"})["echo"] == "a"
    # Kill the worker "out of band"; the next request transparently respawns.
    with sup._lock:
        sup._proc.kill()
    time.sleep(0.2)
    assert sup.request({"action": "synthesize", "text": "b"})["echo"] == "b"
    sup.shutdown()


def test_cuda_gate_marks_degraded(tmp_path, stub):
    sup = make(tmp_path, stub, "gate")
    with pytest.raises(WorkerDegradedError, match="CUDA gate"):
        sup.request({"action": "synthesize", "text": "hello"})
    # A degraded worker refuses every later request instead of respinning.
    with pytest.raises(WorkerDegradedError, match="CUDA gate"):
        sup.request({"action": "synthesize", "text": "hello"})


def test_idle_timeout_releases_process(tmp_path, stub):
    sup = make(tmp_path, stub, "echo", idle_timeout_s=0.5)
    sup.request({"action": "synthesize", "text": "hello"})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and sup.stats()["alive"]:
        time.sleep(0.1)
    assert not sup.stats()["alive"]
    # A request after the idle release lazily respawns and works.
    assert sup.request({"action": "synthesize", "text": "back"})["echo"] == "back"
    sup.shutdown()


def test_request_threshold_recycles_worker(tmp_path, stub):
    sup = make(tmp_path, stub, "echo", max_requests=2)
    sup.request({"action": "synthesize", "text": "a"})
    sup.request({"action": "synthesize", "text": "b"})
    assert not sup.stats()["alive"]  # recycled after the threshold
    assert sup.request({"action": "synthesize", "text": "c"})["echo"] == "c"
    assert sup.stats()["requests_served"] == 1  # fresh worker counts from zero
    sup.shutdown()


def test_unload_terminates_worker(tmp_path, stub):
    sup = make(tmp_path, stub, "echo")
    sup.request({"action": "synthesize", "text": "hello"})
    sup.unload()
    assert not sup.stats()["alive"]


def test_ping_style_logging(tmp_path, stub):
    lines: list[str] = []
    sup = make(tmp_path, stub, "echo", log=lines.append)
    sup.request({"action": "synthesize", "text": "hello"})
    assert "ready" in lines
    sup.shutdown()
