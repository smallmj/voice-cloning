"""Spawn a REAL sidecar process per session and hand the contract to tests.

This is the spec's single seam: tests run against a real process bound to
127.0.0.1 on a dynamic port, with the fake engine registered through the
real default registry.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

SIDECAR_DIR = Path(__file__).resolve().parent.parent


def _wait_for_port(port: int, process: subprocess.Popen, timeout: float = 30.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError as exc:
            if process.poll() is not None:
                raise RuntimeError(f"sidecar exited early: {process.stderr.read()!r}") from exc
            time.sleep(0.1)
    raise TimeoutError("sidecar did not start listening in time")


@pytest.fixture(scope="session")
def sidecar(tmp_path_factory):
    # `=` form: a token_urlsafe value can start with "-", which argparse
    # would misread as an option when passed as a separate argv item
    # (intermittent "--token: expected one argument" CI failures).
    token = secrets.token_urlsafe(16)
    audio_dir = tmp_path_factory.mktemp("audio")
    # Issue #61: the app process must see a HERMETIC runtime root. Otherwise
    # a machine with the local ASR tool installed makes "uninstalled local
    # provider" contracts (409 guidance) unreachable — the gate transcribes
    # for real instead. Engine installs under test register their own roots.
    runtime_root = tmp_path_factory.mktemp("runtime")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "voiceclone_sidecar",
            "--port",
            "0",
            f"--token={token}",
            "--audio-dir",
            str(audio_dir),
        ],
        env={
            **os.environ,
            "VOICECLONE_TEST_ENGINES": "1",
            "VOICECLONE_KEY_BACKEND": "memory",
            "VOICECLONE_RUNTIME_ROOT": str(runtime_root),
        },
        cwd=SIDECAR_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = process.stdout.readline()
        if not line.strip():
            # Empty handshake = the sidecar died at startup. Surface stderr,
            # otherwise CI shows only a bare JSONDecodeError with no cause.
            stderr = process.stderr.read() if process.poll() is not None else "(still running)"
            raise RuntimeError(f"sidecar produced no handshake line; stderr:\n{stderr}")
        handshake = json.loads(line)
        assert handshake["event"] == "ready", line
        port = handshake["port"]
        _wait_for_port(port, process)
        yield {
            "base_url": f"http://127.0.0.1:{port}",
            "token": token,
            "process": process,
            "audio_dir": audio_dir,
            "data_dir": audio_dir.parent,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture()
def client(sidecar) -> httpx.Client:
    with httpx.Client(
        base_url=sidecar["base_url"],
        headers={"Authorization": f"Bearer {sidecar['token']}"},
        timeout=30.0,
    ) as c:
        yield c


@pytest.fixture()
def ws_url(sidecar) -> str:
    return f"ws://127.0.0.1:{sidecar['base_url'].rsplit(':', 1)[1]}/ws/logs"
