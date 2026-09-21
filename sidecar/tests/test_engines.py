"""Contract tests for the engine install/status endpoints, plus the Qwen3-TTS
engine's install wiring against a stubbed runtime (no network, no uv).

Endpoint tests run in-process against create_app with a stub registry; the
subprocess-based contract suite in test_contract.py keeps covering the real
boot path."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from voiceclone_sidecar import main as sidecar_main
from voiceclone_sidecar import sources
from voiceclone_sidecar.capabilities import Capabilities
from voiceclone_sidecar.engines import qwen3_tts
from voiceclone_sidecar.registry import (
    Engine,
    GenerationResult,
    InstallableEngine,
    Registry,
)


class StubEngine(Engine):
    engine_id = "stub"
    display_name = "Stub"

    def __init__(self):
        self._installed = False

    def capabilities(self):
        return Capabilities(languages=("zh",))

    def synthesize(self, request, log):
        return GenerationResult(audio_path="x.wav", sample_rate=16000)


class StubInstallable(StubEngine, InstallableEngine):
    engine_id = "stub-installable"
    display_name = "Stub Installable"

    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.installed_once = False

    def is_installed(self):
        return self._installed

    def install_state(self):
        return {
            "installed": self._installed,
            "steps": {"weights": {"status": "completed" if self._installed else "pending"}},
        }

    def install(self, log, progress=None):
        if self.fail:
            raise RuntimeError("boom: download failed")
        self._installed = True
        self.installed_once = True
        return {"installed": True}


@pytest.fixture()
def client(tmp_path):
    registry = Registry()
    stub = StubEngine()
    installable = StubInstallable()
    registry.register(stub)
    registry.register(installable)
    app = sidecar_main.create_app(registry, token="t", audio_dir=tmp_path)
    with TestClient(app, headers={"Authorization": "Bearer t"}) as c:
        yield c, installable


def test_status_of_plain_engine_is_installed(client):
    c, _ = client
    r = c.get("/engines/stub/status")
    assert r.status_code == 200
    body = r.json()
    assert body["installed"] is True
    assert body["steps"] == {}


def test_status_unknown_engine_404(client):
    c, _ = client
    assert c.get("/engines/nope/status").status_code == 404


def test_install_runs_in_background_and_updates_status(client):
    c, installable = client
    r = c.post("/engines/stub-installable/install")
    assert r.status_code == 200
    assert r.json()["status"] == "running"
    for _ in range(100):
        if installable.installed_once:
            break
        import time

        time.sleep(0.05)
    assert installable.installed_once
    assert c.get("/engines/stub-installable/status").json()["installed"] is True


def test_install_of_plain_engine_reports_installed(client):
    c, _ = client
    r = c.post("/engines/stub/install")
    assert r.status_code == 200
    assert r.json()["status"] == "installed"


def test_double_install_is_409(tmp_path):
    from voiceclone_sidecar import main as sidecar_main
    from voiceclone_sidecar.registry import Registry

    class Slow(StubInstallable):
        engine_id = "slow"

        def install(self, log, progress=None):
            import time

            time.sleep(0.4)
            self._installed = True

    registry = Registry()
    registry.register(Slow())
    app = sidecar_main.create_app(registry, token="t", audio_dir=tmp_path)
    with TestClient(app, headers={"Authorization": "Bearer t"}) as c:
        r1 = c.post("/engines/slow/install")
        r2 = c.post("/engines/slow/install")
        assert r1.status_code == 200
        assert r2.status_code == 409
        for _ in range(100):
            if c.get("/engines/slow/status").json()["installed"]:
                break
            import time

            time.sleep(0.05)
        assert c.get("/engines/slow/status").json()["installed"] is True


# --- qwen3 engine wiring (stubbed runtime) ---------------------------------


def test_qwen3_engine_declares_capabilities():
    engine = qwen3_tts.Qwen3TtsMlxEngine(root=Path("/tmp/ascii-root"))
    caps = engine.capabilities()
    assert caps.voice_cloning is True
    assert "zh" in caps.languages


def test_qwen3_engine_install_steps_cover_expected_steps(tmp_path):
    engine = qwen3_tts.Qwen3TtsMlxEngine(root=tmp_path)
    step_ids = [s.id for s in engine.install_steps()]
    assert step_ids == ["python", "venv", "packages", "weights"]
    weights_step = engine.install_steps()[3]
    assert (
        weights_step.artifact
        == tmp_path / "engines" / qwen3_tts.Qwen3TtsMlxEngine.engine_id / "weights"
    )


def test_qwen3_engine_declares_modelscope_twin():
    """ADR-0016: qwen3-tts-mlx has a verified ModelScope twin and resolves
    its weight chain at install time from the injected preferred source."""
    assert qwen3_tts.MS_REPO == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    chain = sources.weight_sources(qwen3_tts.REPO, ms_repo=qwen3_tts.MS_REPO)
    assert chain[0].startswith("https://huggingface.co/")
    assert chain[1].startswith("https://hf-mirror.com/")
    assert chain[2].startswith("https://modelscope.cn/models/mlx-community/")


# --- install progress on the status endpoint (issue #35) ----------------------


class ProgressStub(StubInstallable):
    """Installable stub whose install reports byte progress mid-run (issue #35)."""

    engine_id = "progress-stub"

    def __init__(self, fail=False):
        super().__init__(fail=fail)
        self.progress = None
        self.reported = threading.Event()
        self.release = threading.Event()
        self.steps = {"weights": {"status": "pending"}}

    def install_state(self):
        state = super().install_state()
        state["steps"] = self.steps
        state["progress"] = self.progress
        return state

    def install(self, log, progress=None):
        progress("weights:model.safetensors", 100, 400)
        self.progress = {
            "file": "weights:model.safetensors",
            "done_bytes": 100,
            "total_bytes": 400,
        }
        self.reported.set()
        self.release.wait(timeout=5)
        if self.fail:
            self.steps = {"weights": {"status": "failed", "error": "boom: download failed"}}
            raise RuntimeError("boom: download failed")
        self.steps = {"weights": {"status": "completed"}}
        self._installed = True
        return {"installed": True}


@pytest.fixture()
def progress_client(tmp_path):
    registry = Registry()
    stub = ProgressStub()
    registry.register(stub)
    app = sidecar_main.create_app(registry, token="t", audio_dir=tmp_path)
    with TestClient(app, headers={"Authorization": "Bearer t"}) as c:
        yield c, stub


def _wait(event) -> None:
    for _ in range(200):
        if event.is_set():
            return
        time.sleep(0.05)
    raise AssertionError("install did not reach the expected phase")


def test_status_exposes_progress_while_downloading(progress_client):
    c, stub = progress_client
    assert c.post("/engines/progress-stub/install").status_code == 200
    _wait(stub.reported)
    body = c.get("/engines/progress-stub/status").json()
    assert body["installing"] is True
    assert body["progress"] == {
        "file": "weights:model.safetensors",
        "done_bytes": 100,
        "total_bytes": 400,
    }
    stub.release.set()
    for _ in range(200):
        if c.get("/engines/progress-stub/status").json()["installed"]:
            break
        time.sleep(0.05)
    done = c.get("/engines/progress-stub/status").json()
    assert done["installed"] is True
    assert done["progress"] is None  # no fake progress after completion


def test_status_clears_progress_after_failed_install(tmp_path):
    registry = Registry()
    stub = ProgressStub(fail=True)
    registry.register(stub)
    app = sidecar_main.create_app(registry, token="t", audio_dir=tmp_path)
    with TestClient(app, headers={"Authorization": "Bearer t"}) as c:
        c.post("/engines/progress-stub/install")
        _wait(stub.reported)
        stub.release.set()
        for _ in range(200):
            body = c.get("/engines/progress-stub/status").json()
            if body["steps"]["weights"]["status"] == "failed":
                break
            time.sleep(0.05)
        assert body["installing"] is False
        assert body["progress"] is None
        assert "download failed" in body["steps"]["weights"]["error"]
