"""Contract tests for the engine install/status endpoints, plus the Qwen3-TTS
engine's install wiring against a stubbed runtime (no network, no uv).

Endpoint tests run in-process against create_app with a stub registry; the
subprocess-based contract suite in test_contract.py keeps covering the real
boot path."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from voiceclone_sidecar import main as sidecar_main
from voiceclone_sidecar.capabilities import Capabilities
from voiceclone_sidecar.engines import qwen3_tts
from voiceclone_sidecar.registry import Engine, GenerationRequest, GenerationResult, InstallableEngine, Registry
from voiceclone_sidecar.runtime import installer


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
    assert weights_step.artifact == tmp_path / "engines" / qwen3_tts.Qwen3TtsMlxEngine.engine_id / "weights"


def test_qwen3_engine_uses_mirror_sources():
    assert qwen3_tts.SOURCES[0].startswith("https://huggingface.co/")
    assert qwen3_tts.SOURCES[1].startswith("https://hf-mirror.com/")
