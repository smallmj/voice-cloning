"""IndexTTS-2.5 engine tests: install wiring, CUDA gate, per-engine queueing.

Everything is stubbed at the runtime seam (uv / downloader) — no network, no
GPU required. The CUDA-gate test runs the REAL worker script; on a machine
without torch it must refuse to boot with exit code 3 and an actionable
message.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from voiceclone_sidecar import main as sidecar_main
from voiceclone_sidecar.engines import indextts25
from voiceclone_sidecar.engines.indextts25 import IndexTts25CudaEngine
from voiceclone_sidecar.registry import (
    Engine,
    GenerationRequest,
    GenerationResult,
    Registry,
)
from voiceclone_sidecar.runtime import downloader, installer, uvman


@pytest.fixture()
def engine(tmp_path: Path) -> IndexTts25CudaEngine:
    return IndexTts25CudaEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})


def test_install_step_order(engine):
    steps = [s.id for s in engine.install_steps()]
    assert steps == ["python", "venv", "torch", "engine", "weights", "aux"]


def test_torch_step_uses_explicit_wheel_sources(engine, tmp_path, monkeypatch):
    recorded = {}

    def fake_download(spec, dest_dir, sources, progress=None, log=None):
        recorded.setdefault("wheels", []).append(spec.path)
        recorded.setdefault("torch_sources", list(sources))
        return dest_dir / spec.path

    installed = {}

    def fake_pip_install(uv, venv, packages, env=None, log=None):
        installed["packages"] = packages

    monkeypatch.setattr(downloader, "download_file", fake_download)
    monkeypatch.setattr(uvman, "pip_install", fake_pip_install)
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")

    engine.install_steps()[2].run(lambda m: None, lambda *a: None)

    # torch and torchaudio, fetched by exact wheel URL — never via an index.
    assert [w.split("%2B")[0] for w in recorded["wheels"]] == ["torch-2.8.0", "torchaudio-2.8.0"]
    assert "%2Bcu128-cp311-cp311-win_amd64.whl" in recorded["wheels"][0]
    sources = " | ".join(recorded["torch_sources"])
    assert "download.pytorch.org" in sources
    assert "mirrors.aliyun.com/pytorch-wheels" in sources  # mirror fallback
    # Install happens from the local files, with no index consulted.
    assert all(str(p).endswith(".whl") for p in installed["packages"])


def test_engine_step_falls_back_to_github(engine, tmp_path, monkeypatch):
    installs = []

    def fake_pip_install(uv, venv, packages, env=None, log=None):
        installs.append(packages)
        if len(installs) == 1:
            raise uvman.RuntimeCommandError("mirror dead")

    monkeypatch.setattr(uvman, "pip_install", fake_pip_install)
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")

    engine.install_steps()[3].run(lambda m: None, lambda *a: None)
    assert len(installs) == 2
    assert indextts25.ENGINE_PACKAGE_URL_FALLBACK in installs[1][0]


def test_weights_sources_include_a_fallback_mirror(engine, tmp_path, monkeypatch):
    recorded = []

    def fake_download(spec, dest_dir, sources, progress=None, log=None):
        recorded.append((spec.path, list(sources)))
        return dest_dir / spec.dest_name

    monkeypatch.setattr(downloader, "download_file", fake_download)

    engine.install_steps()[4].run(lambda m: None, lambda *a: None)  # main weights
    engine.install_steps()[5].run(lambda m: None, lambda *a: None)  # aux weights

    main_sources = dict(recorded)["config.yaml"]
    joined = " | ".join(main_sources)
    assert "huggingface.co" in joined
    assert "hf-mirror.com" in joined
    assert "modelscope.cn" in joined
    # w2v-bert-2.0 aux file lands in the hf_cache layout the engine expects.
    aux = dict(recorded)["model.safetensors"]
    assert "w2v-bert-2.0" in aux[0]


def test_cuda_gate_refuses_without_torch(tmp_path):
    """Run the REAL worker on this interpreter (no torch here) — it must exit
    with the gate code and an actionable message instead of hanging."""
    worker = Path(indextts25.__file__).with_name("indextts25_worker.py")
    proc = subprocess.run(
        [sys.executable, "-u", str(worker)],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == indextts25_worker_gate_code()
    out = proc.stdout
    assert '"fatal": true' in out
    assert "torch is not importable" in out


def indextts25_worker_gate_code() -> int:
    from voiceclone_sidecar.engines.indextts25_worker import CUDA_GATE_EXIT_CODE

    return CUDA_GATE_EXIT_CODE


def test_synthesize_requires_install(engine):
    with pytest.raises(RuntimeError, match="not installed"):
        engine.synthesize(
            GenerationRequest(generation_id="g1", text="你好", params={}), lambda m: None
        )


class _SlowEngine(Engine):
    """Records overlap to prove the sidecar serializes per engine."""

    engine_id = "slow"
    display_name = "Slow"

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.release = threading.Event()
        self.first_started = threading.Event()

    def capabilities(self):
        from voiceclone_sidecar.capabilities import Capabilities

        return Capabilities(languages=("zh",))

    def synthesize(self, request, log):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.first_started.set()
        self.release.wait(timeout=5)
        self.active -= 1
        return GenerationResult(audio_path="x.wav", sample_rate=16000)


def test_generations_serialize_per_engine(tmp_path):
    slow = _SlowEngine()
    registry = Registry()
    registry.register(slow)
    app = sidecar_main.create_app(registry, token="t", audio_dir=tmp_path)

    # Real uvicorn server: the two requests arrive from separate threads and
    # must queue on the per-engine lock, which no in-process transport shows.
    import socket
    import time as _time

    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    t_server = threading.Thread(target=server.run, daemon=True)
    t_server.start()
    deadline = _time.monotonic() + 10
    while not server.started and _time.monotonic() < deadline:
        _time.sleep(0.05)
    assert server.started

    try:
        import httpx

        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            headers={"Authorization": "Bearer t"},
            timeout=10,
        ) as client:
            results: list[int] = []

            def post(engine_id):
                results.append(
                    client.post("/generations", json={"text": "hi", "engine_id": engine_id}).status_code
                )

            t1 = threading.Thread(target=post, args=("slow",))
            t1.start()
            assert slow.first_started.wait(timeout=5)
            t2 = threading.Thread(target=post, args=("slow",))
            t2.start()
            time.sleep(0.5)  # give request 2 time to arrive and queue up
            assert slow.max_active == 1  # queued, not racing the worker
            slow.release.set()
            t1.join(timeout=5)
            t2.join(timeout=5)
            assert results == [200, 200]
    finally:
        server.should_exit = True
        t_server.join(timeout=5)
