"""IndexTTS-2.5 MPS engine tests: install wiring, local-weight reuse, MPS gate.

Mirrors test_indextts25.py. Everything is stubbed at the runtime seam
(uv / downloader) — no network, no model load. The gate test runs the REAL
worker script; on an interpreter without torch it must refuse to boot with
exit code 3 and an actionable message.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from voiceclone_sidecar.engines import indextts25, indextts25_mps
from voiceclone_sidecar.engines.indextts25_mps import IndexTts25MpsEngine
from voiceclone_sidecar.registry import GenerationRequest
from voiceclone_sidecar.runtime import downloader, uvman


@pytest.fixture()
def engine(tmp_path: Path) -> IndexTts25MpsEngine:
    return IndexTts25MpsEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})


def test_install_step_order(engine):
    steps = [s.id for s in engine.install_steps()]
    assert steps == ["python", "venv", "torch", "engine", "weights", "aux"]


def test_torch_step_pins_pypi_versions(engine, tmp_path, monkeypatch):
    installed = {}

    def fake_pip_install(uv, venv, packages, env=None, log=None):
        installed["packages"] = packages

    monkeypatch.setattr(uvman, "pip_install", fake_pip_install)
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")

    # No explicit wheel downloads may happen: macOS wheels come from the index.
    monkeypatch.setattr(
        downloader, "download_file",
        lambda *a, **k: pytest.fail("torch step must not fetch explicit wheels on macOS"),
    )

    engine.install_steps()[2].run(lambda m: None, lambda *a: None)
    assert installed["packages"] == ["torch==2.8.0", "torchaudio==2.8.0"]


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


def test_weights_reuse_local_source_before_network(tmp_path, monkeypatch):
    """An existing checkpoint tree satisfies the install without any download."""
    local = tmp_path / "checkpoints"
    (local / "qwen0.6bemo4-merge").mkdir(parents=True)
    (local / "hf_cache" / "w2v-bert-2.0").mkdir(parents=True)
    (local / "config.yaml").write_text("cfg")
    (local / "gpt.pth").write_bytes(b"gpt")
    (local / "qwen0.6bemo4-merge" / "config.json").write_text("{}")
    (local / "hf_cache" / "w2v-bert-2.0" / "config.json").write_text("{}")

    engine = IndexTts25MpsEngine(
        output_dir=tmp_path / "audio",
        root=tmp_path / "runtime",
        env={indextts25_mps.LOCAL_WEIGHTS_ENV: str(local)},
    )

    downloaded: list[str] = []

    def fake_download(spec, dest_dir, sources, progress=None, log=None):
        downloaded.append(spec.path)
        dest = dest_dir / spec.dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(downloader, "download_file", fake_download)

    weights_dir = tmp_path / "runtime" / "engines" / "indextts-25-mps" / "weights"
    logs: list[str] = []
    engine.install_steps()[4].run(logs.append, lambda *a: None)  # main weights
    engine.install_steps()[5].run(logs.append, lambda *a: None)  # aux weights

    assert (weights_dir / "config.yaml").read_text() == "cfg"
    assert (weights_dir / "gpt.pth").read_bytes() == b"gpt"
    assert (weights_dir / "qwen0.6bemo4-merge" / "config.json").read_text() == "{}"
    assert (weights_dir / "hf_cache" / "w2v-bert-2.0" / "config.json").read_text() == "{}"
    assert any("reusing local config.yaml" in m for m in logs)
    # Present files never hit the network; missing ones do.
    assert "config.yaml" not in downloaded and "gpt.pth" not in downloaded
    assert "s2mel.pth" in downloaded


def test_weights_fall_back_to_download_for_missing_local_files(tmp_path, monkeypatch):
    """Files absent from the local tree go through the downloader with mirrors."""
    local = tmp_path / "checkpoints"
    local.mkdir()

    engine = IndexTts25MpsEngine(
        output_dir=tmp_path / "audio",
        root=tmp_path / "runtime",
        env={indextts25_mps.LOCAL_WEIGHTS_ENV: str(local)},
    )

    recorded = []

    def fake_download(spec, dest_dir, sources, progress=None, log=None):
        recorded.append((spec.path, list(sources)))
        dest = dest_dir / spec.dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(downloader, "download_file", fake_download)

    engine.install_steps()[4].run(lambda m: None, lambda *a: None)
    main_sources = dict(recorded)["config.yaml"]
    joined = " | ".join(main_sources)
    assert "huggingface.co" in joined
    assert "hf-mirror.com" in joined
    assert "modelscope.cn" in joined


def test_mps_gate_refuses_without_torch(tmp_path):
    """Run the REAL worker on this interpreter (no torch here) — it must exit
    with the gate code and an actionable message instead of hanging."""
    worker = Path(indextts25_mps.__file__).with_name("indextts25_mps_worker.py")
    proc = subprocess.run(
        [sys.executable, "-u", str(worker)],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == indextts25_mps.MPS_GATE_EXIT_CODE
    out = proc.stdout
    assert '"fatal": true' in out
    assert "torch is not importable" in out


def test_synthesize_requires_install(engine):
    with pytest.raises(RuntimeError, match="not installed"):
        engine.synthesize(
            GenerationRequest(generation_id="g1", text="你好", params={}), lambda m: None
        )


def test_registry_registers_mps_engine_on_apple_silicon(tmp_path):
    """This test machine IS darwin/arm64 — the default registry must carry it."""
    from voiceclone_sidecar.registry import default_registry

    registry = default_registry(output_dir=tmp_path / "audio")
    assert registry.get("indextts-25-mps") is not None
    assert registry.get("indextts-25-mps").display_name == "IndexTTS-2.5（本地 · MPS）"


def test_local_weights_env_is_expanded(tmp_path):
    local = tmp_path / "cp"
    local.mkdir()
    engine = IndexTts25MpsEngine(
        output_dir=tmp_path, root=tmp_path, env={indextts25_mps.LOCAL_WEIGHTS_ENV: str(local)}
    )
    assert engine._local_weights_root() == local


def test_synthesize_refuses_without_reference_audio(engine):
    engine.is_installed = lambda: True  # guard sits after the install check
    with pytest.raises(RuntimeError, match="参考音频"):
        engine.synthesize(
            GenerationRequest(generation_id="g2", text="你好", params={}), lambda m: None
        )


def test_worker_constants_do_not_poison_sidecar_env(monkeypatch):
    """Regression (install 'no reaction'): importing the gate constant out of
    the worker script dragged indextts25_common_worker — which force-sets
    os.environ['HF_HUB_OFFLINE']='1' for the worker process — into the
    sidecar process. Every install-time download child then inherited offline
    mode and failed instantly. Importing the engine module must NOT flip the
    sidecar process into HF offline mode."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    import importlib

    import voiceclone_sidecar.engines.indextts25_mps as mps

    importlib.reload(mps)
    assert os.environ.get("HF_HUB_OFFLINE") != "1"
    assert mps.MPS_GATE_EXIT_CODE == 3
