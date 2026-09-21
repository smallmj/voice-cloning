"""VoxCPM2 engine tests (issue #25): install wiring, three cloning modes,
MPS/CUDA gates. Everything is stubbed at the runtime seam — no network, no
model load. Gate tests run the REAL worker script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from voiceclone_sidecar.engines import voxcpm2_base, voxcpm2_mps
from voiceclone_sidecar.engines.voxcpm2_mps import VoxCPM2MpsEngine
from voiceclone_sidecar.pronunciation import rewrite
from voiceclone_sidecar.registry import GenerationRequest
from voiceclone_sidecar.runtime import uvman


@pytest.fixture()
def engine(tmp_path: Path) -> VoxCPM2MpsEngine:
    return VoxCPM2MpsEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})


def test_install_step_order(engine):
    steps = [s.id for s in engine.install_steps()]
    assert steps == ["python", "venv", "torch", "engine", "weights"]


def test_torch_step_pins_pypi_versions_on_mps(engine, tmp_path, monkeypatch):
    installed = {}
    monkeypatch.setattr(
        uvman, "pip_install",
        lambda uv, venv, packages, env=None, log=None: installed.setdefault("packages", packages),
    )
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    engine.install_steps()[2].run(lambda m: None, lambda *a: None)
    assert installed["packages"] == ["torch==2.8.0", "torchaudio==2.8.0"]


def test_cuda_variant_fetches_exact_wheels(tmp_path, monkeypatch):
    from voiceclone_sidecar.engines.voxcpm2_cuda import VoxCPM2CudaEngine

    engine = VoxCPM2CudaEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})
    wheels = []
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(uvman, "pip_install", lambda *a, **k: wheels.append(a[2]))
    # Stub the wheel download (real wheels are GB-sized; the URL shape is
    # what this test asserts).
    from voiceclone_sidecar.runtime import downloader

    def fake_download(spec, dest_dir, sources, progress=None, log=None):
        dest = dest_dir / spec.dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"wheel")
        return dest

    monkeypatch.setattr(downloader, "download_file", fake_download)
    engine.install_steps()[2].run(lambda m: None, lambda *a: None)
    # Exact CUDA wheel URLs, installed from local files — no index resolution.
    # local files carry a literal "+" (the URL name uses %2B)
    assert any("torch-2.8.0+cu128-cp311-cp311-win_amd64.whl" in p for p in wheels[0])


def test_engine_step_pins_voxcpm_version(engine, tmp_path, monkeypatch):
    installed = {}
    monkeypatch.setattr(
        uvman, "pip_install",
        lambda uv, venv, packages, env=None, log=None: installed.setdefault("packages", packages),
    )
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    engine.install_steps()[3].run(lambda m: None, lambda *a: None)
    assert installed["packages"] == ["voxcpm==2.0.3"]


def test_pronunciation_grammar_voxcpm():
    assert rewrite("你好世界", "你=ni3", "voxcpm") == "{ni3}好世界"


def test_synthesize_requires_install(engine):
    with pytest.raises(RuntimeError, match="not installed"):
        engine.synthesize(GenerationRequest(generation_id="g1", text="你好", params={}), lambda m: None)


def test_synthesize_requires_reference_audio(engine):
    engine.is_installed = lambda: True
    with pytest.raises(RuntimeError, match="参考音频"):
        engine.synthesize(GenerationRequest(generation_id="g2", text="你好", params={}), lambda m: None)


def test_registry_registers_mps_engine_on_apple_silicon(tmp_path):
    from voiceclone_sidecar.registry import default_registry

    registry = default_registry(output_dir=tmp_path / "audio")
    assert registry.get("voxcpm2-mps") is not None
    assert registry.get("voxcpm2-mps").display_name == "VoxCPM2（本地 · MPS）"


def test_param_specs_exposed_params_carry_applies_to(engine):
    for spec in engine.param_specs():
        if spec.exposed:
            assert spec.applies_to is not None
            assert spec.applies_to.engine == "voxcpm2-mps"


def test_normalize_param_is_declared_not_exposed(engine):
    """Engine-side TN duplicates this app's ADR-0008 layer — declared data."""
    spec = next(s for s in engine.param_specs() if s.name == "normalize")
    assert spec.exposed is False
    assert spec.not_exposed_reason == "breaks-pipeline"


def test_denoise_param_not_exposed_without_denoiser(engine):
    spec = next(s for s in engine.param_specs() if s.name == "denoise")
    assert spec.exposed is False


def test_worker_gate_refuses_without_torch(tmp_path):
    """The REAL worker on this interpreter (no torch in the engine venv
    sense) must exit with the gate code and an actionable message."""
    worker = Path(voxcpm2_mps.__file__).with_name("voxcpm2_worker.py")
    proc = subprocess.run(
        [sys.executable, "-u", str(worker), "mps"],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 3
    assert '"fatal": true' in proc.stdout
    assert "torch is not importable" in proc.stdout


def test_worker_refuses_missing_gate_argument():
    worker = Path(voxcpm2_mps.__file__).with_name("voxcpm2_worker.py")
    proc = subprocess.run(
        [sys.executable, "-u", str(worker)],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 3
    assert "gate not specified" in proc.stdout


def test_cuda_and_mps_share_the_base():
    from voiceclone_sidecar.engines.voxcpm2_cuda import VoxCPM2CudaEngine

    assert issubclass(VoxCPM2MpsEngine, voxcpm2_base.VoxCPM2EngineBase)
    assert issubclass(VoxCPM2CudaEngine, voxcpm2_base.VoxCPM2EngineBase)
    # Identical model, identical capability surface.
    a = VoxCPM2MpsEngine(output_dir=Path("/tmp/a"), root=Path("/tmp/a"), env={})
    b = VoxCPM2CudaEngine(output_dir=Path("/tmp/a"), root=Path("/tmp/a"), env={})
    assert a.capabilities().to_dict() == b.capabilities().to_dict()
