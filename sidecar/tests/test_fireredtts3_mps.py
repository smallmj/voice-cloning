"""FireRedTTS3-MPS engine tests (issue #25 / ADR-0019): install wiring,
patch-source-of-truth, MPS gate, honest reference-text requirement.

Everything is stubbed at the runtime seam — no network, no model load. The
gate test runs the REAL worker script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from voiceclone_sidecar.engines import fireredtts3_mps, fireredtts3_patch
from voiceclone_sidecar.engines.fireredtts3_mps import FireRedTts3MpsEngine
from voiceclone_sidecar.engines.fireredtts3_patch import PatchAnchorError, apply_patches
from voiceclone_sidecar.registry import GenerationRequest
from voiceclone_sidecar.runtime import uvman


@pytest.fixture()
def engine(tmp_path: Path) -> FireRedTts3MpsEngine:
    return FireRedTts3MpsEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})


def test_install_step_order(engine):
    steps = [s.id for s in engine.install_steps()]
    assert steps == ["python", "venv", "torch", "engine", "weights"]


def test_torch_and_engine_pins(engine, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        uvman,
        "pip_install",
        lambda uv, venv, packages, env=None, log=None: calls.append(list(packages)),
    )
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    # The engine step also downloads + extracts + patches the upstream
    # checkout; stub the download with a real (tiny) zip so no network
    # happens and the patch step runs against a valid tree.
    import zipfile

    zip_path = tmp_path / "runtime" / "engines" / "fireredtts3-mps" / "fireredtts3-main.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        content: dict[str, str] = {}
        for rel, old, _, expected in fireredtts3_patch.PATCHES:
            # Anchor content so every patch of that file applies cleanly
            # (anchors expected N times appear N times).
            content[rel] = content.get(rel, "") + (old + "\n") * expected
        for rel, text in content.items():
            zf.writestr("FireRedTTS3-main/" + rel, text)
    from voiceclone_sidecar.runtime import downloader

    def fake_download(spec, dest_dir, sources, progress=None, log=None):
        dest = dest_dir / spec.dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zip_path.read_bytes())
        return dest

    monkeypatch.setattr(downloader, "download_file", fake_download)
    engine.install_steps()[2].run(lambda m: None, lambda *a: None)  # torch
    engine.install_steps()[3].run(lambda m: None, lambda *a: None)  # engine deps + patch
    assert calls[0] == ["torch==2.8.0", "torchaudio==2.8.0"]
    # PoC-verified dependency set: no flash_attn, no fasttext, no torchcodec.
    assert "transformers==5.6.2" in calls[1]
    joined = "\n".join(calls[1])
    for banned in ("flash", "fasttext", "torchcodec"):
        assert banned not in joined


def test_weights_list_is_base_only(engine, tmp_path, monkeypatch):
    recorded = []
    from voiceclone_sidecar.runtime import downloader

    def fake_download(spec, dest_dir, sources, progress=None, log=None):
        recorded.append(spec.path)
        dest = dest_dir / spec.dest_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(downloader, "download_file", fake_download)
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    engine.install_steps()[4].run(lambda m: None, lambda *a: None)
    assert "fireredtts3_base/model.safetensors" in recorded
    assert not any("instruct" in p.lower() for p in recorded), "Instruct weights are out of scope"


def test_patch_table_matches_poc_script():
    """ADR-0019 decision 2: the PoC CLI and the engine install must share
    one patch surface — the script delegates to the package module."""

    repo = Path(fireredtts3_mps.__file__).resolve().parents[3]
    script = repo / "scripts" / "fireredtts3_poc" / "patch_fireredtts3.py"
    assert script.is_file(), f"PoC patch CLI missing: {script}"
    source = script.read_text(encoding="utf-8")
    # The script must DELEGATE, not duplicate: it imports the package module
    # and must not embed its own patch table.
    assert "fireredtts3_patch" in source
    assert "PATCHES: list" not in source


def test_patch_fails_loudly_on_drift(tmp_path):
    upstream = tmp_path / "upstream"
    (upstream / "fireredtts3" / "llm").mkdir(parents=True)
    (upstream / "fireredtts3" / "llm" / "fireredtts3_base.py").write_text(
        "self.device = torch.device('somewhere_else')\n"
    )
    with pytest.raises(PatchAnchorError):
        apply_patches(upstream)


def test_synthesize_requires_install(engine):
    with pytest.raises(RuntimeError, match="not installed"):
        engine.synthesize(
            GenerationRequest(generation_id="g1", text="你好", params={}), lambda m: None
        )


def test_synthesize_requires_reference_audio(engine):
    engine.is_installed = lambda: True
    with pytest.raises(RuntimeError, match="参考音频"):
        engine.synthesize(
            GenerationRequest(generation_id="g2", text="你好", params={}), lambda m: None
        )


def test_synthesize_requires_reference_text(engine):
    """requires_reference_text=True must ALSO be enforced at the engine
    boundary with an actionable message (the PoC showed prompt_audio without
    prompt_text fails deep inside the model)."""
    engine.is_installed = lambda: True
    with pytest.raises(RuntimeError, match="参考文本"):
        engine.synthesize(
            GenerationRequest(
                generation_id="g3", text="你好", params={"ref_audio": "/tmp/ref.wav"}
            ),
            lambda m: None,
        )


def test_registry_registers_on_apple_silicon(tmp_path):
    from voiceclone_sidecar.registry import default_registry

    registry = default_registry(output_dir=tmp_path / "audio")
    assert registry.get("fireredtts3-mps") is not None
    assert "实验性" in registry.get("fireredtts3-mps").display_name


def test_instruct_param_declared_not_exposed(engine):
    spec = next(s for s in engine.param_specs() if s.name == "instruct_mode")
    assert spec.exposed is False
    assert spec.not_exposed_reason == "wrong-mode"


def test_worker_gate_refuses_without_torch(tmp_path):
    worker = Path(fireredtts3_mps.__file__).with_name("fireredtts3_mps_worker.py")
    proc = subprocess.run(
        [sys.executable, "-u", str(worker)],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 3
    assert '"fatal": true' in proc.stdout
    assert "torch is not importable" in proc.stdout


# -- issue #51: quality params (inference_cfg / n_timesteps / cross_fade_ms) --


def _fake_supervisor(monkeypatch, capture, tmp_path):
    """Stub WorkerSupervisor so synthesize() records the wire payload."""
    from voiceclone_sidecar.engines import worker_supervisor as ws

    # synthesize() refuses a broken venv before reaching the supervisor —
    # fake an engine venv python inside the tmp runtime root.
    venv_python = tmp_path / "runtime" / "engines" / "fireredtts3-mps" / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("")

    class FakeSupervisor:
        def __init__(self, command, cwd, idle_timeout_s=0.0, max_requests=0):
            pass

        def request(self, payload, on_log=None):
            capture.append(payload)
            return {"audio_path": payload["output"], "sample_rate": 24000}

        def unload(self):
            pass

    monkeypatch.setattr(ws, "WorkerSupervisor", FakeSupervisor)


def test_synthesize_sends_quality_params(engine, tmp_path, monkeypatch):
    capture = []
    _fake_supervisor(monkeypatch, capture, tmp_path)
    engine.is_installed = lambda: True
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x")
    engine.synthesize(
        GenerationRequest(
            generation_id="g4",
            text="你好",
            params={
                "ref_audio": str(ref),
                "ref_text": "参考文本",
                "seed": 7,
                "inference_cfg": 3.0,
                "n_timesteps": 20,
                "cross_fade_ms": 100,
            },
        ),
        lambda m: None,
    )
    payload = capture[0]
    assert payload["seed"] == 7
    assert payload["inference_cfg"] == 3.0
    assert payload["n_timesteps"] == 20
    assert payload["cross_fade_ms"] == 100


def test_synthesize_omits_unset_and_default_quality_params(engine, tmp_path, monkeypatch):
    """#38 口径：默认值不发送——未设置的三个质量参数一律不出现在 wire payload
    （上游默认 inference_cfg=2.0 / n_timesteps=10 / cross_fade_ms=50 兜底）。"""
    capture = []
    _fake_supervisor(monkeypatch, capture, tmp_path)
    engine.is_installed = lambda: True
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"x")
    engine.synthesize(
        GenerationRequest(
            generation_id="g5",
            text="你好",
            params={"ref_audio": str(ref), "ref_text": "参考文本"},
        ),
        lambda m: None,
    )
    payload = capture[0]
    assert "seed" not in payload
    assert "inference_cfg" not in payload
    assert "n_timesteps" not in payload
    assert "cross_fade_ms" not in payload


def test_quality_param_specs_exposed_with_upstream_defaults(engine):
    specs = {s.name: s for s in engine.param_specs()}
    assert specs["inference_cfg"].default == 2.0
    assert specs["inference_cfg"].exposed is True
    assert specs["inference_cfg"].layer == "engine"
    assert specs["n_timesteps"].default == 10
    assert specs["n_timesteps"].integer is True
    assert specs["cross_fade_ms"].default == 50
    assert specs["cross_fade_ms"].unit == "ms"
    for name in ("inference_cfg", "n_timesteps", "cross_fade_ms"):
        assert specs[name].applies_to is not None
