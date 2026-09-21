"""dots.tts CUDA engine tests (issue #25): install wiring, the
WeTextProcessing-optional patch, honest parameter declarations.

Everything is stubbed at the runtime seam — no network, no model load. The
patch tests run against the REAL upstream text.py content (fetched from
tag v0.3.1); the gate test runs the REAL worker script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from voiceclone_sidecar.engines import dots_tts_cuda
from voiceclone_sidecar.engines.dots_tts_cuda import DotsTtsCudaEngine
from voiceclone_sidecar.engines.dots_tts_patch import PatchAnchorError, apply_patches
from voiceclone_sidecar.pronunciation import rewrite
from voiceclone_sidecar.registry import GenerationRequest
from voiceclone_sidecar.runtime import uvman

# The exact top-of-file lines of dots_tts/utils/text.py at tag v0.3.1
# (verified from upstream source; see CORRECTIONS.md C-003).
UPSTREAM_TEXT_PY_HEAD = '''from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from langcodes import Language as LangcodesLanguage
from lingua import Language, LanguageDetectorBuilder
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer


@lru_cache(maxsize=1)
def get_chinese_text_normalizer() -> ZhNormalizer:
    return ZhNormalizer()


@lru_cache(maxsize=1)
def get_english_text_normalizer() -> EnNormalizer:
    return EnNormalizer()
'''


@pytest.fixture()
def engine(tmp_path: Path) -> DotsTtsCudaEngine:
    return DotsTtsCudaEngine(output_dir=tmp_path / "audio", root=tmp_path / "runtime", env={})


def test_install_step_order(engine):
    steps = [s.id for s in engine.install_steps()]
    assert steps == ["python", "venv", "torch", "engine", "weights"]


def test_engine_step_installs_no_deps_and_skips_pynini(engine, tmp_path, monkeypatch):
    """The whole point of the patch (CORRECTIONS.md C-003): WeTextProcessing
    (pynini) must never enter the dependency resolution."""
    calls = []

    def fake_pip_install(uv, venv, packages, env=None, log=None):
        calls.append(list(packages))

    monkeypatch.setattr(uvman, "pip_install", fake_pip_install)
    monkeypatch.setattr(uvman, "find_uv", lambda *a, **k: Path("uv"))
    monkeypatch.setattr(uvman, "create_venv", lambda *a, **k: tmp_path / "venv")
    from voiceclone_sidecar.engines import dots_tts_cuda

    monkeypatch.setattr(
        dots_tts_cuda, "apply_to_venv",
        lambda venv, log: ["patched (stub)"],
    )
    engine.install_steps()[3].run(lambda m: None, lambda *a: None)
    assert calls[0] == ["dots.tts==0.3.1", "--no-deps"]
    joined = "\n".join("\n".join(c) for c in calls)
    assert "WeTextProcessing" not in joined and "pynini" not in joined
    # The verified headless runtime deps ARE installed, pinned where the
    # upstream constraints file pins them.
    assert "transformers==4.57.0" in calls[1]
    assert "lingua-language-detector" in calls[1]


def test_patch_makes_tn_imports_optional():
    patched = apply_patches(UPSTREAM_TEXT_PY_HEAD)
    assert "try:" in patched and "ZhNormalizer = None" in patched
    assert "get_chinese_text_normalizer" in patched
    # Idempotent
    assert apply_patches(patched) == patched


def test_patch_fails_loudly_on_drift():
    drifted = UPSTREAM_TEXT_PY_HEAD.replace(
        "from tn.chinese.normalizer import Normalizer as ZhNormalizer",
        "from tn.chinese.normalizer import Normalizer as ZhNorm  # upstream drifted",
    )
    with pytest.raises(PatchAnchorError):
        apply_patches(drifted)


def test_pronunciation_grammar_dots_uses_tone_marks_only():
    out = rewrite("我生平不好此道", "好=hao4", "dots")
    assert "hào" in out and "hao4" not in out


def test_synthesize_requires_install(engine):
    with pytest.raises(RuntimeError, match="not installed"):
        engine.synthesize(GenerationRequest(generation_id="g1", text="你好", params={}), lambda m: None)


def test_synthesize_requires_reference_audio(engine):
    engine.is_installed = lambda: True
    with pytest.raises(RuntimeError, match="参考音频"):
        engine.synthesize(GenerationRequest(generation_id="g2", text="你好", params={}), lambda m: None)


def test_fake_parameters_declared_not_exposed(engine):
    """No temperature (no token sampler) and no speed parameter exist in
    dots.tts (verified, README) — declared as data, never shown."""
    for name in ("temperature", "speed"):
        spec = next(s for s in engine.param_specs() if s.name == name)
        assert spec.exposed is False
        assert spec.not_exposed_reason == "no-op"


def test_normalize_text_declared_breaks_pipeline(engine):
    spec = next(s for s in engine.param_specs() if s.name == "normalize_text")
    assert spec.exposed is False
    assert spec.not_exposed_reason == "breaks-pipeline"


def test_worker_gate_refuses_without_torch(tmp_path):
    worker = Path(dots_tts_cuda.__file__).with_name("dots_tts_worker.py")
    proc = subprocess.run(
        [sys.executable, "-u", str(worker)],
        input="", capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 3
    assert '"fatal": true' in proc.stdout
    assert "torch is not importable" in proc.stdout


def test_windows_only_shape():
    # The engine class exists and is importable on any platform (tests run
    # everywhere), but the REGISTRY only registers it on win32 — asserted
    # here by checking the gate decision lives in the registry module.
    import inspect

    from voiceclone_sidecar import registry

    source = inspect.getsource(registry.default_registry)
    assert 'sys.platform == "win32"' in source
    assert "DotsTtsCudaEngine" in source
