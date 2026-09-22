"""Qwen3-TTS-MLX local parameter panel (issues #47 / #48).

#47: the canonical language slot used to be a silent no-op — the worker sent
`language=` while upstream only knows `lang_code`, and the capitalized value
never matched the weight codec_language_id keys. These tests lock the fix:
lowercase 10-language wire mapping, auto = key absent.

#48: four engine-layer sampling parameters (temperature/top_p/top_k/
repetition_penalty). They are ALWAYS sent, including the declared defaults:
mlx-audio's hardcoded silent defaults (0.6/0.8/-1/1.3) differ from the
official generation_config.json (0.9/1.0/50/1.05) the UI shows, so
"not sending the default" would make the engine do something else than
what the user sees.
"""

from __future__ import annotations

import io
import json
import struct
import sys
import types
from pathlib import Path

import pytest

from voiceclone_sidecar.engines.qwen3_tts import Qwen3TtsMlxEngine
from voiceclone_sidecar.registry import GenerationRequest

ENGINE = Qwen3TtsMlxEngine(root=Path("/tmp/ascii-root"))

EXPECTED_LANGUAGES = (
    "chinese", "english", "french", "german", "italian",
    "japanese", "korean", "portuguese", "russian", "spanish",
)


def language_spec():
    return next(s for s in ENGINE.param_specs() if s.name == "language")


# --- issue #47: language wire mapping ---------------------------------------


def test_language_choices_are_the_ten_upstream_languages_plus_auto():
    # Issue #47: the wire mapping is correct (lang_code, lowercase), but V3
    # real-machine verification proved mlx-audio never consumes lang_code —
    # so the selector stays honest no-op DATA (audit decision 3, ADR-0018).
    spec = language_spec()
    assert spec.exposed is False
    assert spec.not_exposed_reason == "no-op"
    assert spec.choices == (
        "Auto", "Chinese", "English", "French", "German", "Italian",
        "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
    )
    assert spec.wire_path == "lang_code"
    assert spec.default == "Auto"


def test_language_wire_adapter_lowercases_known_choices():
    to_wire = language_spec().to_wire
    assert to_wire("Chinese") == "chinese"
    assert to_wire("Spanish") == "spanish"
    for choice in language_spec().choices:
        if choice == "Auto":
            assert to_wire(choice) is None
        else:
            assert to_wire(choice) == choice.lower()


def test_language_auto_and_unknown_map_to_none():
    to_wire = language_spec().to_wire
    assert to_wire(None) is None
    assert to_wire("") is None
    assert to_wire("Auto") is None
    assert to_wire("Klingon") is None  # unknown never guesses a fallback


def test_payload_carries_lowercase_lang_code_only_when_chosen(tmp_path):
    req = GenerationRequest(generation_id="g1", text="你好", params={"language": "Japanese"})
    payload = ENGINE.build_worker_payload(req, tmp_path, tmp_path / "o.wav")
    assert payload["lang_code"] == "japanese"
    assert "language" not in payload  # the old silent-no-op key is gone

    auto = GenerationRequest(generation_id="g2", text="你好", params={"language": "Auto"})
    payload = ENGINE.build_worker_payload(auto, tmp_path, tmp_path / "o.wav")
    assert "lang_code" not in payload  # auto = key absent, never empty string

    nochoice = GenerationRequest(generation_id="g3", text="你好", params={})
    payload = ENGINE.build_worker_payload(nochoice, tmp_path, tmp_path / "o.wav")
    assert "lang_code" not in payload


# --- issue #48: engine-layer sampling parameters -----------------------------


def test_sampling_specs_declared_with_construction_time_contract():
    specs = {s.name: s for s in ENGINE.param_specs()}
    expected = {
        "temperature": (0.9, 0.0, 2.0),
        "top_p": (1.0, 0.0, 1.0),
        "top_k": (50, 1, 200),
        "repetition_penalty": (1.05, 1.0, 2.0),
    }
    for name, (default, lo, hi) in expected.items():
        spec = specs[name]
        assert spec.exposed is True, name
        assert spec.layer == "engine", name
        assert spec.default == default, name
        assert spec.min == lo and spec.max == hi, name
        assert spec.applies_to is not None, name
        assert spec.to_wire is not None, name
    assert specs["top_k"].integer is True


def test_sampling_defaults_match_official_generation_config(tmp_path):
    # Issue #48: the declared defaults are the official weight
    # generation_config.json values. They are ALWAYS sent (see below),
    # because mlx-audio's hardcoded silent defaults (0.6/0.8/-1/1.3)
    # differ from the official config — "not sending" would silently mean
    # something else than what the UI shows.
    payload = ENGINE.build_worker_payload(
        GenerationRequest(generation_id="g1", text="hi", params={}),
        tmp_path, tmp_path / "o.wav",
    )
    assert payload["temperature"] == 0.9
    assert payload["top_p"] == 1.0
    assert payload["top_k"] == 50
    assert isinstance(payload["top_k"], int)
    assert payload["repetition_penalty"] == 1.05


def test_sampling_changed_values_are_sent_typed(tmp_path):
    req = GenerationRequest(generation_id="g1", text="hi", params={
        "temperature": 0.7, "top_p": 0.85, "top_k": 20, "repetition_penalty": 1.2,
    })
    payload = ENGINE.build_worker_payload(req, tmp_path, tmp_path / "o.wav")
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.85
    assert payload["top_k"] == 20
    assert isinstance(payload["top_k"], int)
    assert payload["repetition_penalty"] == 1.2


def test_speed_stays_noop_data():
    # Audit 2026-09-18 §4: the speed dead no-op is DATA and must not regress.
    spec = next(s for s in ENGINE.param_specs() if s.name == "speed")
    assert spec.exposed is False
    assert spec.not_exposed_reason == "no-op"


# --- worker regression: mock the upstream generate ----------------------------


class _Chunk:
    def __init__(self, audio, sample_rate):
        self.audio = audio
        self.sample_rate = sample_rate


# Minimal numpy stand-in: the worker's post-processing only needs
# asarray/concatenate/abs/max/clip and a tobytes-able sequence. Keeps the
# worker regression runnable in the sidecar env (which has no numpy).
class _FArr(list):
    @property
    def size(self):
        return len(self)

    def __mul__(self, factor):
        return _FArr(v * factor for v in self)

    def astype(self, _dtype):
        return self

    def tobytes(self):
        return b"".join(
            struct.pack("<h", max(-32767, min(32767, int(v * 32767)))) for v in self
        )


class _FNp:
    float32 = "float32"

    def asarray(self, x, dtype=None):
        return _FArr(x)

    def concatenate(self, seq):
        return _FArr(v for arr in seq for v in arr)

    def abs(self, x):
        return _FArr(abs(v) for v in x)

    def max(self, x):
        return max(x) if len(x) else 0.0

    def clip(self, x, lo, hi):
        return _FArr(min(hi, max(lo, v)) for v in x)


class _StubModel:
    def __init__(self, captured):
        self.captured = captured

    def generate(self, **kwargs):
        self.captured.append(kwargs)
        return [_Chunk(_FArr([0.1] * 2400), 24000)]


class _FakeUtils(types.ModuleType):
    def __init__(self, captured):
        super().__init__("mlx_audio.tts.utils")
        self.captured = captured

    def load_model(self, weights_dir):
        assert weights_dir  # comes from the payload
        return _StubModel(self.captured)


@pytest.fixture()
def fake_mlx(monkeypatch):
    captured = []
    utils = _FakeUtils(captured)
    mlx_audio = types.ModuleType("mlx_audio")
    tts = types.ModuleType("mlx_audio.tts")
    mlx_audio.tts = tts
    tts.utils = utils
    monkeypatch.setitem(sys.modules, "mlx_audio", mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils)
    monkeypatch.setitem(sys.modules, "numpy", _FNp())
    return captured


def _run_worker(payload, fake_mlx, monkeypatch, tmp_path):
    out = tmp_path / "out.wav"
    payload = {**payload, "output": str(out)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    buffer = io.StringIO()
    monkeypatch.setattr("sys.stdout", buffer)
    from voiceclone_sidecar.engines import qwen3_tts_worker as worker

    worker.main()
    assert out.exists()
    return buffer.getvalue()


def test_worker_forwards_lowercase_lang_code(fake_mlx, monkeypatch, tmp_path):
    out = _run_worker({"weights_dir": "/w", "text": "你好", "lang_code": "japanese",
                       "temperature": 0.9, "top_p": 1.0, "top_k": 50,
                       "repetition_penalty": 1.05},
                      fake_mlx, monkeypatch, tmp_path)
    assert "RESULT: " in out
    kwargs = fake_mlx[0]
    assert kwargs["lang_code"] == "japanese"  # lowercase, matching weight keys
    assert "language" not in kwargs  # the silent-no-op key is gone
    assert kwargs["verbose"] is False
    # Issue #48: the sampling defaults always ride along.
    assert kwargs["temperature"] == 0.9
    assert kwargs["top_p"] == 1.0
    assert kwargs["top_k"] == 50
    assert kwargs["repetition_penalty"] == 1.05


def test_worker_omits_lang_code_for_auto(fake_mlx, monkeypatch, tmp_path):
    _run_worker({"weights_dir": "/w", "text": "你好"}, fake_mlx, monkeypatch, tmp_path)
    assert "lang_code" not in fake_mlx[0]


def test_worker_forwards_changed_sampling_params(fake_mlx, monkeypatch, tmp_path):
    _run_worker({"weights_dir": "/w", "text": "hi", "temperature": 0.7, "top_k": 20,
                 "top_p": 1.0, "repetition_penalty": 1.05},
                fake_mlx, monkeypatch, tmp_path)
    kwargs = fake_mlx[0]
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_k"] == 20
    assert kwargs["top_p"] == 1.0  # declared default always sent
    assert kwargs["repetition_penalty"] == 1.05
