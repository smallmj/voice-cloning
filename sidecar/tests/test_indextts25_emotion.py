"""Issue #37: IndexTTS emotion controls + sampling parameters.

The parameter surface is aligned with the official webui: four emotion modes,
emotion weight, the 8-dim emotion vector, emotion text + random sampling,
and the engine-layer sampling block (temperature/top_p/top_k/num_beams/
repetition_penalty/length_penalty/max_mel_tokens/do_sample). Everything is
declared data (ADR-0018): exposed specs carry applies_to, mode-gated specs
carry ignored_when, and the synthesize adapter re-enforces the gating.
"""

from __future__ import annotations

import pytest

from voiceclone_sidecar.engines.indextts25_base import (
    EMO_MODE_CHOICES,
    emo_vector_to_wire,
    param_specs_for,
    prepare_synthesis,
)


def _spec_map():
    return {s.name: s for s in param_specs_for("indextts-25-mps")}


# --- declarations -------------------------------------------------------------


def test_sampling_params_exposed_with_official_defaults():
    specs = _spec_map()
    official = {
        "temperature": (0.1, 2.0, 0.8),
        "top_p": (0.0, 1.0, 0.8),
        "top_k": (0, 100, 30),
        "num_beams": (1, 10, 3),
        "repetition_penalty": (1.0, 20.0, 10.0),
        "length_penalty": (-2.0, 2.0, 0.0),
        "max_mel_tokens": (100, 6000, 1500),
    }
    for name, (lo, hi, default) in official.items():
        spec = specs[name]
        assert spec.exposed and spec.layer == "engine", name
        assert spec.default == default, name
        assert spec.min == lo and spec.max == hi, name
    # do_sample is a verified NO-OP upstream (inference_speech hardcodes
    # do_sample=True) — declared as data, never a slider (ADR-0018 decision 3).
    assert specs["do_sample"].exposed is False
    assert specs["do_sample"].not_exposed_reason == "no-op"


def test_emotion_specs_exposed_and_mode_gated():
    specs = _spec_map()
    assert specs["emo_mode"].choices == EMO_MODE_CHOICES
    assert specs["emo_mode"].default == EMO_MODE_CHOICES[0]
    assert specs["emo_weight"].default == 0.65
    assert specs["emo_weight"].min == 0.0 and specs["emo_weight"].max == 1.0
    # 情感参考音频是生成期上传输入（kind audio），仅参考音频模式可见。
    assert specs["emo_audio"].kind == "audio"
    assert all(p.startswith("emo_mode!=") for p in specs["emo_audio"].ignored_when)
    # 文本模式专属参数。
    for name in ("emo_text", "emo_random"):
        assert specs[name].ignored_when == ("emo_mode!=" + EMO_MODE_CHOICES[3],), name


def test_emo_vector_is_fixed_order_8_dim_array():
    spec = _spec_map()["emo_vector"]
    assert spec.kind == "array" and spec.exposed
    assert [i.label for i in spec.items] == [
        "高兴", "愤怒", "悲伤", "恐惧", "厌恶", "低落", "惊喜", "平静",
    ]
    assert len(spec.items) == 8 and spec.max_items == 8
    assert all(i.min == 0.0 and i.max == 1.0 for i in spec.items)


def test_every_exposed_spec_carries_applies_to():
    for spec in param_specs_for("indextts-25-mps"):
        if spec.exposed:
            assert spec.applies_to is not None, spec.name


# --- wire converters ------------------------------------------------------------


def test_emo_vector_to_wire_validates_and_clamps():
    vec = emo_vector_to_wire("0.9,0,0,0,0,0,0,0.1")
    assert vec == [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    assert emo_vector_to_wire("0.9,0") is None  # wrong dimensionality
    assert emo_vector_to_wire("a,b,c,d,e,f,g,h") is None
    assert emo_vector_to_wire("2,0,0,0,0,0,0,-1") == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert emo_vector_to_wire("") is None  # empty -> engine default, not zeros


# --- synthesize adapter (mode gating re-enforced engine-side) --------------------


def test_same_as_reference_sends_no_emotion_keys():
    _, wire = prepare_synthesis("你好", {}, lambda m: None)
    assert not any(k.startswith("emo") or k in ("use_emo_text", "use_random") for k in wire)


def test_reference_audio_mode_requires_the_upload():
    with pytest.raises(ValueError, match="情感参考音频"):
        prepare_synthesis("你好", {"emo_mode": EMO_MODE_CHOICES[1]}, lambda m: None)
    _, wire = prepare_synthesis(
        "你好",
        {"emo_mode": EMO_MODE_CHOICES[1], "emo_audio": "/audio/upload-x.wav"},
        lambda m: None,
    )
    assert wire["emo_audio_prompt"] == "/audio/upload-x.wav"
    assert wire["emo_alpha"] == 0.65  # the official default rides along


def test_vector_mode_sends_scaled_vector_and_alpha():
    _, wire = prepare_synthesis(
        "你好",
        {"emo_mode": EMO_MODE_CHOICES[2], "emo_vector": "0.9,0,0,0,0,0,0,0.1", "emo_weight": "0.65"},
        lambda m: None,
    )
    assert wire["emo_vector"] == [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
    assert wire["emo_alpha"] == 0.65


def test_vector_mode_with_invalid_vector_is_refused():
    with pytest.raises(ValueError, match="情感向量"):
        prepare_synthesis("你好", {"emo_mode": EMO_MODE_CHOICES[2], "emo_vector": "0.5"}, lambda m: None)


def test_text_mode_sends_qwen_emotion_keys():
    _, wire = prepare_synthesis(
        "你好",
        {"emo_mode": EMO_MODE_CHOICES[3], "emo_text": "非常悲伤", "emo_random": "true"},
        lambda m: None,
    )
    assert wire["use_emo_text"] is True
    assert wire["emo_text"] == "非常悲伤"
    assert wire["use_random"] is True


def test_sampling_params_forwarded_when_set():
    _, wire = prepare_synthesis(
        "你好",
        {"temperature": "1.2", "top_k": "40", "do_sample": "false"},
        lambda m: None,
    )
    assert wire["temperature"] == 1.2
    assert wire["top_k"] == 40
    # do_sample is a no-op upstream — never forwarded.
    assert "do_sample" not in wire
    # Unset values stay out: upstream defaults stay authoritative.
    assert "top_p" not in wire and "num_beams" not in wire


def test_stale_emo_params_of_other_modes_never_reach_wire():
    """A rerun carries remembered params from ALL modes — the adapter must
    gate them, not the (possibly stale) frontend."""
    _, wire = prepare_synthesis(
        "你好",
        {"emo_vector": "0.9,0,0,0,0,0,0,0", "emo_text": "开心", "emo_audio": "/x.wav"},
        lambda m: None,
    )
    assert "emo_vector" not in wire
    assert "emo_text" not in wire
    assert "emo_audio_prompt" not in wire


# --- generation-time upload route + pipeline resolution --------------------------


def test_upload_audio_endpoint_returns_bare_filename(client):
    r = client.post(
        "/uploads/audio",
        files={"file": ("emo.wav", b"fake-wav-bytes", "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["file"] == r.json()["file"]
    assert "/" not in body["file"] and "\\" not in body["file"]
    assert body["size_bytes"] == len(b"fake-wav-bytes")


def test_upload_audio_requires_a_file(client):
    assert client.post("/uploads/audio").status_code == 422


def test_pipeline_resolves_emo_audio_inside_audio_dir(sidecar, client):
    import json

    audio_dir = sidecar["audio_dir"]
    (audio_dir / "emo-resolve.wav").write_bytes(b"x")
    # fake engine accepts anything; the point is the resolved path in params
    r = client.post(
        "/generations",
        json={
            "engine_id": "fake",
            "text": "测试",
            "params": {"emo_mode": EMO_MODE_CHOICES[1], "emo_audio": "emo-resolve.wav"},
        },
    )
    assert r.status_code == 200
    params = json.loads(json.dumps(r.json().get("params", {})))
    assert params.get("emo_audio", "").endswith("emo-resolve.wav")


def test_pipeline_rejects_emo_audio_traversal(client):
    r = client.post(
        "/generations",
        json={
            "engine_id": "fake",
            "text": "测试",
            "params": {"emo_audio": "../secrets.db"},
        },
    )
    assert r.status_code == 422


def test_pipeline_rejects_missing_emo_audio(client):
    r = client.post(
        "/generations",
        json={"engine_id": "fake", "text": "测试", "params": {"emo_audio": "gone.wav"}},
    )
    assert r.status_code == 422
