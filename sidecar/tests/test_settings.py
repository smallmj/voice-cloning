"""Contract tests for the issue-#9 surface: BYOK keys, engine disclosure,
cloud bindings, and capability metadata. Runs against the real sidecar
process (conftest), which uses an in-memory key backend and the fake-key
test engine."""

from __future__ import annotations

import io
import wave

import httpx


def create_voice(client: httpx.Client) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(3.5 * 16000))
    r = client.post(
        "/voices",
        data={"name": "测试音色", "description": ""},
        files={"file": ("ref.wav", buf.getvalue(), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


# --- engine disclosure ------------------------------------------------------


def test_engines_expose_issue9_metadata(client):
    engines = {e["id"]: e for e in client.get("/engines").json()["engines"]}
    cloud = engines["qwen3-tts-vc-cloud"]
    assert cloud["requires_key"] is True
    assert cloud["key_configured"] is False
    assert "0.8 元" in cloud["billing_note"]
    assert "汉字" in cloud["billing_note"]  # the Chinese unit-price rule
    assert "训练" in cloud["data_usage_note"]
    param_names = [p["name"] for p in cloud["params"]]
    # canonical `language` (issue #23) + the issue #38 emotion DATA annotation
    # (exposed=False, no-op): emotion is a text-embedded vendor convention.
    assert param_names == ["language", "emotion"]

    fake = engines["fake"]
    assert fake["requires_key"] is False
    assert fake["key_configured"] is None
    assert fake["params"] == []


# --- settings/keys contract ---------------------------------------------------


def test_key_endpoints_roundtrip(client):
    keys = client.get("/settings/keys").json()["keys"]
    ids = {k["engine_id"] for k in keys}
    assert {"qwen3-tts-vc-cloud", "fake-key"} <= ids

    r = client.put("/settings/keys", json={"engine_id": "fake-key", "key": "sk-abc"})
    assert r.status_code == 200
    assert r.json() == {"engine_id": "fake-key", "configured": True}

    engines = {e["id"]: e for e in client.get("/engines").json()["engines"]}
    assert engines["fake-key"]["key_configured"] is True
    # The contract never echoes key values.
    assert "sk-abc" not in r.text

    r = client.delete("/settings/keys/fake-key")
    assert r.status_code == 200
    engines = {e["id"]: e for e in client.get("/engines").json()["engines"]}
    assert engines["fake-key"]["key_configured"] is False


def test_key_endpoints_validate(client):
    # Unknown engine.
    assert client.put("/settings/keys", json={"engine_id": "nope", "key": "x"}).status_code == 404
    # Engine that does not use keys.
    r = client.put("/settings/keys", json={"engine_id": "fake", "key": "x"})
    assert r.status_code == 422
    # Empty key.
    r = client.put("/settings/keys", json={"engine_id": "fake-key", "key": "   "})
    assert r.status_code == 422
    # Missing body fields.
    assert client.put("/settings/keys", json={"engine_id": "fake-key"}).status_code == 422


# --- generation gating for BYOK engines -----------------------------------------


def test_generation_without_key_is_refused_with_guidance(client):
    r = client.post(
        "/generations", json={"engine_id": "fake-key", "text": "你好"}
    )
    assert r.status_code == 409
    assert "API Key" in r.json()["detail"]


def test_generation_without_voice_is_refused(client):
    assert (
        client.put("/settings/keys", json={"engine_id": "fake-key", "key": "sk-abc"}).status_code
        == 200
    )
    r = client.post("/generations", json={"engine_id": "fake-key", "text": "你好"})
    assert r.status_code == 422
    assert "音色" in r.json()["detail"]


def test_generation_binds_voice_and_records_cloud_binding(client):
    voice_id = create_voice(client)
    r = client.post(
        "/generations",
        json={"engine_id": "fake-key", "text": "你好，世界！", "voice_id": voice_id},
    )
    assert r.status_code == 200, r.text
    record = r.json()
    # The bound voice_id minted by bind_reference reached the engine.
    assert record["model_version"].startswith("fake-voice-")
    assert record["status"] == "succeeded"
    # The binding now carries the cloud voice id (issue #9: one voice, one
    # binding per engine, local and cloud side by side).
    voice = client.get(f"/voices/{voice_id}").json()
    binding = voice["bindings"]["fake-key"]
    assert binding["voice_id"].startswith("fake-voice-")
    assert binding["status"] == "ready"
    # Cost estimate is recorded for a cloud run.
    assert record["cost"] is not None and record["cost"] > 0
