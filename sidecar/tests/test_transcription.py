"""Contract tests: reference diagnostics + transcription (issue #8)."""

from __future__ import annotations

import io
import math
import struct
import wave

import httpx
import pytest

from .test_voices import create_voice, wav_bytes


def sine_wav_bytes(seconds: float = 4.0, freq: float = 220.0, rate: int = 16000) -> bytes:
    frames = int(rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(frames)
        ))
    return buf.getvalue()


# --- diagnostics --------------------------------------------------------------


def test_diagnose_upload_returns_diagnostics_with_advice(client):
    r = client.post(
        "/diagnose",
        files={"file": ("ref.wav", sine_wav_bytes(), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    ids = {d["id"] for d in data["diagnostics"]}
    assert {"snr", "speaker", "clipping", "silence"} <= ids
    for d in data["diagnostics"]:
        assert d["status"] in ("good", "warn", "bad")
        assert d["message"].strip()
        assert d["advice"].strip()


def test_diagnose_rejects_empty_upload(client):
    r = client.post("/diagnose", files={"file": ("ref.wav", b"", "audio/wav")})
    assert r.status_code == 422


def test_voice_diagnose_never_modifies_the_reference(client):
    """The app must not touch user material: byte-identical before/after."""
    voice = create_voice(client, seconds=4.0)
    r0 = client.get(f"/voices/{voice['id']}/reference")
    before = r0.content
    r = client.get(f"/voices/{voice['id']}/diagnose")
    assert r.status_code == 200, r.text
    assert r.json()["diagnostics"]
    r1 = client.get(f"/voices/{voice['id']}/reference")
    assert r1.content == before


def test_voice_diagnose_unknown_voice(client):
    assert client.get("/voices/nope/diagnose").status_code == 404


# --- transcription providers + settings ---------------------------------------


def test_providers_lists_local_and_fake_engine(client):
    r = client.get("/transcription/providers")
    assert r.status_code == 200
    data = r.json()
    assert data["default"] == "local"
    assert data["provider"] == "local"
    assert data["local"]["supported"] is True
    assert any(e["id"] == "fake" for e in data["engines"])


def test_set_provider_rejects_unknown(client):
    r = client.put("/transcription/provider", json={"provider": "no-such-vendor"})
    assert r.status_code == 422


def test_set_provider_accepts_registered_engine_and_persists(client):
    r = client.put("/transcription/provider", json={"provider": "fake"})
    assert r.status_code == 200, r.text
    assert client.get("/transcription/providers").json()["provider"] == "fake"
    # restore default for other tests
    client.put("/transcription/provider", json={"provider": "local"})


def test_local_tool_status_reports_uninstalled(client):
    r = client.get("/transcription/local/status")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "transcribe-local"
    assert data["installed"] is False


def test_transcribe_via_engine_provider_stores_transcript(client):
    """Cloud transcription reuses a registered engine — no new vendor."""
    client.put("/transcription/provider", json={"provider": "fake"})
    voice = create_voice(client, seconds=4.0)
    r = client.post(f"/voices/{voice['id']}/transcribe", json={})
    assert r.status_code == 200, r.text
    assert r.json()["reference"]["transcript"].strip()
    stored = client.get(f"/voices/{voice['id']}").json()
    assert stored["reference"]["transcript"] == r.json()["reference"]["transcript"]
    client.put("/transcription/provider", json={"provider": "local"})


def test_transcribe_with_uninstalled_local_tool_is_actionable(client):
    voice = create_voice(client, seconds=4.0)
    r = client.post(f"/voices/{voice['id']}/transcribe", json={})
    assert r.status_code == 409
    assert "安装" in r.json()["detail"]  # the error says what to do next


def test_transcribe_unknown_voice(client):
    assert client.post("/voices/nope/transcribe", json={}).status_code == 404


# --- reference-text auto-fill at generation time --------------------------------


def test_generation_auto_fills_ref_text_from_transcript(client):
    voice = create_voice(client, seconds=4.0)
    client.put("/transcription/provider", json={"provider": "fake"})
    r = client.post(
        "/generations",
        json={"engine_id": "fake-ref-text", "text": "你好", "voice_id": voice["id"]},
    )
    assert r.status_code == 200, r.text
    record = r.json()
    assert record["status"] == "succeeded"
    assert record["params"]["ref_text"].strip()
    # The transcript was persisted, so a rerun never re-transcribes by hand.
    stored = client.get(f"/voices/{voice['id']}").json()
    assert stored["reference"]["transcript"] == record["params"]["ref_text"]
    client.put("/transcription/provider", json={"provider": "local"})


def test_generation_with_ref_text_engine_fails_actionably_without_transcription(client):
    """Local provider uninstalled + no transcript ⇒ a 409 that says what to do,
    never a silent send-to-cloud fallback."""
    client.put("/transcription/provider", json={"provider": "local"})
    voice = create_voice(client, seconds=4.0)
    r = client.post(
        "/generations",
        json={"engine_id": "fake-ref-text", "text": "你好", "voice_id": voice["id"]},
    )
    assert r.status_code == 409, r.text
    assert "转写" in r.json()["detail"]


def test_ref_text_engine_without_voice_still_generates(client):
    r = client.post("/generations", json={"engine_id": "fake-ref-text", "text": "无音色"})
    assert r.status_code == 200, r.text


def test_engines_expose_requires_reference_text(client):
    engines = {e["id"]: e for e in client.get("/engines").json()["engines"]}
    assert engines["fake"]["capabilities"]["requires_reference_text"] is False
    assert engines["fake-ref-text"]["capabilities"]["requires_reference_text"] is True
