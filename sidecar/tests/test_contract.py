"""Contract tests: assert only observable outputs of the sidecar's public
contract (HTTP responses, audio files, WebSocket events, error codes)."""

from __future__ import annotations

import struct
import wave

import httpx

EXPECTED_CAPABILITY_KEYS = {
    "languages",
    "voice_cloning",
    "voice_design",
    "pronunciation_control",
    "emotion",
    "commercial_license",
    "cross_device_use",
    "upload_used_for_training",
    "api_closed_loop",
}


# --- auth -----------------------------------------------------------------


def test_health_without_token_is_rejected(sidecar):
    r = httpx.get(f"{sidecar['base_url']}/health", timeout=10)
    assert r.status_code == 401


def test_health_with_wrong_token_is_rejected(sidecar):
    r = httpx.get(
        f"{sidecar['base_url']}/health",
        headers={"Authorization": "Bearer not-the-token"},
        timeout=10,
    )
    assert r.status_code == 401


def test_health_with_valid_token(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_websocket_without_token_is_rejected(sidecar, ws_url):
    import asyncio

    import websockets

    async def run():
        # Rejected connections are accepted then closed with 4401.
        ws = await websockets.connect(ws_url)
        try:
            await asyncio.wait_for(ws.recv(), timeout=10)
            raise AssertionError("expected the connection to be closed")
        except websockets.ConnectionClosed as exc:
            assert exc.rcvd is not None and exc.rcvd.code == 4401
        finally:
            await ws.close()

    asyncio.run(asyncio.wait_for(run(), timeout=10))


def auth_headers(sidecar):
    return {"Authorization": f"Bearer {sidecar['token']}"}


# --- engine registry + capabilities ---------------------------------------


def test_engines_list_includes_fake_engine_with_capabilities(client):
    r = client.get("/engines")
    assert r.status_code == 200
    engines = r.json()["engines"]
    fake = [e for e in engines if e["id"] == "fake"]
    assert len(fake) == 1
    caps = fake[0]["capabilities"]
    assert set(caps.keys()) >= EXPECTED_CAPABILITY_KEYS, "capability declaration is incomplete"


# --- generation end-to-end -------------------------------------------------


def test_full_generation_text_to_audio(client, sidecar):
    r = client.post("/generations", json={"engine_id": "fake", "text": "你好，世界。Hello!"})
    assert r.status_code == 200, r.text
    record = r.json()
    gen_id = record["id"]
    assert record["status"] == "succeeded"
    assert record["engine_id"] == "fake"
    assert record["audio_url"].startswith("/audio/")

    # Record is retrievable afterwards.
    r2 = client.get(f"/generations/{gen_id}")
    assert r2.status_code == 200
    assert r2.json()["id"] == gen_id

    # The audio artifact is a real, playable WAV.
    audio = client.get(record["audio_url"])
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"
    with wave.open(__import__("io").BytesIO(audio.content)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() > 0
        # peak sanity: samples stay inside int16 and are not all silence
        frames = w.readframes(w.getnframes())
        samples = struct.unpack(f"<{len(frames)//2}h", frames)
        assert max(abs(s) for s in samples) > 1000


def test_generation_to_unknown_engine_is_404(client):
    r = client.post("/generations", json={"engine_id": "does-not-exist", "text": "hi"})
    assert r.status_code == 404


def test_generation_with_empty_text_is_422(client):
    r = client.post("/generations", json={"engine_id": "fake", "text": "   "})
    assert r.status_code == 422


# --- websocket log streaming -----------------------------------------------


def test_generation_streams_logs_over_websocket(client, sidecar, ws_url):
    import asyncio
    import json as jsonlib

    import websockets

    async def run():
        # Subscribe BEFORE generating so we observe the whole stream.
        async with websockets.connect(ws_url, additional_headers=auth_headers(sidecar)) as ws:
            r = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.post("/generations", json={"engine_id": "fake", "text": "日志测试"}),
            )
            assert r.status_code == 200
            gen_id = r.json()["id"]

            events = []
            while True:
                event = jsonlib.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                events.append(event)
                if "wrote" in event["message"]:
                    break
            assert any(e["generation_id"] == gen_id and e["type"] == "log" for e in events)
            assert any("synthesizing" in e["message"] for e in events)
            assert events[-1]["ts"] >= events[0]["ts"]
            return gen_id

    asyncio.run(asyncio.wait_for(run(), timeout=30))


def test_engines_list_exposes_vendor_label_for_keyed_engines(client):
    """Issue #53: the key UI groups cloud engines by vendor so one vendor's
    key is entered once (the two Qwen3-TTS cloud engines share 阿里百炼)."""
    engines = client.get("/engines").json()["engines"]
    # The fake test doubles mimic the BYOK surface but have no vendor; only
    # real cloud engines must carry the vendor label the key UI groups by.
    keyed = [e for e in engines if e["requires_key"] and not e["id"].startswith("fake")]
    assert keyed, "expected at least one BYOK cloud engine"
    for e in keyed:
        assert isinstance(e["vendor_label"], str) and e["vendor_label"]
    qwen = {
        e["vendor_label"]
        for e in engines
        if e["id"].startswith("qwen3-tts-") and e["requires_key"]
    }
    assert qwen == {"阿里百炼"}
    for e in engines:
        if not e["requires_key"]:
            assert e["vendor_label"] is None
