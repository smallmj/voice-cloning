"""Security hardening tests for issue #19.

Covers four independent problems from the audit:
1. ``/openapi.json`` / ``/docs`` / ``/redoc`` must not be exposed.
2. The raw sidecar token must no longer be accepted as a query parameter on
   non-media routes — only bearer auth works there.
3. Media routes (``/audio/*``, voice reference/avatar images, ``/ws/logs``)
   accept a short-TTL, media-scoped token fetched over bearer auth.
4. The generation-failure 500 detail must not echo the raw exception text.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import time
import uuid
import wave

import httpx
import pytest
import websockets


def auth_headers(sidecar) -> dict:
    return {"Authorization": f"Bearer {sidecar['token']}"}


def make_reference_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        # 4 seconds: clears the reference-audio minimum-duration gate.
        w.writeframes(b"\x00\x01" * 16000 * 4)
    return buf.getvalue()


@pytest.fixture()
def voice(client) -> str:
    resp = client.post(
        "/voices",
        data={"name": f"sec-{uuid.uuid4().hex[:8]}"},
        files={"file": ("ref.wav", make_reference_wav(), "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    voice_id = resp.json()["id"]
    return voice_id


def get_media_token(sidecar) -> dict:
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get("/media-token", headers=auth_headers(sidecar))
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# 1. Docs endpoints are disabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect"])
def test_docs_endpoints_disabled(sidecar, path):
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get(path, headers=auth_headers(sidecar))
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 2. Query token rejected on non-media routes
# ---------------------------------------------------------------------------


def test_query_token_rejected_on_read_route(sidecar):
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get(f"/engines?token={sidecar['token']}")
    assert resp.status_code == 401


def test_query_token_rejected_on_write_route(sidecar):
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.post(f"/voices?token={sidecar['token']}", json={"name": "x"})
    assert resp.status_code == 401


def test_bearer_auth_still_works(sidecar):
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get("/engines", headers=auth_headers(sidecar))
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. Media-scoped short-TTL token
# ---------------------------------------------------------------------------


def test_media_token_route_requires_bearer(sidecar):
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get("/media-token")
    assert resp.status_code == 401


def test_media_token_has_expiry(sidecar):
    payload = get_media_token(sidecar)
    assert payload["expires_at"] > time.time()
    assert payload["media_token"]


def test_media_token_works_on_audio_route(sidecar, voice, client):
    resp = client.post(
        "/generations",
        json={"voice_id": voice, "engine_id": "fake", "text": "测试"},
    )
    assert resp.status_code == 200, resp.text
    audio_url = resp.json()["audio_url"]
    payload = get_media_token(sidecar)
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get(f"{audio_url}?token={payload['media_token']}")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/")


def test_raw_token_rejected_on_media_route(sidecar):
    # The full read/write token must NOT be a valid media credential.
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get(f"/audio/nonexistent.wav?token={sidecar['token']}")
    assert resp.status_code == 401


def test_expired_media_token_rejected(sidecar):
    from voiceclone_sidecar.media_token import make_media_token

    stale = make_media_token(sidecar["token"], ttl_seconds=-10)
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get(f"/audio/nonexistent.wav?token={stale}")
    assert resp.status_code == 401


def test_media_token_from_other_token_rejected(sidecar):
    from voiceclone_sidecar.media_token import make_media_token

    foreign = make_media_token("not-the-real-token", ttl_seconds=60)
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get(f"/audio/nonexistent.wav?token={foreign}")
    assert resp.status_code == 401


def test_media_token_works_on_voice_images(sidecar, voice):
    payload = get_media_token(sidecar)
    with httpx.Client(base_url=sidecar["base_url"], timeout=10.0) as c:
        resp = c.get(f"/voices/{voice}/reference?token={payload['media_token']}")
        assert resp.status_code == 200, resp.text[:200]
        # Avatar is optional content: without one the route answers 404, but
        # it must NOT be 401 — media auth itself has passed.
        resp = c.get(f"/voices/{voice}/avatar?token={payload['media_token']}")
        assert resp.status_code != 401


def test_media_token_works_on_ws_logs(sidecar):
    payload = get_media_token(sidecar)
    ws_url = f"ws://127.0.0.1:{sidecar['base_url'].rsplit(':', 1)[1]}/ws/logs"

    async def run():
        async with websockets.connect(f"{ws_url}?token={payload['media_token']}") as ws:
            # Just prove the handshake succeeded: try to receive with a tiny
            # timeout; log traffic is not guaranteed immediately.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(ws.recv(), timeout=0.5)

    asyncio.run(run())


def test_ws_rejects_raw_query_token(sidecar):
    ws_url = f"ws://127.0.0.1:{sidecar['base_url'].rsplit(':', 1)[1]}/ws/logs"

    async def run():
        # The verifier accepts the handshake then closes with a custom 4401.
        conn = await websockets.connect(f"{ws_url}?token={sidecar['token']}")
        try:
            with pytest.raises(websockets.ConnectionClosed) as exc_info:
                await conn.recv()
        finally:
            await conn.close()
        assert exc_info.value.code == 4401

    asyncio.run(run())


def test_ws_without_token_rejected(sidecar):
    ws_url = f"ws://127.0.0.1:{sidecar['base_url'].rsplit(':', 1)[1]}/ws/logs"

    async def run():
        conn = await websockets.connect(ws_url)
        try:
            with pytest.raises(websockets.ConnectionClosed) as exc_info:
                await conn.recv()
        finally:
            await conn.close()
        assert exc_info.value.code == 4401

    asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. Generation failure detail does not leak the exception text
# ---------------------------------------------------------------------------


def test_generation_failure_detail_sanitized(sidecar, voice, client):
    # fake-broken engine registered only under VOICECLONE_TEST_ENGINES fails
    # with an exception whose text contains a filesystem-ish path.
    resp = client.post(
        "/generations",
        json={"voice_id": voice, "engine_id": "fake-broken", "text": "测试"},
    )
    assert resp.status_code == 500, resp.text
    detail = resp.json()["detail"]
    assert "/tmp" not in detail
    assert "boom" not in detail
    # The durable record still carries the diagnostic error for history.
    failed = [
        g
        for g in client.get("/generations").json()["records"]
        if g.get("error") and "model.bin" in g["error"]
    ]
    assert failed, "failed record keeps its error message"


def test_bad_request_for_missing_voice(client):
    resp = client.post(
        "/generations", json={"voice_id": "missing", "engine_id": "fake", "text": "x"}
    )
    assert resp.status_code == 404
