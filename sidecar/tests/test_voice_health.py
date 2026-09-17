"""Contract tests for issue #12: cloud voice health check + auto rebuild.

Runs against the real sidecar process (conftest), which registers the
fake-vanishing-voice and fake-broken-rebuild test engines. The vendor's
"silently recycle enrolled voices" behavior is simulated entirely by the
fakes — no real vendor deletion is needed.
"""

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


def _set_key(client: httpx.Client, engine_id: str) -> None:
    r = client.put("/settings/keys", json={"engine_id": engine_id, "key": "sk-test"})
    assert r.status_code == 200, r.text


def _generate(client: httpx.Client, engine_id: str, voice_id: str) -> httpx.Response:
    return client.post(
        "/generations", json={"engine_id": engine_id, "text": "你好，世界！", "voice_id": voice_id}
    )


def test_vanished_cloud_voice_is_rebuilt_automatically(client):
    """Gen 1 enrolls and succeeds; the vendor then deletes the voice; gen 2
    must detect it, rebuild from the local reference and succeed — with a
    NEW cloud voice id and a ready binding, no user intervention."""
    _set_key(client, "fake-vanishing-voice")
    voice_id = create_voice(client)

    r1 = _generate(client, "fake-vanishing-voice", voice_id)
    assert r1.status_code == 200, r1.text
    first_voice_id = client.get(f"/voices/{voice_id}").json()["bindings"][
        "fake-vanishing-voice"
    ]["voice_id"]

    r2 = _generate(client, "fake-vanishing-voice", voice_id)
    assert r2.status_code == 200, r2.text
    binding = client.get(f"/voices/{voice_id}").json()["bindings"]["fake-vanishing-voice"]
    assert binding["voice_id"] != first_voice_id  # rebuilt, not reused
    assert binding["status"] == "ready"
    assert r2.json()["model_version"] == binding["voice_id"]


def test_failed_rebuild_marks_binding_unavailable_and_spares_other_bindings(client):
    """When the rebuild itself fails, THIS binding is marked unavailable with
    a clear reason — and a binding on another engine of the same voice stays
    ready and usable."""
    _set_key(client, "fake-broken-rebuild")
    _set_key(client, "fake-key")
    voice_id = create_voice(client)

    # Gen 1 on the broken engine: enrollment succeeds, binding is ready.
    r1 = _generate(client, "fake-broken-rebuild", voice_id)
    assert r1.status_code == 200, r1.text

    # A healthy binding on a second cloud engine, created before the trouble.
    r_other = _generate(client, "fake-key", voice_id)
    assert r_other.status_code == 200, r_other.text

    # Gen 2: the voice probes as dead and the vendor refuses the rebuild.
    r2 = _generate(client, "fake-broken-rebuild", voice_id)
    assert r2.status_code == 502, r2.text
    assert "重建失败" in r2.json()["detail"]
    assert "模拟厂商拒绝重建" in r2.json()["detail"]

    bindings = client.get(f"/voices/{voice_id}").json()["bindings"]
    assert bindings["fake-broken-rebuild"]["status"] == "unavailable"
    assert "模拟厂商拒绝重建" in bindings["fake-broken-rebuild"]["error"]
    # Other bindings are untouched.
    assert bindings["fake-key"]["status"] == "ready"
    # And the healthy engine still generates with the same voice.
    r3 = _generate(client, "fake-key", voice_id)
    assert r3.status_code == 200, r3.text


def test_healthy_binding_generates_without_probe_noise(client):
    """A binding whose voice is alive just runs; the binding is not replaced."""
    _set_key(client, "fake-key")
    voice_id = create_voice(client)
    r1 = _generate(client, "fake-key", voice_id)
    assert r1.status_code == 200, r1.text
    first = client.get(f"/voices/{voice_id}").json()["bindings"]["fake-key"]["voice_id"]
    r2 = _generate(client, "fake-key", voice_id)
    assert r2.status_code == 200, r2.text
    assert (
        client.get(f"/voices/{voice_id}").json()["bindings"]["fake-key"]["voice_id"]
        == first
    )
