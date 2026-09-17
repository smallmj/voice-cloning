"""Contract tests for the voice library (issue #6).

Voices are engine-independent entities; the reference sample is the source
of truth and engine bindings are a rebuildable cache (ADR-0001).
"""

from __future__ import annotations

import io
import struct
import wave
from pathlib import Path

import httpx


def wav_bytes(seconds: float = 5.0, rate: int = 16000) -> bytes:
    frames = int(rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(8000 * (i % 100) / 100)) for i in range(frames)
        ))
    return buf.getvalue()


def create_voice(client: httpx.Client, name: str = "测试音色", **kwargs) -> dict:
    files = {"file": ("ref.wav", wav_bytes(kwargs.pop("seconds", 5.0)), "audio/wav")}
    data = {"name": name, "description": "一段说明"}
    if "description" in kwargs:
        data["description"] = kwargs.pop("description")
    avatar = kwargs.pop("avatar", None)
    if avatar is not None:
        files["avatar"] = ("avatar.png", avatar, "image/png")
    r = client.post("/voices", data=data, files=files)
    assert r.status_code == 200, r.text
    return r.json()


# --- creation + hard validation ---------------------------------------------


def test_create_voice_with_metadata_and_avatar(client):
    voice = create_voice(client, avatar=b"\x89PNG\r\n\x1a\nfakepng")
    assert voice["id"]
    assert voice["name"] == "测试音色"
    assert voice["description"] == "一段说明"
    assert voice["avatar"] is not None
    ref = voice["reference"]
    assert ref["format"] == "wav"
    assert 4.9 < ref["duration_seconds"] < 5.2
    assert ref["sha256"]


def test_get_voice_and_reference_audio(client):
    voice = create_voice(client)
    r = client.get(f"/voices/{voice['id']}")
    assert r.status_code == 200
    r = client.get(f"/voices/{voice['id']}/reference")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")


def test_avatar_is_served(client):
    png = b"\x89PNG\r\n\x1a\nfakepng"
    voice = create_voice(client, avatar=png)
    r = client.get(f"/voices/{voice['id']}/avatar")
    assert r.status_code == 200
    assert r.content == png


def test_voice_without_avatar_returns_404_on_avatar_route(client):
    voice = create_voice(client)
    r = client.get(f"/voices/{voice['id']}/avatar")
    assert r.status_code == 404


def test_reject_unsupported_audio_format(client):
    r = client.post(
        "/voices",
        data={"name": "x"},
        files={"file": ("ref.txt", b"not audio", "text/plain")},
    )
    assert r.status_code == 422
    assert "格式" in r.json()["detail"]


def test_reject_oversized_audio(client, monkeypatch_tmp=None):
    big = b"\x00" * (21 * 1024 * 1024)
    r = client.post("/voices", data={"name": "x"}, files={"file": ("ref.wav", big, "audio/wav")})
    assert r.status_code == 422
    assert "过大" in r.json()["detail"]


def test_reject_too_short_audio(client):
    r = client.post(
        "/voices",
        data={"name": "x"},
        files={"file": ("ref.wav", wav_bytes(seconds=1.0), "audio/wav")},
    )
    assert r.status_code == 422
    assert "太短" in r.json()["detail"]


def test_reject_empty_name(client):
    r = client.post("/voices", data={"name": "  "}, files={"file": ("ref.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 422


def test_invalid_wav_is_rejected_with_reason(client):
    r = client.post(
        "/voices", data={"name": "x"}, files={"file": ("ref.wav", b"RIFFbroken", "audio/wav")}
    )
    assert r.status_code == 422
    assert "WAV" in r.json()["detail"]


# --- library ----------------------------------------------------------------


def test_list_voices_returns_created_voices(client):
    before = {v["id"] for v in client.get("/voices").json()["voices"]}
    a = create_voice(client, name="甲")
    b = create_voice(client, name="乙")
    after = {v["id"] for v in client.get("/voices").json()["voices"]}
    assert {a["id"], b["id"]} <= (after - before)


def test_update_voice_metadata(client):
    voice = create_voice(client)
    r = client.patch(f"/voices/{voice['id']}", json={"name": "新名", "description": "新说明"})
    assert r.status_code == 200
    assert r.json()["name"] == "新名"
    assert r.json()["description"] == "新说明"
    # Avatar must survive a metadata edit.
    assert client.get(f"/voices/{voice['id']}/avatar").status_code == (
        200 if voice.get("avatar") else 404
    )


def test_delete_voice_removes_record_and_files(client, sidecar):
    voice = create_voice(client)
    voice_dir = Path(sidecar["data_dir"]) / "voices" / voice["id"]
    assert voice_dir.is_dir()
    r = client.delete(f"/voices/{voice['id']}")
    assert r.status_code == 200
    assert client.get(f"/voices/{voice['id']}").status_code == 404
    assert not voice_dir.exists()
    assert voice["id"] not in {v["id"] for v in client.get("/voices").json()["voices"]}


# --- engine bindings --------------------------------------------------------


def test_bind_voice_on_engine_and_persist(client, sidecar):
    voice = create_voice(client)
    r = client.post(f"/voices/{voice['id']}/bindings/fake")
    assert r.status_code == 200
    binding = r.json()
    assert binding["status"] == "ready"
    assert binding["reference_sha256"] == voice["reference"]["sha256"]
    # The binding is persisted on the voice record.
    stored = client.get(f"/voices/{voice['id']}").json()
    assert stored["bindings"]["fake"]["status"] == "ready"


def test_bind_on_unknown_engine_is_404(client):
    voice = create_voice(client)
    r = client.post(f"/voices/{voice['id']}/bindings/no-such-engine")
    assert r.status_code == 404


def test_binding_is_rebuildable_cache(client):
    """Rebinding replaces the binding from the reference — no unique state."""
    voice = create_voice(client)
    first = client.post(f"/voices/{voice['id']}/bindings/fake").json()
    second = client.post(f"/voices/{voice['id']}/bindings/fake").json()
    assert first["reference_sha256"] == second["reference_sha256"]
    assert first["status"] == second["status"] == "ready"


def test_delete_voice_removes_bindings(client):
    voice = create_voice(client)
    client.post(f"/voices/{voice['id']}/bindings/fake")
    client.delete(f"/voices/{voice['id']}")
    # Nothing left to query: the voice (and with it every binding) is gone.
    assert client.get(f"/voices/{voice['id']}").status_code == 404


# --- generation with a voice -------------------------------------------------


def test_generation_with_voice_succeeds_and_persists_binding(client):
    voice = create_voice(client)
    r = client.post("/generations", json={"engine_id": "fake", "text": "你好", "voice_id": voice["id"]})
    assert r.status_code == 200, r.text
    record = r.json()
    assert record["voice_id"] == voice["id"]
    assert record["status"] == "succeeded"
    assert record["binding"]["status"] == "ready"
    stored = client.get(f"/voices/{voice['id']}").json()
    assert stored["bindings"]["fake"]["status"] == "ready"


def test_generation_with_unknown_voice_is_404(client):
    r = client.post("/generations", json={"engine_id": "fake", "text": "你好", "voice_id": "nope"})
    assert r.status_code == 404


def test_generation_without_voice_still_works(client):
    r = client.post("/generations", json={"engine_id": "fake", "text": "无音色生成"})
    assert r.status_code == 200
    assert r.json()["voice_id"] is None
