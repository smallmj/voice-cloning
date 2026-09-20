"""Direct unit tests for the shared generation pipeline (issue #20).

The point of extracting ``run_generation`` out of the ``create_app``
closure: the synthesis pipeline can now be exercised as a plain function
over an ``AppContext`` — no FastAPI app, no sidecar subprocess.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from voiceclone_sidecar.context import AppContext
from voiceclone_sidecar.engines.fake import FakeEngine
from voiceclone_sidecar.pipeline import run_generation
from voiceclone_sidecar.registry import Registry


@pytest.fixture()
def ctx(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    registry = Registry()
    registry.register(FakeEngine(output_dir=audio_dir))
    return AppContext(
        registry=registry,
        token="t",
        audio_dir=audio_dir,
        data_root=tmp_path,
    )


def test_run_generation_is_a_plain_function(ctx):
    """A full synthesis run with no app object anywhere."""
    record = asyncio.run(run_generation(ctx, "fake", "你好世界"))
    assert record["status"] == "succeeded"
    assert record["audio_url"].startswith("/audio/")
    assert record["normalized_text"]  # normalization layer applied
    stored = ctx.generation_store.get(record["id"])
    assert stored is not None and stored["status"] == "succeeded"


def test_run_generation_unknown_engine_404(ctx):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_generation(ctx, "nope", "text"))
    assert exc.value.status_code == 404


def test_run_generation_unknown_voice_404(ctx):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_generation(ctx, "fake", "text", voice_id="missing"))
    assert exc.value.status_code == 404


def _minimax_ctx(tmp_path):
    """A real MiniMaxCloudEngine behind a MockTransport: exercises the
    issue-#27 system-voice-without-a-voice-record path end to end."""
    import io, wave as _wave
    import httpx

    from voiceclone_sidecar.engines.minimax_cloud import MiniMaxCloudEngine
    from voiceclone_sidecar.secrets import KeyStore, MemoryBackend

    def wav_bytes():
        buf = io.BytesIO()
        with _wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x00" * 4800)
        return buf.getvalue()

    def handler(request):
        return httpx.Response(200, json={
            "data": {"audio": wav_bytes().hex()},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        })

    audio_dir = tmp_path / "audio-mm"
    audio_dir.mkdir(exist_ok=True)
    registry = Registry()
    store = KeyStore(backend=MemoryBackend())
    store.set("minimax-speech-cloud", "mk-test")
    registry.register(MiniMaxCloudEngine(
        output_dir=audio_dir, key_store=store,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    ))
    return AppContext(
        registry=registry, token="t", audio_dir=audio_dir, data_root=tmp_path,
    )


def test_cloud_engine_without_voice_or_system_voice_is_422(tmp_path):
    ctx = _minimax_ctx(tmp_path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_generation(ctx, "minimax-speech-cloud", "你好"))
    assert exc.value.status_code == 422
    assert "系统音色" in exc.value.detail


def test_cloud_engine_system_voice_generates_without_voice_record(tmp_path):
    ctx = _minimax_ctx(tmp_path)
    record = asyncio.run(run_generation(
        ctx, "minimax-speech-cloud", "你好", params={"system_voice": "male-qn-qingse"},
    ))
    assert record["status"] == "succeeded"
    assert record["voice_id"] is None
    # The record echoes the user params verbatim; the system_voice ->
    # voice_setting.voice_id mapping happens inside the engine (covered by
    # the engine unit tests).
    assert record["params"]["system_voice"] == "male-qn-qingse"


def test_run_generation_without_voice_works(ctx):
    record = asyncio.run(run_generation(ctx, "fake", "hello"))
    assert record["voice_id"] is None
    assert record["status"] == "succeeded"
