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
    import io
    import wave as _wave

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
        return httpx.Response(
            200,
            json={
                "data": {"audio": wav_bytes().hex()},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            },
        )

    audio_dir = tmp_path / "audio-mm"
    audio_dir.mkdir(exist_ok=True)
    registry = Registry()
    store = KeyStore(backend=MemoryBackend())
    store.set("minimax-speech-cloud", "mk-test")
    registry.register(
        MiniMaxCloudEngine(
            output_dir=audio_dir,
            key_store=store,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )
    # keys=store is REQUIRED for hermeticity: without it the ctx defaults to
    # the real Keychain KeyStore, so this test only passed on machines where
    # the developer's own MiniMax key happened to be stored (CI caught it).
    return AppContext(
        registry=registry,
        token="t",
        audio_dir=audio_dir,
        data_root=tmp_path,
        keys=store,
    )


def test_cloud_engine_without_voice_or_system_voice_is_422(tmp_path):
    ctx = _minimax_ctx(tmp_path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_generation(ctx, "minimax-speech-cloud", "你好"))
    assert exc.value.status_code == 422
    assert "系统音色" in exc.value.detail


def test_cloud_engine_system_voice_generates_without_voice_record(tmp_path):
    ctx = _minimax_ctx(tmp_path)
    record = asyncio.run(
        run_generation(
            ctx,
            "minimax-speech-cloud",
            "你好",
            params={"system_voice": "male-qn-qingse"},
        )
    )
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


# --- placeholder transcripts must never reach an engine as ref_text ----------


class _CapturingRefTextEngine(FakeEngine):
    """Records the params it receives; requires reference text."""

    engine_id = "fake-capture-ref-text"

    def __init__(self, output_dir=None):
        super().__init__(output_dir=output_dir)
        self.seen_params = []

    def capabilities(self):
        from voiceclone_sidecar.capabilities import Capabilities

        caps = super().capabilities()
        return Capabilities(**{**caps.to_dict(), "requires_reference_text": True})

    def synthesize(self, request, log):
        self.seen_params.append(dict(request.params))
        return super().synthesize(request, log)


def _voice_with_placeholder_transcript(ctx):
    import wave as _wave

    wav = ctx.data_root / "ref.wav"
    with _wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000 * 3)  # 3s of silence: validation passes
    record = ctx.voice_store.create("占位音色", "回归测试", wav)
    # Simulate the legacy fake engine having written its fixed placeholder
    # into the voice library (issue #18 bad data).
    ctx.voice_store.set_transcript(record["id"], "这是一段用于测试的转写文本。")
    return record["id"]


def test_placeholder_transcript_is_not_used_as_ref_text(ctx):
    """The issue-#18 placeholder must not ride along as ref_text: the
    ref-text-conditioned engine gets a real transcript instead (and the
    placeholder on the voice record is overwritten by the re-transcription)."""
    voice_id = _voice_with_placeholder_transcript(ctx)
    engine = _CapturingRefTextEngine(output_dir=ctx.audio_dir)
    ctx.registry.register(engine)

    real = "今天我们要讲的是，为什么日本从一开始就注定必败。"
    calls = []

    def fake_transcribe(voice_id_, provider):
        calls.append(voice_id_)
        return real

    ctx.transcribe_voice_sync = fake_transcribe  # type: ignore[method-assign]

    asyncio.run(run_generation(ctx, engine.engine_id, "你好世界", voice_id=voice_id))

    assert calls == [voice_id]  # re-transcription actually ran
    assert engine.seen_params[0]["ref_text"] == real
    # The re-transcription replaces the placeholder on the voice record.
    assert ctx.voice_store.get(voice_id)["reference"]["transcript"] == real


def test_placeholder_transcript_not_passed_to_engines_that_tolerate_text(ctx):
    """Engages the elif branch: a non-ref-text engine receives NO ref_text
    derived from a placeholder transcript, and does not transcribe on
    demand."""
    voice_id = _voice_with_placeholder_transcript(ctx)
    engine = _CapturingRefTextEngine(output_dir=ctx.audio_dir)
    # Flip the capability to "doesn't require ref text" to reach the elif.
    import voiceclone_sidecar.engines.fake as fake_mod

    class _Tolerant(_CapturingRefTextEngine):
        engine_id = "fake-tolerant"

        def capabilities(self):
            from voiceclone_sidecar.capabilities import Capabilities

            caps = fake_mod.FakeEngine.capabilities(self)
            return Capabilities(**{**caps.to_dict(), "requires_reference_text": False})

    engine = _Tolerant(output_dir=ctx.audio_dir)
    ctx.registry.register(engine)

    def boom(voice_id_, provider):  # pragma: no cover - must not be called
        raise AssertionError("on-demand transcription must not run")

    ctx.transcribe_voice_sync = boom  # type: ignore[method-assign]

    asyncio.run(run_generation(ctx, engine.engine_id, "你好世界", voice_id=voice_id))
    assert "ref_text" not in engine.seen_params[0]


def test_real_stored_transcript_is_reused_without_retranscribing(ctx):
    """The original contract (issue #8) still holds for a REAL transcript:
    it is reused as-is, even when no transcription provider is available."""
    voice_id = _voice_with_placeholder_transcript(ctx)
    real = "今天我们要讲的是，为什么日本从一开始就注定必败。"
    ctx.voice_store.set_transcript(voice_id, real)  # overwrite placeholder
    engine = _CapturingRefTextEngine(output_dir=ctx.audio_dir)
    ctx.registry.register(engine)

    def boom(voice_id_, provider):  # pragma: no cover - must not be called
        raise AssertionError("re-transcription must not run")

    ctx.transcribe_voice_sync = boom  # type: ignore[method-assign]

    record = asyncio.run(run_generation(ctx, engine.engine_id, "你好世界", voice_id=voice_id))
    assert record["status"] == "succeeded"
    assert engine.seen_params[0]["ref_text"] == real
