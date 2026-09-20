"""The fake engine: the first end-to-end backend.

It produces a real, playable WAV (a synthesized tone) through the exact same
code path a real engine uses. It fully declares its capabilities.
"""

from __future__ import annotations

import math
import struct
import time
import uuid
import wave
from pathlib import Path

from ..capabilities import Capabilities
from .qwen_tts_cloud import billed_chars
from ..registry import Engine, GenerationRequest, GenerationResult

SAMPLE_RATE = 22050
BASE_FREQ = 220.0


def write_tone_wav(path: Path, duration: float, amplitude: float = 0.35) -> None:
    """Write the fake engine's tone output as a real, playable WAV."""
    frames = bytearray()
    for i in range(int(duration * SAMPLE_RATE)):
        t = i / SAMPLE_RATE
        envelope = min(1.0, t * 8.0, max(0.0, duration - t) * 8.0)
        sample = amplitude * envelope * math.sin(2 * math.pi * BASE_FREQ * t)
        frames += struct.pack("<h", int(sample * 32767))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(frames))


class FakeEngine(Engine):
    engine_id = "fake"
    display_name = "Fake Engine (built-in)"

    def __init__(self, output_dir: Path | None = None, amplitude: float = 0.35) -> None:
        self.output_dir = output_dir
        # Output-level knob. Real engines disagree wildly on loudness; the
        # level-skewed test seams below set this to verify the comparison
        # feature's LUFS normalization end to end (issue #10).
        self.amplitude = amplitude

    def capabilities(self) -> Capabilities:
        return Capabilities(
            languages=("zh", "en"),
            voice_cloning=True,
            voice_design=True,
            pronunciation_control=False,
            emotion=True,
            commercial_license=True,
            cross_device_use=True,
            upload_used_for_training=False,
            api_closed_loop=True,
            # Small on purpose: the contract tests exercise multi-segment
            # long-text jobs without generating huge strings (issue #13).
            max_chars_per_request=100,
        )

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        started = time.monotonic()
        log(f"fake: received generation {request.generation_id}, {len(request.text)} chars")
        duration = max(0.5, min(5.0, len(request.text) * 0.05))
        log(f"fake: synthesizing {duration:.2f}s of audio at {SAMPLE_RATE} Hz")

        out_dir = self.output_dir or Path.cwd() / "data" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav"
        write_tone_wav(out_path, duration, self.amplitude)
        elapsed = time.monotonic() - started
        log(f"fake: wrote {out_path.name} in {elapsed:.3f}s")
        return GenerationResult(
            audio_path=str(out_path),
            sample_rate=SAMPLE_RATE,
            model_version="fake-1.0",
            cost=0.0,  # local synthesis has no per-run cost
        )

    def design_voice(self, description: str, preview_text: str, log) -> dict:
        """Fake voice design (issue #11): the described "voice" is a fixed
        tone. Returns the preview sample so the sidecar can attach it as the
        designed voice's reference — the same flow a real design engine runs."""
        log(f"fake: designing voice from description ({len(description)} chars)")
        out_dir = self.output_dir or Path.cwd() / "data" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_path = out_dir / f"design-{uuid.uuid4().hex}.wav"
        write_tone_wav(sample_path, 2.0)
        voice_id = f"fake-voice-{uuid.uuid4().hex[:8]}"
        log(f"fake: designed voice {voice_id} ready")
        return {
            "voice_id": voice_id,
            "sample_audio_path": str(sample_path),
            "transcript": preview_text,
        }

    def transcribe(self, audio_path: str, log) -> str:
        """Cloud-transcription surface: any registered engine may expose this.

        The fake returns a deterministic transcript so the transcription
        contract (providers, stored transcripts, ref_text auto-fill) is fully
        testable and demonstrable without a real ASR provider.
        """
        log(f"fake: transcribing {Path(audio_path).name}")
        return "这是一段用于测试的转写文本。"


class FakeRefTextEngine(FakeEngine):
    """Test seam: the fake engine, but declaring it needs reference text.

    Registered only when VOICECLONE_TEST_ENGINES=1 (the contract-test
    harness); it exercises the ref_text auto-fill path end to end.
    """

    engine_id = "fake-ref-text"
    display_name = "Fake Engine (requires ref text)"

    def capabilities(self) -> Capabilities:
        caps = super().capabilities()
        return Capabilities(**{**caps.to_dict(), "requires_reference_text": True})


class FakeKeyEngine(FakeEngine):
    """Test seam: a BYOK cloud-shaped engine.

    Registered only when VOICECLONE_TEST_ENGINES=1. It exercises the whole
    issue-#9 surface without touching any real vendor: key requirements and
    the /settings/keys contract, billing/data-usage disclosure, parameter
    specs, reference binding that mints a voice_id, and the voice_id
    injection into generation params.
    """

    engine_id = "fake-key"
    display_name = "Fake Engine (BYOK cloud)"
    requires_key = True
    billing_note = "fake：0.8 元 / 万字符（1 个汉字计 2 个字符），输出不计费"
    data_usage_note = "fake：上传内容不会用于训练（测试声明）"

    def capabilities(self) -> Capabilities:
        caps = super().capabilities()
        return Capabilities(**{**caps.to_dict(), "requires_reference_text": False})

    def param_specs(self):
        from ..capabilities import AppliesTo, ParamSpec

        return [
            ParamSpec(
                name="fake_mode",
                label="测试模式",
                kind="select",
                default="auto",
                choices=("auto", "fast"),
                applies_to=AppliesTo(engine=self.engine_id, model="fake-1.0"),
                help="仅用于测试的参数",
            )
        ]

    def bind_reference(self, ref_path, ref_text: str | None, log) -> dict:
        log("fake-key: enrolling reference")
        return {"voice_id": f"fake-voice-{uuid.uuid4().hex[:8]}"}

    def check_voice(self, voice_id: str, log) -> bool:
        log(f"fake-key: probing voice {voice_id}")
        return True

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        voice_id = request.params.get("voice_id")
        if not voice_id:
            raise RuntimeError("fake-key requires a bound voice_id")
        result = super().synthesize(request, log)
        return GenerationResult(
            audio_path=result.audio_path,
            sample_rate=result.sample_rate,
            model_version=voice_id,
            cost=round(billed_chars(request.text) / 10000 * 0.8, 4),
        )


class FakeLoudEngine(FakeEngine):
    """Test seam (issue #10): emits a HOT master (near full scale)."""

    engine_id = "fake-loud"
    display_name = "Fake Engine (loud)"

    def __init__(self, output_dir: Path | None = None) -> None:
        super().__init__(output_dir, amplitude=0.95)


class FakeQuietEngine(FakeEngine):
    """Test seam (issue #10): emits a very quiet master.

    Together with FakeLoudEngine the pair differs by roughly 30 dB of output
    level — exactly the "electric level differs greatly" engines the issue
    requires for proving the LUFS normalization actually equalizes.
    """

    engine_id = "fake-quiet"
    display_name = "Fake Engine (quiet)"

    def __init__(self, output_dir: Path | None = None) -> None:
        super().__init__(output_dir, amplitude=0.02)


class FakeFailingDesignEngine(FakeEngine):
    """Test seam (issue #11): design_voice always fails with a non-cloud
    error, proving the sidecar leaves no orphaned reference-less voice
    behind even when the failure is not a CloudEngineError."""

    engine_id = "fake-failing-design"
    display_name = "Fake Engine (design always fails)"

    def design_voice(self, description: str, preview_text: str, log) -> dict:
        log("fake-failing-design: about to fail")
        raise OSError("disk exploded")


class FakeVanishingVoiceEngine(FakeKeyEngine):
    """Test seam (issue #12): the vendor silently deletes each enrolled
    voice after its first successful synthesis — exactly the "回收音色"
    behavior the health check must survive. Generation N+1 must therefore
    rebuild the binding from the local reference without user intervention."""

    engine_id = "fake-vanishing-voice"
    display_name = "Fake Engine (cloud voice silently recycled)"

    def __init__(self, output_dir: Path | None = None) -> None:
        super().__init__(output_dir)
        self.deleted: set[str] = set()

    def check_voice(self, voice_id: str, log) -> bool:
        log(f"fake-vanishing-voice: probing voice {voice_id}")
        return voice_id not in self.deleted

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        voice_id = request.params.get("voice_id")
        result = super().synthesize(request, log)
        # Simulate the vendor's GC: the voice existed for this run, then it
        # is silently gone.
        self.deleted.add(voice_id)
        return result


class FakeBrokenRebuildEngine(FakeKeyEngine):
    """Test seam (issue #12): every bound voice probes as dead and the
    re-enrollment itself fails — the sidecar must mark THIS binding
    unavailable with a clear reason while leaving every other binding
    of the same voice untouched."""

    engine_id = "fake-broken-rebuild"
    display_name = "Fake Engine (rebuild always fails)"

    def __init__(self, output_dir: Path | None = None) -> None:
        super().__init__(output_dir)
        # The first enrollment succeeds (so a binding exists at all, like it
        # would for a user whose voice worked once); every rebuild after that
        # fails at the vendor.
        self.rebuild_attempted = False

    def check_voice(self, voice_id: str, log) -> bool:
        log(f"fake-broken-rebuild: probing voice {voice_id}")
        return not self.rebuild_attempted

    def bind_reference(self, ref_path, ref_text: str | None, log) -> dict:
        if self.rebuild_attempted:
            log("fake-broken-rebuild: rebuild attempted")
            from .cloud_base import CloudEngineError

            raise CloudEngineError("模拟厂商拒绝重建：参考音频校验未通过（HTTP 400）")
        self.rebuild_attempted = True
        log("fake-broken-rebuild: enrolling reference")
        return {"voice_id": f"fake-voice-{uuid.uuid4().hex[:8]}"}


class FakeSlowEngine(FakeEngine):
    """Test seam (issue #13): every synthesis sleeps, so a job's segments
    take long enough to observe the queue — cancel-while-running and
    cancel-while-queued are testable without racing a millisecond engine."""

    engine_id = "fake-slow"
    display_name = "Fake Engine (slow)"

    def __init__(self, output_dir: Path | None = None, sleep_seconds: float = 2.0) -> None:
        super().__init__(output_dir)
        self.sleep_seconds = sleep_seconds

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        log(f"fake-slow: sleeping {self.sleep_seconds}s")
        time.sleep(self.sleep_seconds)
        return super().synthesize(request, log)


class FakeBrokenEngine(FakeEngine):
    """Test seam (issue #19): synthesis always fails with an exception whose
    message embeds a filesystem path. Verifies the generation-failure 500
    detail is sanitized instead of echoing the raw exception text."""

    engine_id = "fake-broken"
    display_name = "Fake Engine (broken)"

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        log("fake-broken: simulating hard failure")
        raise RuntimeError("/tmp/voiceclone/secret/model.bin exploded: boom")


class FakeRateLimitedEngine(FakeKeyEngine):
    """Test seam (issue #13): the vendor throttles the first N synthesis
    calls with a rate-limit error, then lets everything through. The
    sidecar must retry with backoff and succeed — the throttle never
    surfaces as a user-facing generation failure."""

    engine_id = "fake-rate-limited"
    display_name = "Fake Engine (rate limited at first)"

    def __init__(self, output_dir: Path | None = None, throttled_calls: int = 2) -> None:
        super().__init__(output_dir)
        self.throttled_calls = throttled_calls
        self.calls = 0

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        self.calls += 1
        if self.calls <= self.throttled_calls:
            from .cloud_base import CloudEngineError

            log(f"fake-rate-limited: call {self.calls} throttled")
            raise CloudEngineError(
                "云端合成失败：Throttling：Requests rate limit exceeded, please try again later. (HTTP 429)"
            )
        return super().synthesize(request, log)
