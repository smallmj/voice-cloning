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
        )

    def synthesize(self, request: GenerationRequest, log) -> GenerationResult:
        started = time.monotonic()
        log(f"fake: received generation {request.generation_id}, {len(request.text)} chars")
        duration = max(0.5, min(5.0, len(request.text) * 0.05))
        log(f"fake: synthesizing {duration:.2f}s of audio at {SAMPLE_RATE} Hz")

        frames = bytearray()
        for i in range(int(duration * SAMPLE_RATE)):
            t = i / SAMPLE_RATE
            envelope = min(1.0, t * 8.0, max(0.0, duration - t) * 8.0)
            sample = self.amplitude * envelope * math.sin(2 * math.pi * BASE_FREQ * t)
            frames += struct.pack("<h", int(sample * 32767))

        out_dir = self.output_dir or Path.cwd() / "data" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{request.generation_id or uuid.uuid4().hex}.wav"
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(bytes(frames))

        elapsed = time.monotonic() - started
        log(f"fake: wrote {out_path.name} in {elapsed:.3f}s")
        return GenerationResult(
            audio_path=str(out_path),
            sample_rate=SAMPLE_RATE,
            model_version="fake-1.0",
            cost=0.0,  # local synthesis has no per-run cost
        )

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
        from ..capabilities import ParamSpec

        return [
            ParamSpec(
                name="fake_mode",
                label="测试模式",
                kind="select",
                default="auto",
                choices=("auto", "fast"),
                help="仅用于测试的参数",
            )
        ]

    def bind_reference(self, ref_path, ref_text: str | None, log) -> dict:
        log("fake-key: enrolling reference")
        return {"voice_id": f"fake-voice-{uuid.uuid4().hex[:8]}"}

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
