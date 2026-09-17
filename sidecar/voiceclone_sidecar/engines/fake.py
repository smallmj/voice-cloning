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
from ..registry import Engine, GenerationRequest, GenerationResult

SAMPLE_RATE = 22050
BASE_FREQ = 220.0


class FakeEngine(Engine):
    engine_id = "fake"
    display_name = "Fake Engine (built-in)"

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir

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
            sample = 0.35 * envelope * math.sin(2 * math.pi * BASE_FREQ * t)
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
