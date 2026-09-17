"""Unit tests for the read-only reference-sample diagnostics."""

from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path

import pytest

from voiceclone_sidecar.diagnostics import DiagnosticError, analyze_audio

RATE = 16000


def write_wav(path: Path, samples: list[float], rate: int = RATE) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(
            b"".join(struct.pack("<h", max(-32767, min(32767, int(s * 32767)))) for s in samples)
        )


def sine(freq: float, seconds: float, amp: float = 0.3, rate: int = RATE) -> list[float]:
    n = int(seconds * rate)
    return [amp * math.sin(2 * math.pi * freq * i / rate) for i in range(n)]


def silence(seconds: float, amp: float = 0.0, rate: int = RATE) -> list[float]:
    n = int(seconds * rate)
    rng = random.Random(7)
    return [amp * rng.uniform(-1, 1) for _ in range(n)]


def by_id(analysis: dict) -> dict:
    return {d["id"]: d for d in analysis["diagnostics"]}


def test_clean_recording_is_all_good(tmp_path):
    samples = silence(0.3) + sine(220, 3.0) + silence(0.2)
    path = tmp_path / "ref.wav"
    write_wav(path, samples)
    result = analyze_audio(path)
    d = by_id(result)
    assert d["snr"]["status"] == "good"
    assert d["clipping"]["status"] == "good"
    assert d["speaker"]["status"] == "good"
    # A clean file can still have a short leading silence — that's expected
    # and only surfaces as a fact, not a failure.


def test_silence_segments_are_reported_with_advice(tmp_path):
    samples = silence(1.5) + sine(220, 3.0) + silence(1.2) + sine(220, 2.0)
    path = tmp_path / "ref.wav"
    write_wav(path, samples)
    d = by_id(analyze_audio(path))
    assert d["silence"]["status"] == "warn"
    assert len(d["silence"]["segments"]) == 2
    assert d["silence"]["advice"]  # every diagnosis carries actionable advice


def test_clipping_is_detected(tmp_path):
    samples = silence(0.3) + [1.0 if math.sin(2 * math.pi * 200 * i / RATE) >= 0 else -1.0
                              for i in range(int(2.0 * RATE))]
    path = tmp_path / "ref.wav"
    write_wav(path, samples)
    d = by_id(analyze_audio(path))
    assert d["clipping"]["status"] == "bad"
    assert d["clipping"]["value"] > 0
    assert "增益" in d["clipping"]["advice"]


def test_noisy_recording_gets_low_snr(tmp_path):
    rng = random.Random(42)
    noisy = [s + 0.2 * rng.uniform(-1, 1) for s in sine(220, 3.0)]
    path = tmp_path / "ref.wav"
    write_wav(path, noisy)
    d = by_id(analyze_audio(path))
    assert d["snr"]["status"] in ("warn", "bad")
    assert d["snr"]["value"] < 20
    assert "安静" in d["snr"]["advice"]


def test_two_pitch_regions_look_like_multiple_speakers(tmp_path):
    # Alternating 220 Hz / 330 Hz tone "speakers" — the heuristic must raise
    # a suspicion. Deliberately a warn, never a hard "bad": it is a heuristic.
    samples = silence(0.2) + sine(220, 2.0) + silence(0.2) + sine(330, 2.0) + silence(0.2)
    path = tmp_path / "ref.wav"
    write_wav(path, samples)
    d = by_id(analyze_audio(path))
    assert d["speaker"]["status"] == "warn"
    assert d["speaker"]["value"] == 2
    assert "疑似" in d["speaker"]["message"]


def test_analysis_never_modifies_the_file(tmp_path):
    samples = silence(0.3) + sine(220, 2.0) + silence(0.2)
    path = tmp_path / "ref.wav"
    write_wav(path, samples)
    before = path.read_bytes()
    analyze_audio(path)
    assert path.read_bytes() == before


def test_broken_wav_raises_user_facing_error(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFFbroken")
    with pytest.raises(DiagnosticError):
        analyze_audio(path)


def test_non_wav_without_ffmpeg_raises_explaining_error(tmp_path, monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    path = tmp_path / "ref.mp3"
    path.write_bytes(b"\x00" * 128)
    with pytest.raises(DiagnosticError, match="ffmpeg"):
        analyze_audio(path)


def test_every_diagnosis_carries_advice(tmp_path):
    samples = silence(1.0) + sine(220, 2.0) + silence(1.0)
    path = tmp_path / "ref.wav"
    write_wav(path, samples)
    for d in analyze_audio(path)["diagnostics"]:
        assert d["advice"].strip()
        assert d["status"] in ("good", "warn", "bad")
        assert d["message"].strip()
