"""Unit tests: BS.1770 loudness measurement + LUFS normalization (issue #10)."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from voiceclone_sidecar.loudness import (
    ANALYSIS_LIMIT_SECONDS,
    LoudnessError,
    TARGET_LUFS,
    integrated_lufs,
    measure_file_lufs,
    normalize_wav_lufs,
    read_wav_mono,
    write_wav_mono,
)


def sine_wav(path: Path, amplitude: float, seconds: float = 2.0, rate: int = 48000,
             freq: float = 997.0) -> None:
    n = int(seconds * rate)
    frames = bytearray()
    for i in range(n):
        t = i / rate
        v = amplitude * math.sin(2 * math.pi * freq * t)
        frames += struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32767))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def test_997hz_sine_matches_reference_level():
    # BS.1770 anchor: a 997 Hz sine at full scale measures -3.01 LUFS; a
    # -20 dBFS (amplitude 0.1) sine therefore measures ≈ -23 LUFS.
    samples = [0.1 * math.sin(2 * math.pi * 997 * i / 48000) for i in range(48000 * 2)]
    lufs = integrated_lufs(samples, 48000)
    assert lufs == pytest.approx(-23.01, abs=0.5)


def test_measurement_is_sample_rate_stable():
    # The same physical signal at 44.1 kHz and 48 kHz must measure the same
    # — comparison fairness cannot depend on an engine's output rate.
    s48 = [0.1 * math.sin(2 * math.pi * 997 * i / 48000) for i in range(48000 * 2)]
    s44 = [0.1 * math.sin(2 * math.pi * 997 * i / 44100) for i in range(44100 * 2)]
    assert integrated_lufs(s44, 44100) == pytest.approx(integrated_lufs(s48, 48000), abs=0.3)


def test_loudness_scaling_is_logarithmic():
    s_lo = [0.05 * math.sin(2 * math.pi * 997 * i / 48000) for i in range(48000)]
    s_hi = [0.5 * math.sin(2 * math.pi * 997 * i / 48000) for i in range(48000)]
    assert integrated_lufs(s_hi, 48000) - integrated_lufs(s_lo, 48000) == pytest.approx(20.0, abs=0.1)


def test_silence_measures_negative_infinity():
    assert integrated_lufs([0.0] * 48000, 48000) == -float("inf")


def test_empty_audio_raises():
    with pytest.raises(LoudnessError):
        integrated_lufs([], 48000)


def test_normalize_equalizes_different_levels(tmp_path: Path):
    quiet = tmp_path / "quiet.wav"
    loud = tmp_path / "loud.wav"
    sine_wav(quiet, 0.02)   # ≈ -37 LUFS
    sine_wav(loud, 0.9)     # ≈ -6.5 LUFS

    out_q = tmp_path / "quiet-norm.wav"
    out_l = tmp_path / "loud-norm.wav"
    mq = normalize_wav_lufs(quiet, out_q)
    ml = normalize_wav_lufs(loud, out_l)

    assert out_q.is_file() and out_l.is_file()
    assert mq["original_lufs"] < ml["original_lufs"]
    assert mq["gain_db"] > 0 and ml["gain_db"] < 0
    # After normalization the two files measure the same loudness — the
    # entire point of issue #10's acceptance criterion.
    a, _ = read_wav_mono(out_q)
    b, _ = read_wav_mono(out_l)
    assert integrated_lufs(a, 48000) == pytest.approx(TARGET_LUFS, abs=0.3)
    assert integrated_lufs(b, 48000) == pytest.approx(TARGET_LUFS, abs=0.3)


def test_normalize_never_clips_a_hot_master(tmp_path: Path):
    # A hot, loud master cannot reach -16 LUFS without clipping; the
    # normalizer must peak-limit honestly and report the miss.
    src = tmp_path / "hot.wav"
    # Sparse full-scale bursts: peak = 1.0 but integrated loudness well below
    # the target — the only shape that needs gain UP and would clip.
    rate = 48000
    samples = [0.0] * (rate * 4)
    burst = int(rate * 0.01)
    for start in range(0, rate * 4, rate):
        for i in range(burst):
            samples[start + i] = 0.9 * math.sin(2 * math.pi * 997 * i / rate)
    write_wav_mono(src, samples, rate)
    out = tmp_path / "hot-norm.wav"
    info = normalize_wav_lufs(src, out)
    assert info["peak_limited"] is True
    assert info["achieved_lufs"] > info["original_lufs"]

    # Output peak must be at full scale, never beyond.
    samples, _ = read_wav_mono(out)
    assert max(abs(s) for s in samples) <= 1.0 + 1e-9


def test_normalize_digital_silence_passes_through(tmp_path: Path):
    src = tmp_path / "silence.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x00" * 48000)
    out = tmp_path / "silence-norm.wav"
    info = normalize_wav_lufs(src, out)
    assert out.is_file()
    assert info["original_lufs"] is None and info["achieved_lufs"] is None


def test_read_wav_rejects_non_wav(tmp_path: Path):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a wav")
    with pytest.raises(LoudnessError):
        read_wav_mono(junk)


# --- 长音频上界（issue #28）-------------------------------------------------


def test_analysis_cap_trims_measurement_not_the_file(tmp_path: Path):
    rate = 8000
    seconds = ANALYSIS_LIMIT_SECONDS + 10  # 130 s
    p = tmp_path / "long.wav"
    _write_wav(p, seconds, rate, 0.1)

    capped, capped_rate = read_wav_mono(p, limit_seconds=ANALYSIS_LIMIT_SECONDS)
    assert capped_rate == rate
    assert len(capped) == int(ANALYSIS_LIMIT_SECONDS * rate)
    # 限窗测量与对前 120 s 手工测量一致
    assert integrated_lufs(capped, capped_rate) == pytest.approx(
        integrated_lufs(read_wav_mono(p)[0][: len(capped)], rate), abs=1e-6
    )
    # 文件本身没有被裁剪
    assert measure_file_lufs(p) == pytest.approx(integrated_lufs(capped, rate), abs=1e-6)


def test_normalize_long_file_preserves_duration_and_loudness(tmp_path: Path):
    rate = 8000
    seconds = ANALYSIS_LIMIT_SECONDS + 5
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    sine_wav(src, 0.02, seconds=seconds, rate=rate, freq=997.0)  # 安静的源
    result = normalize_wav_lufs(src, dst)
    assert result["original_lufs"] is not None
    assert dst.exists()
    full, full_rate = read_wav_mono(dst)
    assert full_rate == rate
    # 整个文件都被重写，不只是被分析的前 120 s
    assert len(full) == int(seconds * rate)
    assert result["achieved_lufs"] == pytest.approx(TARGET_LUFS, abs=0.5)


def _write_wav(path: Path, seconds: float, rate: int, amplitude: float) -> None:
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", int(amplitude * 32767)) * frames)
