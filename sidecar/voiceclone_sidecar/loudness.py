"""Loudness: ITU-R BS.1770-4 integrated LUFS + gain normalization.

The blind-comparison feature (issue #10, ADR-0009) only works if every
candidate plays at the SAME perceived loudness — otherwise "louder" wins
every time. Each compared artifact is therefore measured and re-gained to a
shared target before it is presented to the user.

Deliberately dependency-free (same constraint as diagnostics.py): the
sidecar venv has no numpy, and comparison must work before any engine is
installed. Pure-Python biquad filters + block loudness over int16 WAV.

Implementation notes:
- K-weighting = a high-shelf pre-filter (head approximation) cascaded with a
  high-pass (RLB weighting), both as standard BS.1770 biquads at 48 kHz
  coefficients resampled-parameterized for the file's actual rate.
- Integrated loudness uses 400 ms blocks with 75% overlap, the absolute
  gate at -70 LUFS and the relative gate 10 LU below the ungated mean.
- Normalization is a pure gain (no dynamic compression): gain clamps so the
  peak never exceeds full scale — a hot-but-quiet master simply cannot reach
  the target, and the achieved LUFS is reported instead of faked.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

FULL_SCALE = 32767.0
TARGET_LUFS = -16.0  # streaming/podcast convention; comparison target
GATE_ABSOLUTE_LUFS = -70.0
GATE_RELATIVE_LU = 10.0
BLOCK_SECONDS = 0.4
OVERLAP = 0.75  # 75% overlap → hop = 100 ms


class LoudnessError(ValueError):
    """Raised when the audio cannot be measured; message is user-facing."""


def read_wav_mono(path: Path) -> tuple[list[float], int]:
    """Decode a WAV file to mono float samples in [-1, 1] and its rate.

    Supports 8/16/24/32-bit integer PCM, downmixing channels by mean.
    """
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise LoudnessError(f"WAV 文件无法解析：{exc}") from exc
    if width == 0 or width > 4:
        raise LoudnessError(f"不支持的采样位宽：{width * 8}-bit")

    frame_bytes = width * channels
    n = len(raw) // frame_bytes
    samples: list[float] = []
    peak = float(1 << (8 * width - 1))
    for i in range(n):
        total = 0
        base = i * frame_bytes
        for c in range(channels):
            off = base + c * width
            v = int.from_bytes(raw[off:off + width], "little", signed=True)
            total += v
        samples.append(total / channels / peak)
    return samples, rate


def write_wav_mono(path: Path, samples: list[float], rate: int) -> None:
    """Encode mono floats to a 16-bit PCM WAV, hard-clamping to full scale."""
    frames = bytearray()
    for s in samples:
        v = int(round(max(-1.0, min(1.0, s)) * FULL_SCALE))
        frames += struct.pack("<h", v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def _biquad(coeffs: tuple[float, ...], samples: list[float]) -> list[float]:
    b0, b1, b2, a1, a2 = coeffs
    out = [0.0] * len(samples)
    x1 = x2 = y1 = y2 = 0.0
    for i, x in enumerate(samples):
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y
        x2, x1 = x1, x
        y2, y1 = y1, y
    return out


# BS.1770 K-weighting stage coefficients, designed for 48 kHz (b/a as in the
# spec's Annex 1). For other rates we recompute the analog prototype via the
# standard bilinear formulas (Cabrera et al. / common implementations).
def _k_weight_coeffs(rate: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if rate == 48000:
        shelf = (1.53512485958697, -2.69169618940638, 1.19839281085285,
                 -1.69219857431982, 0.732339617801503)
        hp = (1.0, -1.99004745483398, 0.99007225036621,
              -1.98996690010094, 0.990051146410498)
        return shelf, hp

    def db_to_lin(db: float) -> float:
        return 10 ** (db / 20.0)

    # Stage 1: high shelf, f0 = 1681.97 Hz, G = 3.99984 dB, Q = 0.7071752
    f0 = 1681.974450955533
    gain = db_to_lin(3.999843853973347)
    q = 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = gain
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = (
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    )

    # Stage 2: high pass, f0 = 38.13 Hz, Q = 0.5003271
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    hp = (
        1.0,
        -2.0,
        1.0,
        2.0 * (k * k - 1.0) / (1.0 + k / q + k * k),
        (1.0 - k / q + k * k) / (1.0 + k / q + k * k),
    )
    return shelf, hp


def _k_weight(samples: list[float], rate: int) -> list[float]:
    shelf, hp = _k_weight_coeffs(rate)
    return _biquad(hp, _biquad(shelf, samples))


def _blocks(samples: list[float], rate: int) -> list[tuple[int, int]]:
    block = int(BLOCK_SECONDS * rate)
    hop = int(block * (1.0 - OVERLAP))
    if block <= 0 or hop <= 0 or len(samples) < block:
        return []
    return [(start, start + block) for start in range(0, len(samples) - block + 1, hop)]


def _block_lufs(block_samples: list[float]) -> float:
    ms = sum(s * s for s in block_samples) / len(block_samples)
    if ms <= 0:
        return -float("inf")
    return -0.691 + 10.0 * math.log10(ms)


def integrated_lufs(samples: list[float], rate: int) -> float:
    """Integrated loudness (LUFS) with the BS.1770-4 two-stage gating."""
    if not samples:
        raise LoudnessError("音频为空，无法测量响度")
    weighted = _k_weight(samples, rate)
    blocks = [
        _block_lufs(weighted[a:b]) for a, b in _blocks(weighted, rate)
    ]
    blocks = [v for v in blocks if v > GATE_ABSOLUTE_LUFS]
    if not blocks:
        return -float("inf")
    ungated = -0.691 + 10.0 * math.log10(
        sum(10 ** (v / 10.0) for v in blocks) / len(blocks)
    )
    relative_threshold = ungated - GATE_RELATIVE_LU
    gated = [v for v in blocks if v > relative_threshold]
    if not gated:
        return ungated
    return -0.691 + 10.0 * math.log10(
        sum(10 ** (v / 10.0) for v in gated) / len(gated)
    )


def measure_file_lufs(path: Path) -> float:
    samples, rate = read_wav_mono(Path(path))
    return integrated_lufs(samples, rate)


def gain_to_reach(current_lufs: float, target_lufs: float = TARGET_LUFS) -> float:
    """Linear gain factor that moves current LUFS onto the target."""
    return 10 ** ((target_lufs - current_lufs) / 20.0)


def normalize_wav_lufs(
    src: Path, dst: Path, target_lufs: float = TARGET_LUFS
) -> dict:
    """Measure ``src`` and write a gain-adjusted 16-bit mono copy to ``dst``.

    Returns the measurement lineage: original LUFS, applied gain (dB) and the
    achieved LUFS after gain (it can miss the target when the peak would
    clip — the caller surfaces that honestly instead of lying).
    """
    src = Path(src)
    dst = Path(dst)
    samples, rate = read_wav_mono(src)
    current = integrated_lufs(samples, rate)
    if current == -float("inf"):
        # Digital silence: there is nothing to normalize — copy through and
        # report the measurement as unachievable rather than +inf gain.
        write_wav_mono(dst, samples, rate)
        return {
            "original_lufs": None,
            "gain_db": 0.0,
            "achieved_lufs": None,
            "peak_limited": False,
        }
    gain = gain_to_reach(current, target_lufs)
    peak = max(abs(s) for s in samples)
    peak_limited = peak * gain > 1.0
    if peak_limited:
        gain = 1.0 / peak  # never introduce clipping
    gained = [s * gain for s in samples]
    write_wav_mono(dst, gained, rate)
    achieved = integrated_lufs(gained, rate)
    return {
        "original_lufs": round(current, 2),
        "gain_db": round(20.0 * math.log10(gain), 2),
        "achieved_lufs": round(achieved, 2),
        "peak_limited": peak_limited,
    }
