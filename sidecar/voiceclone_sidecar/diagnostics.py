"""Reference-sample diagnostics: read-only signal analysis + advice.

The app NEVER modifies the user's material (issue #8). Everything here reads
the audio and reports facts with one actionable piece of advice per finding;
any repair (re-record, trim, re-gain) is the user's action, not ours.

Analyses (all heuristic, deliberately dependency-free — the sidecar venv has
no numpy, and diagnostics must work before any engine is installed):

- SNR: 10th-percentile frame energy (noise floor) vs 90th-percentile frame
  energy (speech level), over 40ms/20ms frames.
- Silence: contiguous runs below an adaptive threshold (noise floor + 10 dB,
  capped at speech level − 35 dB).
- Clipping: fraction of samples at/near full scale plus repeated-extreme runs.
- Speaker count: sparse autocorrelation pitch sampling over voiced frames;
  a bimodal F0 distribution (two clusters ≥20% each, medians ≥1.25× apart)
  is reported as a SUSPICION, never as a fact.

Non-WAV formats are decoded by shelling out to ffmpeg into a TEMP file; the
user's original file is only ever read.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

TARGET_RATE = 16000
FRAME_WIN = int(0.04 * TARGET_RATE)  # 40ms analysis window
FRAME_HOP = int(0.02 * TARGET_RATE)  # 20ms hop
MIN_SILENCE_SECONDS = 0.5
FULL_SCALE = 32767.0
CLIP_WARN_RATIO = 1e-4
CLIP_BAD_RATIO = 1e-3
SNR_GOOD_DB = 20.0
SNR_WARN_DB = 12.0
PITCH_MIN_HZ = 80.0
PITCH_MAX_HZ = 400.0


class DiagnosticError(ValueError):
    """Raised when the audio cannot be analyzed; message is user-facing."""


def _decode_wav_stdlib(path: Path) -> tuple[list[float], int]:
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            n_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            if sampwidth != 2:
                raise DiagnosticError(
                    f"诊断只支持 16-bit PCM WAV（该文件是 {sampwidth * 8}-bit）；"
                    "请先用录音软件另存为 16-bit WAV"
                )
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise DiagnosticError(f"WAV 文件无法解析：{exc}") from exc
    n = len(raw) // (2 * n_channels)
    samples: list[float] = []
    step = n_channels
    for i in range(0, n * n_channels, step):
        total = 0
        for c in range(n_channels):
            off = (i + c) * 2
            v = int.from_bytes(raw[off:off + 2], "little", signed=True)
            total += v
        samples.append(total / n_channels / FULL_SCALE)
    return samples, rate


def _resample(samples: list[float], rate: int) -> list[float]:
    if rate == TARGET_RATE:
        return samples
    if rate < TARGET_RATE:
        return samples  # already below target: analyze at native rate
    factor = rate // TARGET_RATE or 1
    if factor <= 1:
        return samples
    out: list[float] = []
    for i in range(0, len(samples) - factor + 1, factor):
        out.append(sum(samples[i:i + factor]) / factor)  # box-average = cheap anti-alias
    return out


def _decode_any(path: Path) -> tuple[list[float], int]:
    """WAV through the stdlib; everything else via ffmpeg into a temp file."""
    if path.suffix.lower() == ".wav":
        return _decode_wav_stdlib(path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise DiagnosticError(
            f"诊断 {path.suffix[1:].upper()} 音频需要系统安装 ffmpeg；"
            "请上传 WAV 文件或安装 ffmpeg 后重试"
        )
    with tempfile.TemporaryDirectory(prefix="vc-diag-") as td:
        tmp = Path(td) / "decode.wav"
        try:
            subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", str(path),
                 "-ac", "1", "-ar", str(TARGET_RATE), "-c:a", "pcm_s16le", str(tmp)],
                capture_output=True, timeout=120, check=True,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise DiagnosticError(f"音频解码失败：{exc}") from exc
        return _decode_wav_stdlib(tmp)


def _frame_db(samples: list[float]) -> list[float]:
    """RMS in dBFS per analysis frame."""
    out = []
    peak = 10.0 ** (-100 / 20)
    for start in range(0, len(samples) - FRAME_WIN + 1, FRAME_HOP):
        acc = 0.0
        for v in samples[start:start + FRAME_WIN]:
            acc += v * v
        rms = math.sqrt(acc / FRAME_WIN)
        out.append(max(20 * math.log10(max(rms, 1e-9)), -100.0))
    return out


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return -100.0
    idx = min(len(sorted_vals) - 1, max(0, int(p * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def _f0_of_frame(frame: list[float], rate: int) -> float | None:
    """Autocorrelation pitch of one frame; None when unvoiced/uncertain."""
    n = len(frame)
    r0 = sum(v * v for v in frame)
    if r0 <= 0:
        return None
    lag_min = max(2, int(rate / PITCH_MAX_HZ))
    lag_max = min(n - 1, int(rate / PITCH_MIN_HZ))
    best_lag, best = 0, 0.0
    for lag in range(lag_min, lag_max + 1):
        acc = 0.0
        for i in range(n - lag):
            acc += frame[i] * frame[i + lag]
        if acc > best:
            best, best_lag = acc, lag
    if best_lag == 0 or best / r0 < 0.5:
        return None
    return rate / best_lag


def _bimodal_split(f0s: list[float]) -> list[list[float]] | None:
    """Two-cluster split of the F0 list, or None when unimodal."""
    vals = sorted(f0s)
    n = len(vals)
    best: tuple[float, list[list[float]]] | None = None
    for cut in range(max(1, n // 5), n - max(1, n // 5)):
        low, high = vals[:cut], vals[cut:]
        if len(low) < 0.2 * n or len(high) < 0.2 * n:
            continue
        med_low = low[len(low) // 2]
        med_high = high[len(high) // 2]
        if med_high <= 0 or med_low <= 0:
            continue
        ratio = max(med_high / med_low, med_low / med_high)
        if ratio >= 1.25 and (best is None or ratio > best[0]):
            best = (ratio, [low, high])
    return best[1] if best else None


def analyze_audio(path: Path, analysis_limit_seconds: float = 120.0) -> dict:
    """Analyze one audio file. Purely read-only; returns facts + diagnostics."""
    samples, rate = _decode_any(path)
    # Cap the analysis window: diagnostics must stay fast on long files, and
    # the cap only trims the *analysis*, never the file itself.
    limit = int(analysis_limit_seconds * rate)
    if len(samples) > limit:
        samples = samples[:limit]
    duration = len(samples) / rate
    frames = _frame_db(samples)
    if not frames:
        raise DiagnosticError("音频过短，无法诊断")
    ordered = sorted(frames)
    noise_db = _percentile(ordered, 0.10)
    speech_db = _percentile(ordered, 0.90)
    snr_db = round(speech_db - noise_db, 1)

    # -- silence -------------------------------------------------------------
    silence_threshold = max(noise_db + 10.0, speech_db - 40.0)
    segments: list[tuple[float, float]] = []
    run_start: int | None = None
    for i, db in enumerate(frames):
        if db < silence_threshold:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            length = (i - run_start) * FRAME_HOP / rate
            if length >= MIN_SILENCE_SECONDS:
                segments.append((run_start * FRAME_HOP / rate,
                                 run_start * FRAME_HOP / rate + length))
            run_start = None
    if run_start is not None:
        length = (len(frames) - run_start) * FRAME_HOP / rate
        if length >= MIN_SILENCE_SECONDS:
            segments.append((run_start * FRAME_HOP / rate,
                             run_start * FRAME_HOP / rate + length))
    silence_total = sum(e - s for s, e in segments)
    edge_silence = 0.0
    if segments:
        if segments[0][0] <= 0.05:
            edge_silence += segments[0][1] - segments[0][0]
        if segments[-1][1] >= duration - 0.05:
            edge_silence += segments[-1][1] - segments[-1][0]

    # -- clipping ------------------------------------------------------------
    clip_count = 0
    clip_run = 0
    prev_extreme = False
    for v in samples:
        av = abs(v)
        if av >= 0.995:
            clip_count += 1
            # three or more consecutive samples pinned at full scale is the
            # tell-tale flat top of a clipped waveform
            if prev_extreme:
                clip_run += 1
                if clip_run >= 2:
                    clip_count += 1
            else:
                prev_extreme, clip_run = True, 0
        else:
            prev_extreme, clip_run = False, 0
    clip_ratio = clip_count / max(1, len(samples))

    # -- speaker count (heuristic) --------------------------------------------
    f0s: list[float] = []
    pitch_hop = FRAME_HOP * 5  # one pitch frame per 100ms
    for start in range(0, len(samples) - FRAME_WIN + 1, pitch_hop):
        db = frames[min(len(frames) - 1, start // FRAME_HOP)]
        if db < speech_db - 15:
            continue
        f0 = _f0_of_frame(samples[start:start + FRAME_WIN], rate)
        if f0 is not None:
            f0s.append(f0)
    clusters = _bimodal_split(f0s) if len(f0s) >= 10 else None

    diagnostics: list[dict] = []

    if snr_db >= SNR_GOOD_DB:
        diagnostics.append({
            "id": "snr", "status": "good", "value": snr_db,
            "message": f"信噪比约 {snr_db:g} dB，环境噪声很低。",
            "advice": "无需处理，可以直接用于声音复刻。",
        })
    else:
        severity = "warn" if snr_db >= SNR_WARN_DB else "bad"
        diagnostics.append({
            "id": "snr", "status": severity, "value": snr_db,
            "message": f"信噪比约 {snr_db:g} dB，背景噪声"
                       f"{'较明显' if severity == 'warn' else '严重'}。",
            "advice": "请在更安静的环境重新录制参考音频，或关闭风扇/空调等噪声源；"
                      "复刻质量对噪声非常敏感。",
        })

    if clip_ratio == 0:
        diagnostics.append({
            "id": "clipping", "status": "good", "value": 0.0,
            "message": "未检测到削波。",
            "advice": "无需处理。",
        })
    else:
        severity = "bad" if clip_ratio >= CLIP_BAD_RATIO else "warn"
        diagnostics.append({
            "id": "clipping", "status": severity, "value": round(clip_ratio, 6),
            "message": f"检测到削波（约 {clip_ratio * 100:.2f}% 采样顶到满幅）。",
            "advice": "录音增益过大导致波形削顶：请降低输入增益重新录制，"
                      "让最大音量保持在满幅的 80% 左右。",
        })

    if not segments:
        diagnostics.append({
            "id": "silence", "status": "good", "value": 0.0,
            "message": "没有超过 0.5 秒的静音段。",
            "advice": "无需处理。",
        })
    else:
        diagnostics.append({
            "id": "silence", "status": "warn",
            "value": len(segments),
            "segments": [[round(s, 2), round(e, 2)] for s, e in segments],
            "message": f"检测到 {len(segments)} 段静音（共 {silence_total:.1f} 秒）。",
            "advice": "过长的静音会稀释复刻效果。应用不会修改你的素材——请自行裁剪出"
                      "一段连续说话的录音，再重新上传。",
        })

    if len(f0s) < 10:
        diagnostics.append({
            "id": "speaker", "status": "warn", "value": None,
            "message": "有效语音帧太少，无法判断是否为单一说话人。",
            "advice": "请提供一段说话更连续的录音（少停顿、单人从头说到尾）。",
        })
    elif clusters is not None:
        diagnostics.append({
            "id": "speaker", "status": "warn", "value": 2,
            "message": "音高分布呈双峰，疑似包含多个说话人（启发式判断，非结论）。",
            "advice": "请使用单人连续说话的录音；多人对话或拼接素材会让复刻结果混淆音色。",
        })
    else:
        diagnostics.append({
            "id": "speaker", "status": "good", "value": 1,
            "message": "未发现多说话人迹象（启发式判断，非结论）。",
            "advice": "无需处理。",
        })

    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": rate,
        "snr_db": snr_db,
        "silence_segments": [[round(s, 2), round(e, 2)] for s, e in segments],
        "clipped_sample_ratio": round(clip_ratio, 6),
        "voiced_pitch_samples": len(f0s),
        "diagnostics": diagnostics,
    }
