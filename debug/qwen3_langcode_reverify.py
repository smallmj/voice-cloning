"""Issue #68 lang_code re-verification on the REAL machine (MPS/MLX, Apple Silicon).

Prior V3 verification (2026-09-22) concluded mlx-audio never consumes
lang_code. That conclusion came from reading the WRONG module at some point
(VyvoTTS qwen3/qwen3.py). The INSTALLED mlx-audio 0.5.4 module
mlx_audio/tts/models/qwen3_tts/qwen3_tts.py DOES consume it:

- L1297: generate() passes language=lang_code into _prepare_generation_inputs
- L393-406 + L739-741: language_id = config.codec_language_id[language.lower()]
  (only when language != "auto"), which becomes the codec prefill language
  token — the weight config maps {'chinese': 2055, 'english': 2050, ...}

This script synthesizes the SAME text three ways — auto (no lang_code),
lang_code="chinese", lang_code="english" — and compares:
  1. audio bytes (auto vs chinese should differ only by sampling noise;
     chinese vs english changes the forced language token path)
  2. duration / waveform maxabs difference
  3. optionally whisper-small ASR transcription if available

Run with the ENGINE venv python:
  "$HOME/Library/Application Support/VoiceClone/runtime/engines/qwen3-tts-mlx/venv/bin/python" \
      debug/qwen3_langcode_reverify.py
"""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

WEIGHTS = Path.home() / "Library/Application Support/VoiceClone/runtime/engines/qwen3-tts-mlx/weights"
OUT = Path(__file__).parent / "qwen3-verify-issue68"
TEXT = "Hello world, this is a language test. 今天天气真好，我们出去散步吧。"

CASES = [
    ("auto", None),
    ("chinese", "chinese"),
    ("english", "english"),
]


def read_wav(path: Path):
    with wave.open(str(path)) as w:
        n = w.getnframes()
        data = w.readframes(n)
    return n, data


def main() -> None:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    from mlx_audio.tts.utils import load_model

    OUT.mkdir(exist_ok=True)
    model = load_model(str(WEIGHTS))
    print(f"model loaded from {WEIGHTS}", flush=True)

    results = []
    for case, lang in CASES:
        out_path = OUT / f"{case}.wav"
        started = time.monotonic()
        kwargs = {"lang_code": lang} if lang else {}
        chunks = list(model.generate(text=TEXT, verbose=False, **kwargs))
        audio = np.concatenate([np.asarray(c.audio, dtype=np.float32) for c in chunks])
        sr = int(chunks[0].sample_rate)
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        import wave

        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        dur = len(audio) / sr
        results.append({
            "case": case,
            "lang_code": lang,
            "audio_s": round(dur, 2),
            "synth_s": round(time.monotonic() - started, 2),
            "peak": round(float(np.max(np.abs(audio))), 4),
            "path": str(out_path),
        })
        print(f"{case}: {dur:.2f}s synth {results[-1]['synth_s']}s", flush=True)

    # Waveform-difference evidence: same text, different language prefill.
    import wave

    def samples(path):
        with wave.open(str(path)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)

    pairs = [("auto", "chinese"), ("auto", "english"), ("chinese", "english")]
    diffs = {}
    for a, b in pairs:
        sa, sb = samples(OUT / f"{a}.wav"), samples(OUT / f"{b}.wav")
        n = min(len(sa), len(sb))
        d = float(np.max(np.abs(sa[:n] - sb[:n]))) if n else 0.0
        diffs[f"{a}_vs_{b}"] = {"maxabs": round(d, 1), "len_a": len(sa), "len_b": len(sb)}
        print(f"{a} vs {b}: maxabs diff {d:.1f} (len {len(sa)} vs {len(sb)})", flush=True)

    # Optional ASR corroboration with whisper-small if installed in this venv.
    asr = {}
    try:
        import mlx_whisper  # type: ignore

        for case, _ in CASES:
            r = mlx_whisper.transcribe(str(OUT / f"{case}.wav"), path_or_hf_repo="mlx-community/whisper-small-mlx")
            asr[case] = r["text"].strip()
            print(f"ASR {case}: {asr[case]}", flush=True)
    except Exception as exc:  # noqa: BLE001 - ASR is optional corroboration
        asr["error"] = str(exc)
        print(f"ASR unavailable: {exc}", flush=True)

    (OUT / "results.json").write_text(json.dumps({
        "text": TEXT, "results": results, "waveform_diffs": diffs, "asr": asr,
    }, ensure_ascii=False, indent=2))
    print("wrote results.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
