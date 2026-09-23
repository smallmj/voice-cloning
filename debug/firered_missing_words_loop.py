#!/usr/bin/env python3
"""[DEBUG-fw1] Tight loop for FireRedTTS3 missing-words bug.

Loads the model ONCE in the engine venv, then generates the same texts under
different seed settings, transcribes each output with mlx-whisper, and prints
a CER matrix. Red = CER > 0.3 for that cell.

Usage (engine venv):
  "$V/bin/python" debug/firered_missing_words_loop.py --runs N
"""
import argparse
import json
import sqlite3
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

DB = Path.home() / "Library/Application Support/voiceclone-app/data/library.db"
RUNTIME = Path.home() / "Library/Application Support/voiceclone-app/runtime/engines/fireredtts3-mps"
WHISPER_VENV = Path.home() / "Library/Application Support/VoiceClone/runtime/engines/transcribe-local/venv"

con = sqlite3.connect(DB)
voice = json.loads(con.execute(
    "select payload from voices where id='7f57bc020e944d3fac6c09eedc11bd94'").fetchone()[0])
REF_AUDIO = Path.home() / "Library/Application Support/voiceclone-app/data/voices/7f57bc020e944d3fac6c09eedc11bd94/ref.mp3"
REF_TEXT = voice["reference"]["transcript"]

TEXTS = {
    "zh-pure": "今天天气真好，我们一起去公园散步吧。",
    "zh-mixed-en": "你好，世界。这是一次端到端生成测试。Hello, world!",
    "zh-long": "今天我们要讲的是为什么日本从一开始就注定必败，九月三号是抗日战争胜利纪念日，在这个特殊的日子里，我们不由得陷入深深的思考。",
}

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--seed", default="none")  # none | fixed
args = ap.parse_args()

sys.path.insert(0, str(RUNTIME / "upstream"))
import torch  # noqa: E402
import torchaudio  # noqa: E402
from fireredtts3.core import FireRedTTS3  # noqa: E402

t0 = time.monotonic()
tts = FireRedTTS3(pretrained_model_dir=str(RUNTIME / "weights"), use_fasttext=False, use_wetext=True)
print(f"[DEBUG-fw1] model loaded in {time.monotonic()-t0:.1f}s", flush=True)

import soundfile as sf  # noqa: E402
prompt_audio, prompt_sr = torchaudio.load(str(REF_AUDIO))

out_dir = Path("debug/fw1_out")
out_dir.mkdir(exist_ok=True)


def cer(ref: str, hyp: str) -> float:
    ref = [c for c in unicodedata.normalize("NFKC", ref).lower()
           if not unicodedata.category(c).startswith(("P", "Z", "N"))]
    hyp = [c for c in unicodedata.normalize("NFKC", hyp).lower()
           if not unicodedata.category(c).startswith(("P", "Z", "N"))]
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref) if ref else 0.0


def transcribe(path: Path) -> str:
    r = subprocess.run(
        [str(WHISPER_VENV / "bin" / "mlx_whisper"), str(path), "--model",
         "mlx-community/whisper-large-v3-turbo", "--language", "zh",
         "--output-format", "txt", "--output-dir", str(out_dir / "asr")],
        capture_output=True, text=True)
    txt = out_dir / "asr" / (path.stem + ".txt")
    return txt.read_text().strip() if txt.exists() else f"<asr-failed: {r.stderr[-200:]}>"


(out_dir / "asr").mkdir(exist_ok=True)
for name, text in TEXTS.items():
    for run in range(args.runs):
        seed = 1234 if args.seed == "fixed" else None
        wav_path = out_dir / f"{name}_r{run}.wav"
        t = time.monotonic()
        gen, sr = tts.generate(language="Chinese", prompt_text=REF_TEXT,
                               prompt_audio=prompt_audio, prompt_audio_sr=prompt_sr,
                               text=text, seed=seed)
        wav = gen.detach().cpu().float().numpy()
        if wav.ndim > 1:
            wav = wav.squeeze()
        sf.write(str(wav_path), wav, sr, subtype="PCM_16")
        dur = len(wav) / sr
        hyp = transcribe(wav_path)
        c = cer(text, hyp)
        verdict = "RED " if c > 0.3 else "ok  "
        print(f"[DEBUG-fw1] [{verdict}] {name} r{run} seed={seed} "
              f"dur={dur:.2f}s cer={c:.2f} asr={hyp[:60]!r}", flush=True)
