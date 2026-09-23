#!/usr/bin/env python3
"""[DEBUG-fw51] issue #51 real-machine verification on MPS.

Loads the patched upstream FireRedTTS3 once in the engine venv and runs:

V5 — quality params audible-difference check:
  - baseline (upstream defaults) vs inference_cfg 1.0 / 3.0
  - baseline vs n_timesteps 4 / 16
  - baseline vs cross_fade_ms 0 / 150 (multi-sentence text; also checks
    total duration shortens with overlap)
V6 — stop_threshold truncation behavior (0.1 / 0.5 / 0.9) to decide
  expose vs disclose.

Outputs pairwise maxabs diffs + durations to debug/fw51_out/results.json.
Usage (engine venv): "$V/bin/python" debug/fw51_verify.py
"""
import json
import sys
import time
from pathlib import Path

import sqlite3

RUNTIME = Path.home() / "Library/Application Support/voiceclone-app/runtime/engines/fireredtts3-mps"
DB = Path.home() / "Library/Application Support/voiceclone-app/data/library.db"
OUT = Path("debug/fw51_out")
OUT.mkdir(exist_ok=True)

con = sqlite3.connect(DB)
voice = json.loads(con.execute(
    "select payload from voices where id='7f57bc020e944d3fac6c09eedc11bd94'").fetchone()[0])
REF_AUDIO = str(Path.home() / "Library/Application Support/voiceclone-app/data/voices/7f57bc020e944d3fac6c09eedc11bd94/ref.mp3")
REF_TEXT = voice["reference"]["transcript"]

TEXT_1 = "今天天气真好，我们一起去公园散步吧。"
TEXT_MULTI = "深夜的城市并没有睡去。霓虹灯下，出租车缓缓驶过空旷的街道，远处传来末班地铁进站的轰鸣。"

sys.path.insert(0, str(RUNTIME / "upstream"))
import torch  # noqa: E402
import torchaudio  # noqa: E402
from fireredtts3.core import FireRedTTS3  # noqa: E402

t0 = time.monotonic()
tts = FireRedTTS3(pretrained_model_dir=str(RUNTIME / "weights"), use_fasttext=False, use_wetext=True)
print(f"[fw51] model loaded in {time.monotonic()-t0:.1f}s on {tts.device}", flush=True)

prompt_audio, prompt_sr = torchaudio.load(REF_AUDIO)

import soundfile as sf  # noqa: E402


def gen(tag: str, text: str, **kwargs):
    started = time.monotonic()
    wav, sr = tts.generate(
        language="Chinese",
        prompt_text=REF_TEXT,
        prompt_audio=prompt_audio,
        prompt_audio_sr=prompt_sr,
        text=text,
        **kwargs,
    )
    wav = wav.detach().cpu().float()
    if wav.ndim > 1:
        wav = wav.squeeze(0)
    path = OUT / f"{tag}.wav"
    sf.write(path, wav.numpy(), sr, subtype="PCM_16")
    dur = len(wav) / sr
    rtf = (time.monotonic() - started) / dur
    print(f"[fw51] {tag}: dur={dur:.2f}s rtf={rtf:.2f} kwargs={kwargs}", flush=True)
    return path, len(wav)


def maxabs(a: Path, b: Path) -> float:
    ia, ib = torchaudio.load(str(a)), torchaudio.load(str(b))
    n = min(ia[0].shape[1], ib[0].shape[1])
    return float((ia[0][0, :n] - ib[0][0, :n]).abs().max())


results = {}
base, base_n = gen("v5-base", TEXT_1)
for cfg in (1.0, 3.0):
    p, _ = gen(f"v5-cfg{cfg}", TEXT_1, inference_cfg=cfg)
    results[f"cfg{cfg}_vs_base_maxabs"] = maxabs(base, p)
for steps in (4, 16):
    p, _ = gen(f"v5-steps{steps}", TEXT_1, n_timesteps=steps)
    results[f"steps{steps}_vs_base_maxabs"] = maxabs(base, p)

mbase, mbase_n = gen("v5-multibase", TEXT_MULTI)
mf0, mf0_n = gen("v5-multifade0", TEXT_MULTI, cross_fade_ms=0.0)
mf150, mf150_n = gen("v5-multifade150", TEXT_MULTI, cross_fade_ms=150.0)
results["crossfade_samples"] = {"base": mbase_n, "fade0": mf0_n, "fade150": mf150_n}

# V6: stop_threshold truncation behavior
for st in (0.1, 0.5, 0.9):
    p, n = gen(f"v6-stop{st}", TEXT_1, stop_threshold=st)
    results[f"stop{st}_samples"] = n
    results[f"stop{st}_vs_base_maxabs"] = maxabs(base, p)

(OUT / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2), flush=True)
print("[fw51] done", flush=True)
