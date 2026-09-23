#!/usr/bin/env python3
"""Feedback loop for the fireredtts/voxcpm2 intelligibility bug (issue #29-era).

For each generation in the library DB: transcribe the produced wav with the
app's bundled mlx-whisper tool and compute CER against the input text.

Red = CER > 0.6 (unintelligible output). Green = CER <= 0.6.
Usage: python3 debug/audio_intelligibility_loop.py [--engine SUBSTR] [--limit N]
"""
import json
import math
import sqlite3
import subprocess
import sys
import unicodedata
from pathlib import Path

db_override = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--db")), "")
DATA = (Path(db_override) if db_override.endswith(".db")
        else Path(db_override) / "library.db") if db_override \
    else Path(__file__).resolve().parent.parent / "sidecar" / "data" / "library.db"
VENV = Path.home() / "Library/Application Support/VoiceClone/runtime/engines/transcribe-local/venv"

engine_filter = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--engine")), "")
limit = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--limit")), "200"))

con = sqlite3.connect(DATA)
rows = con.execute(
    "select id, engine_id, status, text, json_extract(payload,'$.audio_file') "
    "from generations where status='succeeded' order by created_at desc limit ?",
    (limit,),
).fetchall()


def cer(ref: str, hyp: str) -> float:
    ref = [c for c in unicodedata.normalize("NFKC", ref).lower()
           if not unicodedata.category(c).startswith(("P", "Z", "N"))]
    hyp = [c for c in unicodedata.normalize("NFKC", hyp).lower()
           if not unicodedata.category(c).startswith(("P", "Z", "N"))]
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


results = {}
for gid, eng, st, text, wav in rows:
    if engine_filter and engine_filter not in eng:
        continue
    path = DATA.parent / "audio" / wav
    if not wav or not path.exists():
        continue
    r = subprocess.run(
        [str(VENV / "bin" / "mlx_whisper"), str(path), "--model",
         "mlx-community/whisper-large-v3-turbo", "--language", "zh",
         "--output-dir", "/tmp/asr_out", "--output-format", "txt"],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"[asr-error] {eng} {wav} rc={r.returncode} {r.stderr[-150:]}", file=sys.stderr)
    hyp = ""
    out = Path("/tmp/asr_out") / (path.stem + ".txt")
    if out.exists():
        hyp = out.read_text().strip()
    score = cer(text, hyp)
    results.setdefault(eng, []).append(score)
    flag = "RED " if score > 0.6 else "ok  "
    print(f"[{flag}] {eng:18} cer={score:.2f} text='{text[:20]}' asr='{hyp[:30]}'")

print("\n=== summary ===")
for eng, scores in results.items():
    avg = sum(scores) / len(scores)
    print(f"{eng:20} n={len(scores):3} avg_cer={avg:.2f} "
          f"red_count={sum(1 for s in scores if s > 0.6)}")
