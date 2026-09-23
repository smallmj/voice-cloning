"""红/绿断言侧：转写 voxcpm2_control_loop.py 的输出并检查指令是否被念出。

用法：transcribe-local venv python debug/voxcpm2_control_asr.py
红 = 转写里出现指令词（slow/gentle/warm/female/voice/soft/tone 的中文或英文痕迹）
"""
import json
import os
import re

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug", "voxcpm2-control")
from mlx_whisper import transcribe  # noqa: E402

meta = json.load(open(os.path.join(OUT, "meta.json")))
INSTR_WORDS = ["slow", "gentle", "warm", "female", "soft", "tone", "voice",
               "慢", "温", "柔", "女声", "语速"]

res = {}
for f in ["hifi-control", "timbre-control", "hifi-baseline"]:
    r = transcribe(os.path.join(OUT, f + ".wav"), path_or_hf_repo="mlx-community/whisper-small-mlx")
    text = r["text"].strip()
    hits = sorted({w for w in INSTR_WORDS if w.lower() in text.lower()})
    res[f] = {"asr": text, "instruction_leak": hits}
    print(f, "=>", text, flush=True)
    print("   leak:", hits or "无", flush=True)

json.dump(res, open(os.path.join(OUT, "asr.json"), "w"), ensure_ascii=False, indent=1)
leak_hifi = bool(res["hifi-control"]["instruction_leak"])
print("\nVERDICT hifi-control:", "RED（指令被念出）" if leak_hifi else "green")
