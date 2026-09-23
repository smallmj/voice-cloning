import os, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from mlx_whisper import transcribe
OUT = os.path.join(os.path.expanduser("~"), "My Projects/声音复刻软件/debug/qwen3-verify")
r = transcribe(os.path.join(OUT, "v4-long-500.wav"), path_or_hf_repo="mlx-community/whisper-small-mlx")
print("LANG", r["language"]); print("TEXT", r["text"][:600])
