import os, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from mlx_whisper import transcribe
OUT = os.path.join(os.path.expanduser("~"), "My Projects/声音复刻软件/debug/qwen3-verify")
files = ["auto-no-code", "chinese", "english", "french", "german", "italian",
         "japanese", "korean", "portuguese", "russian", "spanish",
         "mismatch-zh-as-japanese"]
res = {}
for f in files:
    r = transcribe(os.path.join(OUT, f + ".wav"), path_or_hf_repo="mlx-community/whisper-small-mlx")
    res[f] = {"language": r["language"], "text": r["text"].strip()[:100]}
    print(f, json.dumps(res[f], ensure_ascii=False), flush=True)
json.dump(res, open(os.path.join(OUT, "asr.json"), "w"), ensure_ascii=False, indent=1)
