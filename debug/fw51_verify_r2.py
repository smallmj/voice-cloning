#!/usr/bin/env python3
"""[DEBUG-fw51b] Round 2: cross_fade on a splitting text + ASR verification."""
import json
import subprocess
import sys
import time
from pathlib import Path

import sqlite3

RUNTIME = Path.home() / "Library/Application Support/voiceclone-app/runtime/engines/fireredtts3-mps"
WHISPER_VENV = Path.home() / "Library/Application Support/VoiceClone/runtime/engines/transcribe-local/venv"
OUT = Path("debug/fw51_out")
OUT.mkdir(exist_ok=True)

con = sqlite3.connect(Path.home() / "Library/Application Support/voiceclone-app/data/library.db")
voice = json.loads(con.execute(
    "select payload from voices where id='7f57bc020e944d3fac6c09eedc11bd94'").fetchone()[0])
REF_AUDIO = str(Path.home() / "Library/Application Support/voiceclone-app/data/voices/7f57bc020e944d3fac6c09eedc11bd94/ref.mp3")
REF_TEXT = voice["reference"]["transcript"]

TEXT_LONG = ("FireRedTTS3 是本次接入的第五个本地引擎。它在苹果芯片上跑得非常稳定，"
             "复刻效果也让人惊喜。我们用十七个中文难例做了完整回归，全部通过。"
             "接下来的问题是参数面板要暴露哪些质量旋钮，真机验证之后才能下结论。"
             "这句足够长，一定会被切成多句再拼接起来。")

sys.path.insert(0, str(RUNTIME / "upstream"))
import torch  # noqa: E402
import torchaudio  # noqa: E402
import soundfile as sf  # noqa: E402
from fireredtts3.core import FireRedTTS3  # noqa: E402

t0 = time.monotonic()
tts = FireRedTTS3(pretrained_model_dir=str(RUNTIME / "weights"), use_fasttext=False, use_wetext=True)
print(f"[fw51b] loaded in {time.monotonic()-t0:.1f}s", flush=True)
prompt_audio, prompt_sr = torchaudio.load(REF_AUDIO)


def gen(tag: str, text: str, **kwargs):
    started = time.monotonic()
    wav, sr = tts.generate(language="Chinese", prompt_text=REF_TEXT,
                           prompt_audio=prompt_audio, prompt_audio_sr=prompt_sr, text=text, **kwargs)
    wav = wav.detach().cpu().float()
    if wav.ndim > 1:
        wav = wav.squeeze(0)
    path = OUT / f"{tag}.wav"
    sf.write(path, wav.numpy(), sr, subtype="PCM_16")
    print(f"[fw51b] {tag}: samples={len(wav)} kwargs={kwargs}", flush=True)
    return path, len(wav)


def asr(path: Path) -> str:
    out = subprocess.run(
        [str(WHISPER_VENV / "bin" / "python"), "-c",
         "import sys,mlx_whisper;print(mlux if 0 else mlx_whisper.transcribe(sys.argv[1],path_or_hf_repo='mlx-community/whisper-large-v3-turbo',language='zh')['text'])",
         str(path)],
        capture_output=True, text=True, timeout=300)
    return out.stdout.strip() or f"ASR-ERR: {out.stderr[-200:]}"


results = {}
variants = [
    ("r2-fade0", dict(cross_fade_ms=0.0)),
    ("r2-fade50", dict(cross_fade_ms=50.0)),
    ("r2-fade200", dict(cross_fade_ms=200.0)),
    ("r2-cfg1", dict(inference_cfg=1.0)),
    ("r2-cfg3", dict(inference_cfg=3.0)),
    ("r2-steps4", dict(n_timesteps=4)),
    ("r2-steps16", dict(n_timesteps=16)),
]
paths = {}
for tag, kw in variants:
    p, n = gen(tag, TEXT_LONG, **kw)
    paths[tag] = p
    results[tag] = {"samples": n, "text": asr(p)}
    print(f"[fw51b] ASR {tag}: {results[tag]['text'][:60]}", flush=True)

# how many sentences did the splitter produce?
from fireredtts3.core import FireRedTTS3 as _F  # already imported
text, lang, sentences = tts._apply_frontend(text=TEXT_LONG, language=None, do_clean=True, do_tn=True, do_split=True)
results["split_sentence_count"] = len(sentences)
results["sentences"] = sentences

(OUT / "results_r2.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in results.items() if k not in ("sentences",)}, ensure_ascii=False, indent=2), flush=True)
print("[fw51b] done", flush=True)
