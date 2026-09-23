# V3/V4 real-machine verification (issues #47/#48) — run inside the qwen3 venv.
import json
import os
import sys
import time
import wave

HOME = os.path.expanduser("~")
WEIGHTS = os.path.join(HOME, "Library/Application Support/voiceclone-app/runtime/engines/qwen3-tts-mlx/weights")
OUT_DIR = os.path.join(HOME, "My Projects/声音复刻软件/debug/qwen3-verify")
REF = os.path.join(HOME, "My Projects/声音复刻软件/docs/research/2026-09-16-apple-silicon-bench/ref.wav")
REF_TEXT = "大家好，欢迎使用这个语音合成系统。今天天气不错，我们一起来测试一下声音复刻的效果如何。"
os.makedirs(OUT_DIR, exist_ok=True)

# Force, not setdefault (same rule as the worker).
os.environ["HF_HUB_OFFLINE"] = "1"

from mlx_audio.tts.utils import load_model  # noqa: E402

import numpy as np  # noqa: E402

t0 = time.perf_counter()
model = load_model(WEIGHTS)
print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

CASES = [
    ("auto-no-code", "这是一段测试语音，看看它听起来如何。", None),
    ("chinese", "这是一段测试语音，看看它听起来如何。", "chinese"),
    ("english", "This is a test sentence, listen to how it sounds.", "english"),
    ("french", "Ceci est une phrase de test, écoutez comment cela sonne.", "french"),
    ("german", "Dies ist ein Testsatz, hören Sie, wie er klingt.", "german"),
    ("italian", "Questa è una frase di prova, ascolta come suona.", "italian"),
    ("japanese", "これはテストの文です、どのように聞こえるか聴いてください。", "japanese"),
    ("korean", "이것은 테스트 문장입니다, 어떻게 들리는지 들어보세요.", "korean"),
    ("portuguese", "Esta é uma frase de teste, ouça como soa.", "portuguese"),
    ("russian", "Это тестовое предложение, послушайте, как оно звучит.", "russian"),
    ("spanish", "Esta es una frase de prueba, escucha cómo suena.", "spanish"),
    # Discriminator: Chinese text + a mismatched lang_code. If the language
    # parameter is truly consumed upstream, this output must differ from
    # the auto case in more than random-sampling noise.
    ("mismatch-zh-as-japanese", "这是一段测试语音，看看它听起来如何。", "japanese"),
]

results = []
for name, text, lang_code in CASES:
    kwargs = {}
    if lang_code:
        kwargs["lang_code"] = lang_code
    t = time.perf_counter()
    try:
        chunks = list(model.generate(
            text=text, ref_audio=REF, ref_text=REF_TEXT, verbose=False, **kwargs))
        audio = np.concatenate([np.asarray(c.audio, dtype=np.float32) for c in chunks])
        sr = int(chunks[0].sample_rate)
        path = os.path.join(OUT_DIR, f"{name}.wav")
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        results.append({"case": name, "lang_code": lang_code, "audio_s": round(len(audio) / sr, 2),
                        "synth_s": round(time.perf_counter() - t, 2), "ok": True})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    except Exception as exc:  # noqa: BLE001
        results.append({"case": name, "lang_code": lang_code, "error": repr(exc), "ok": False})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

# V4: 500-char Chinese text, default max_tokens — does truncation occur?
LONG_TEXT = ("声音是人类最自然的表达方式。一副好的声音不仅传递信息，更传递温度与信任。"
             "复刻一段声音，需要捕捉音色、语气、节奏与情感的微妙平衡。"
             "技术让这件事第一次变得触手可及，但真正的挑战在于细节："
             "一个字的轻重、一次呼吸的停顿、一句尾音的上扬，都可能让复刻从相似变成神似。"
             "我们相信，好的工具应当尊重创作者，把复杂的参数留在幕后，"
             "把表达的自由交还给用户。这才是声音复刻的意义所在。")
LONG_TEXT = (LONG_TEXT + "细节决定成败，温度决定共鸣。") * 4
print(f"long text chars: {len(LONG_TEXT)}", flush=True)
t = time.perf_counter()
chunks = list(model.generate(text=LONG_TEXT, ref_audio=REF, ref_text=REF_TEXT, verbose=False))
audio = np.concatenate([np.asarray(c.audio, dtype=np.float32) for c in chunks])
sr = int(chunks[0].sample_rate)
path = os.path.join(OUT_DIR, "v4-long-500.wav")
pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
with wave.open(path, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(pcm.tobytes())
v4 = {"case": "v4-long", "chars": len(LONG_TEXT), "segments": len(chunks),
      "audio_s": round(len(audio) / sr, 2), "synth_s": round(time.perf_counter() - t, 2),
      "truncated": None}
print(json.dumps(v4, ensure_ascii=False), flush=True)
json.dump({"v3": results, "v4": v4}, open(os.path.join(OUT_DIR, "results.json"), "w"), ensure_ascii=False, indent=1)
print("DONE", flush=True)
