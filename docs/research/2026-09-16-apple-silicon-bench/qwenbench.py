import time, resource, numpy as np, soundfile as sf
from mlx_audio.tts.utils import load_model
t0=time.perf_counter()
m = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
print("load=%.2fs"%(time.perf_counter()-t0))
REF="大家好，欢迎使用这个语音合成系统。今天天气不错，我们一起来测试一下声音复刻的效果如何。"
texts=["人工智能模型在2025年处理了100万条数据。",
       "欢迎使用这款桌面软件，它可以在本地完成声音复刻与文字转语音。",
       "千里之行，始于足下。"]
ta=ts=0.0
for i,t in enumerate(texts):
    t0=time.perf_counter()
    res=list(m.generate(t, ref_audio="ref.wav", ref_text=REF))
    el=time.perf_counter()-t0
    a=np.concatenate([np.array(r.audio) for r in res]) if len(res)>1 else np.array(res[0].audio)
    dur=len(a)/24000; ta+=dur; ts+=el
    sf.write(f"qw{i}.wav", a, 24000)
    print(f"line{i}: audio={dur:.2f}s synth={el:.2f}s rtf={el/dur:.3f}")
print(f"TOTAL audio={ta:.2f}s synth={ts:.2f}s RTF={ts/ta:.3f}")
print("maxrss_MB=%.0f"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576))
