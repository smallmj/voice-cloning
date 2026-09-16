import os, time, resource, numpy as np
os.environ.setdefault("HF_HUB_OFFLINE","1")
from index_tts_2_5_mlx import IndexTTS
t0=time.perf_counter()
tts = IndexTTS()
t_load=time.perf_counter()-t0
t0=time.perf_counter(); spk = tts.build_speaker("ref.wav"); t_spk=time.perf_counter()-t0
texts=["人工智能模型在2025年处理了100万条数据。",
       "欢迎使用这款桌面软件，它可以在本地完成声音复刻与文字转语音。",
       "Please note that this is a mixed language test, 都可以。"]
tot_a=tot_s=0.0
for i,t in enumerate(texts):
    t0=time.perf_counter()
    sr,pcm = tts.clone(t, ref_audio_path=None, spk=spk, out=f"api{i}.wav")
    el=time.perf_counter()-t0
    dur=len(pcm)/sr
    tot_a+=dur; tot_s+=el
    print(f"line{i}: audio={dur:.2f}s synth={el:.2f}s rtf={el/dur:.3f}")
print(f"TOTAL audio={tot_a:.2f}s synth={tot_s:.2f}s RTF={tot_s/tot_a:.3f}")
print(f"load={t_load:.2f}s build_speaker={t_spk:.2f}s")
print("maxrss_MB=%.0f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576))
