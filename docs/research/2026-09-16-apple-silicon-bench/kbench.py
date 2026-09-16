import time, resource, soundfile as sf
from misaki import zh
from kokoro_onnx import Kokoro
t0=time.perf_counter()
g2p = zh.ZHG2P(version="1.1")
kokoro = Kokoro("kokoro-v1.1-zh.onnx", "voices-v1.1-zh.bin", vocab_config="config.json")
print("load=%.2fs"%(time.perf_counter()-t0))
texts=["人工智能模型在2025年处理了100万条数据。",
       "欢迎使用这款桌面软件，它可以在本地完成声音复刻与文字转语音。",
       "千里之行，始于足下。"]
tot_a=tot_s=0.0
for i,t in enumerate(texts):
    t0=time.perf_counter(); ph,_=g2p(t); t_g=time.perf_counter()-t0
    t0=time.perf_counter(); s,sr=kokoro.create(ph,voice="zf_001",speed=1.0,is_phonemes=True); el=time.perf_counter()-t0
    dur=len(s)/sr; tot_a+=dur; tot_s+=el
    sf.write(f"k{i}.wav", s, sr)
    print(f"line{i}: g2p={t_g:.2f}s audio={dur:.2f}s synth={el:.2f}s rtf={el/dur:.3f}")
print(f"TOTAL audio={tot_a:.2f}s synth={tot_s:.2f}s RTF={tot_s/tot_a:.3f}")
print("maxrss_MB=%.0f"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576))
