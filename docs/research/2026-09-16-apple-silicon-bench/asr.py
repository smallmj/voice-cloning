from faster_whisper import WhisperModel
m = WhisperModel("Systran/faster-whisper-medium", device="cpu", compute_type="int8")
for f in ["api0.wav","api1.wav","api2.wav"]:
    segs, info = m.transcribe(f, language="zh", beam_size=5)
    print(f, "->", "".join(s.text for s in segs).strip())
