from faster_whisper import WhisperModel
m = WhisperModel("Systran/faster-whisper-medium", device="cpu", compute_type="int8")
import glob
for f in sorted(glob.glob("cbx*.wav"))+sorted(glob.glob("cb/cb*_000.wav")):
    segs, info = m.transcribe(f, language="zh", beam_size=5)
    print(f, "->", "".join(s.text for s in segs).strip())
