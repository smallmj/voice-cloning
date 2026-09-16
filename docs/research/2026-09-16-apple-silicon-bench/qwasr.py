from faster_whisper import WhisperModel
import glob
m = WhisperModel("Systran/faster-whisper-medium", device="cpu", compute_type="int8")
for f in sorted(glob.glob("qw*.wav")):
    segs, info = m.transcribe(f, language="zh", beam_size=5)
    print(f, "->", "".join(s.text for s in segs).strip())
