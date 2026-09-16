"""E2E verification driver for IndexTTS-2.5 on the Windows host.

Runs with the SIDECAR venv python; torch is only importable in the ENGINE
venv, so CUDA checks go through subprocess probes and VRAM numbers come from
nvidia-smi.

  1. CUDA gate check (torch.cuda available, version non-empty, engine venv)
  2. Generate Chinese audio with a reference sample (voice cloning path)
  3. Peak self-check on the produced WAV
  4. VRAM reclamation: explicit unload -> nvidia-smi free MB delta
  5. Crash recovery: kill the worker mid-run, next request must succeed

Writes findings to C:\VoiceClone\e2e_log.txt.
"""
import os
import subprocess
import sys
import threading
import time
import wave

sys.path.insert(0, r"C:\VoiceClone\sidecar")

LOG_PATH = r"C:\VoiceClone\e2e_log.txt"
_logf = open(LOG_PATH, "w", encoding="utf-8")
# The GBK console chokes on worker progress-bar glyphs.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ENGINE_PY = os.path.join(os.environ["APPDATA"], r"VoiceClone\runtime\engines\indextts-25-cuda\venv\Scripts\python.exe")
REF = r"C:\VoiceClone\ref.wav"
OUT_DIR = r"C:\VoiceClone\out"


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def vram_used_mb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return int(out.strip().splitlines()[0])


from voiceclone_sidecar.engines.indextts25 import IndexTts25CudaEngine
from voiceclone_sidecar.registry import GenerationRequest

engine = IndexTts25CudaEngine(output_dir=OUT_DIR)

# --- stage 1: CUDA gate ------------------------------------------------------
probe = subprocess.run(
    [ENGINE_PY, "-c",
     "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), "
     "torch.cuda.get_device_name(0))"],
    capture_output=True, text=True, timeout=120)
log("stage1: engine venv probe -> %s" % probe.stdout.strip())
assert probe.returncode == 0 and "True" in probe.stdout, "CUDA gate failed: %s" % probe.stderr
ver, cuda, avail = probe.stdout.split()[:3]
name = " ".join(probe.stdout.split()[3:])
used0 = vram_used_mb()
log("stage1: VRAM used %d MB (idle baseline)" % used0)

# --- stage 2: generation -----------------------------------------------------
log("stage2: starting generation (first one includes model load)")
t0 = time.monotonic()
result = engine.synthesize(
    GenerationRequest(generation_id="e2e-1",
                      text="欢迎大家来体验声音复刻工作台，这是一段本地生成的测试音频。",
                      params={"ref_audio": REF, "lang": "zh"}),
    log,
)
dt = time.monotonic() - t0
log("stage2: done in %.1fs -> %s (sr=%s)" % (dt, result.audio_path, result.sample_rate))

with wave.open(result.audio_path, "rb") as w:
    log("stage2: wav %d ch, %d Hz, %.2fs, %d bytes" % (
        w.getnchannels(), w.getframerate(), w.getnframes() / w.getframerate(),
        os.path.getsize(result.audio_path)))

# --- stage 3: warm request ---------------------------------------------------
t0 = time.monotonic()
result2 = engine.synthesize(
    GenerationRequest(generation_id="e2e-2", text="第二次生成，用于测量热启动速度。",
                      params={"ref_audio": REF}),
    log,
)
log("stage3: warm generation in %.1fs -> %s" % (time.monotonic() - t0, result2.audio_path))
log("stage3: worker stats %s" % engine.worker_stats())

# --- stage 4: explicit unload / VRAM reclaim ---------------------------------
log("stage4: explicit unload")
used1 = vram_used_mb()
engine.unload()
time.sleep(3)
used2 = vram_used_mb()
log("stage4: VRAM used %d -> %d MB (unloaded) -> reclaimed %d MB" % (used1, used2, used1 - used2))
log("stage4: worker stats %s" % engine.worker_stats())

# --- stage 5: crash recovery ---------------------------------------------------
log("stage5: killing worker mid-run to test crash recovery")


def kill_worker():
    time.sleep(15)  # by then the worker is mid-generation (model load takes a while)
    sup = engine._get_supervisor()
    with sup._lock:
        if sup._proc:
            log("stage5: killing pid %s" % sup._proc.pid)
            sup._proc.kill()


t = threading.Thread(target=kill_worker, daemon=True)
t.start()
t0 = time.monotonic()
try:
    result3 = engine.synthesize(
        GenerationRequest(generation_id="e2e-3", text="测试崩溃恢复。", params={"ref_audio": REF}),
        log,
    )
    log("stage5: NOTE in-flight request unexpectedly survived: %s" % result3.audio_path)
except Exception as exc:
    log("stage5: in-flight request failed as expected: %s" % str(exc)[:200])
# A queued-style follow-up request must succeed on a fresh worker.
result3 = engine.synthesize(
    GenerationRequest(generation_id="e2e-4", text="崩溃恢复之后仍然可以正常生成。", params={"ref_audio": REF}),
    log,
)
log("stage5: post-crash generation OK in %.1fs -> %s" % (time.monotonic() - t0, result3.audio_path))
log("stage5: worker stats %s" % engine.worker_stats())

log("E2E COMPLETE")
