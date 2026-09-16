"""Install driver: runs the IndexTTS-2.5 engine install on the Windows host
and writes a timestamped log. Launched detached via Start-Process."""
import sys
import time

sys.path.insert(0, r"C:\VoiceClone\sidecar")

LOG_PATH = r"C:\VoiceClone\install_log.txt"
_logf = open(LOG_PATH, "a", encoding="utf-8")
_last_pct = {}


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def progress(name, done, total):
    if not total:
        return
    pct = done * 100 // total
    last = _last_pct.get(name, -1)
    if pct >= last + 10 or pct == 100:
        _last_pct[name] = pct
        log("%s: %d/%d bytes (%d%%)" % (name, done, total, pct))


from voiceclone_sidecar.engines.indextts25 import IndexTts25CudaEngine

engine = IndexTts25CudaEngine(output_dir=r"C:\VoiceClone\out")
log("engine: %s, runtime root: %s" % (engine.engine_id, engine.root))
try:
    engine.install(log, progress)
    log("INSTALL COMPLETE")
except Exception as exc:
    log("INSTALL FAILED: %s" % exc)
    raise
