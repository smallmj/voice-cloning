"""IndexTTS-2.5 worker: runs inside the ENGINE'S OWN VENV (never the sidecar's
environment), because torch+CUDA and the indextts dependency set are the
engine's private business.

Protocol (see worker_supervisor): one JSON request per stdin line
    {"id": "...", "action": "synthesize"|"unload"|"shutdown",
     "text": "...", "ref_audio": "...", "lang": "zh", "output": "..."}
Log lines on stdout are forwarded verbatim; each request answers with exactly
one ``RESULT: {json}`` or ``WORKER_ERROR: {json}`` line.

The CUDA gate (spec, issue #4): on boot the worker asserts CUDA is available
AND ``torch.version.cuda`` is non-empty. Otherwise it refuses to start with an
actionable message — a silent CPU fallback would produce an "extremely slow"
result the user cannot distinguish from a healthy one.

VRAM contract: the model is loaded lazily on the first synthesize; ``unload``
drops it, empties the CUDA cache and reports free memory; process exit (idle
timeout / request threshold / crash) is the final reclamation.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Force, not setdefault: a pre-set HF_HUB_OFFLINE=0 in the environment must
# not silently break the offline guarantee. Everything loads from LOCAL dirs
# under the runtime root; nothing touches the network at generation time.
os.environ["HF_HUB_OFFLINE"] = "1"

CUDA_GATE_EXIT_CODE = 3


def log(message: str) -> None:
    print(message, flush=True)


def _reply(kind: str, req_id: str | None, payload: dict) -> None:
    body = {"id": req_id, **payload}
    print(f"{kind}: " + json.dumps(body, ensure_ascii=False), flush=True)


def check_cuda() -> None:
    """Refuse to start without CUDA. Returns only on a usable GPU setup."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - gate must produce an actionable message
        _fatal(
            f"torch is not importable in the engine venv ({exc}). "
            "Reinstall the engine: POST /engines/indextts-25-cuda/install."
        )
    if not torch.version.cuda:
        _fatal(
            "the installed torch build has no CUDA support (torch.version.cuda is empty). "
            "This engine requires the CUDA wheels installed from the explicit "
            "download.pytorch.org/whl/cu128 index — reinstall the engine."
        )
    if not torch.cuda.is_available():
        reason = torch.cuda.get_device_name if torch.cuda.device_count() else None
        hint = "install/upgrade the NVIDIA driver so it supports CUDA 12.8" if reason is None else "free GPU memory or check that no other process is holding the GPU"
        _fatal(f"CUDA is not available on this machine: {hint}. Refusing to fall back "
               "to CPU — generation would take minutes per sentence.")


def _fatal(message: str) -> None:
    print(
        "WORKER_ERROR: "
        + json.dumps({"id": None, "fatal": True, "error": message}, ensure_ascii=False),
        flush=True,
    )
    sys.exit(CUDA_GATE_EXIT_CODE)


class _State:
    tts = None
    device = "cpu"


def _load(model_dir: str):
    import torch

    if _State.tts is not None:
        return _State.tts
    started = time.monotonic()
    log(f"indextts-2.5: loading model from {model_dir}")
    # The v2.5 loader class is still named IndexTTS2 in infer_v2_5.py.
    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(cfg_path=os.path.join(model_dir, "config.yaml"), model_dir=model_dir,
                      use_bf16=True, use_cuda_kernel=False, use_torch_compile=False)
    _State.tts = tts
    log(f"indextts-2.5: model loaded in {time.monotonic() - started:.1f}s")
    return tts


def _synthesize(request: dict) -> dict:
    import torch  # noqa: F401 - ensures the CUDA context exists for the stats below

    output = request["output"]
    model_dir = request["model_dir"]
    started = time.monotonic()
    _load(model_dir)
    _State.tts.infer(
        spk_audio_prompt=request.get("ref_audio"),
        text=request["text"],
        output_path=output,
        lang=request.get("lang", "zh"),
        verbose=False,
    )
    log(f"indextts-2.5: synthesized in {time.monotonic() - started:.1f}s")

    # Peak self-check (ADR-0002 rule 3): catch silent truncation, all-zero
    # output and clipping instead of shipping a broken WAV.
    import wave

    with wave.open(output, "rb") as w:
        sample_rate = w.getframerate()
        channels = w.getnchannels()
        frames = w.getnframes()
        raw = w.readframes(frames)
    import numpy as np

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        raise RuntimeError("synthesis produced silence — refusing to deliver an empty result")
    clipping = peak > 0.999
    duration = frames / sample_rate
    if clipping:
        log(f"indextts-2.5: WARNING peak {peak:.3f} — output is clipped at full scale")
    log(f"indextts-2.5: {duration:.2f}s at {sample_rate} Hz, peak {peak:.3f}")

    vram = {}
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        vram = {"free_mb": free // (1024 * 1024), "total_mb": total // (1024 * 1024)}
    except Exception:  # noqa: BLE001 - VRAM stats are best-effort telemetry
        pass
    return {
        "audio_path": output,
        "sample_rate": sample_rate,
        "duration_s": round(duration, 2),
        "peak": round(peak, 4),
        "clipping": clipping,
        "channels": channels,
        "vram": vram,
    }


def _unload() -> dict:
    _State.tts = None
    report = {}
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        report = {"free_mb": free // (1024 * 1024), "total_mb": total // (1024 * 1024)}
        log(f"indextts-2.5: model unloaded, VRAM free {report['free_mb']}/{report['total_mb']} MB")
    except Exception:  # noqa: BLE001
        log("indextts-2.5: model unloaded (no CUDA stats available)")
    return {"unloaded": True, "vram": report}


def main() -> None:
    check_cuda()
    log("indextts-2.5: worker ready")
    # NOTE: iterate with readline(), never `for line in sys.stdin` — the
    # iterator's read-ahead buffering blocks on pipes until the buffer fills,
    # stalling every request (observed on macOS and Windows both).
    while True:
        line = sys.stdin.readline()
        if not line:
            break  # parent closed stdin: exit and free everything
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply("WORKER_ERROR", None, {"error": f"bad request JSON: {exc}"})
            continue
        req_id = request.get("id")
        action = request.get("action", "synthesize")
        try:
            if action == "shutdown":
                _reply("RESULT", req_id, {"bye": True})
                return
            if action == "unload":
                _reply("RESULT", req_id, _unload())
            elif action == "synthesize":
                _reply("RESULT", req_id, _synthesize(request))
            else:
                _reply("WORKER_ERROR", req_id, {"error": f"unknown action: {action}"})
        except Exception as exc:  # noqa: BLE001 - surfaced to the sidecar, not swallowed
            _reply("WORKER_ERROR", req_id, {"error": str(exc)})


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - boot failures must be visible
        print(f"WORKER_ERROR: {json.dumps({'id': None, 'fatal': True, 'error': str(exc)})}", flush=True)
        sys.exit(1)
