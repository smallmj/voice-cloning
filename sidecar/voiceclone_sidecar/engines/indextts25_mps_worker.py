"""IndexTTS-2.5 worker for Apple Silicon (MPS). Runs inside the ENGINE'S OWN
VENV, same protocol as the CUDA worker (see indextts25_worker.py): one JSON
request per stdin line, one ``RESULT:``/``WORKER_ERROR:`` line per request.

The MPS gate mirrors the CUDA gate (spec, issue #4): on boot the worker
asserts MPS is actually available. A silent CPU fallback would take minutes
per sentence and is indistinguishable from a hang to the user, so it refuses
to start instead.

Device choice: the upstream IndexTTS2 loader auto-detects MPS, but we pin
``device="mps"`` explicitly so the gate and the runtime can never disagree.
Precision stays fp32 (``use_bf16=False``) — the fp32-on-MPS path is the one
verified end to end on this machine family (2026-09-16, ComfyUI T8 node,
real generation); bf16 autocast on MPS is unverified upstream.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"

MPS_GATE_EXIT_CODE = 3


def log(message: str) -> None:
    print(message, flush=True)


def _reply(kind: str, req_id: str | None, payload: dict) -> None:
    body = {"id": req_id, **payload}
    print(f"{kind}: " + json.dumps(body, ensure_ascii=False), flush=True)


def _fatal(message: str) -> None:
    print(
        "WORKER_ERROR: "
        + json.dumps({"id": None, "fatal": True, "error": message}, ensure_ascii=False),
        flush=True,
    )
    sys.exit(MPS_GATE_EXIT_CODE)


def check_mps() -> None:
    """Refuse to start without a usable MPS device."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - gate must produce an actionable message
        _fatal(
            f"torch is not importable in the engine venv ({exc}). "
            "Reinstall the engine: POST /engines/indextts-25-mps/install."
        )
    if not torch.backends.mps.is_built():
        _fatal(
            "the installed torch build has no MPS support (torch.backends.mps is not built). "
            "This engine requires the standard macOS arm64 torch wheels — reinstall the engine."
        )
    if not torch.backends.mps.is_available():
        _fatal(
            "MPS is not available on this machine (torch.backends.mps.is_available() is False). "
            "This engine requires Apple Silicon running macOS 12.3+. Refusing to fall back to "
            "CPU — generation would take minutes per sentence."
        )


class _State:
    tts = None
    device = "cpu"


def _load(model_dir: str):
    import torch

    if _State.tts is not None:
        return _State.tts
    started = time.monotonic()
    log(f"indextts-2.5: loading model from {model_dir} (device=mps, fp32)")
    # The v2.5 loader class is still named IndexTTS2 in infer_v2_5.py.
    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=os.path.join(model_dir, "config.yaml"),
        model_dir=model_dir,
        use_bf16=False,  # fp32 on MPS: the only precision verified on Apple Silicon
        device="mps",
        use_cuda_kernel=False,  # CUDA-only fused kernel; never on MPS
        use_torch_compile=False,
    )
    _State.tts = tts
    _State.device = "mps"
    log(f"indextts-2.5: model loaded in {time.monotonic() - started:.1f}s")
    return tts


def _synthesize(request: dict) -> dict:
    output = request["output"]
    model_dir = request["model_dir"]
    started = time.monotonic()
    _load(model_dir)
    # duration_factor (issue #23): the sidecar's canonical speed adapter
    # maps user speed onto this INVERSE multiplier; only forwarded when set.
    infer_kwargs = {}
    duration_factor = request.get("duration_factor")
    if duration_factor:
        infer_kwargs["duration_factor"] = duration_factor
    _State.tts.infer(
        spk_audio_prompt=request.get("ref_audio"),
        text=request["text"],
        output_path=output,
        lang=request.get("lang", "zh"),
        verbose=False,
        **infer_kwargs,
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
    if frames == 0:
        raise RuntimeError("synthesis produced an empty audio file")
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

    memory = {}
    try:
        import torch

        memory = {"active_mb": torch.mps.current_allocated_memory() // (1024 * 1024)}
    except Exception:  # noqa: BLE001 - memory stats are best-effort telemetry
        pass
    return {
        "audio_path": output,
        "sample_rate": sample_rate,
        "duration_s": round(duration, 2),
        "peak": round(peak, 4),
        "clipping": clipping,
        "channels": channels,
        "memory": memory,
    }


def _unload() -> dict:
    import gc

    import torch

    _State.tts = None
    gc.collect()
    report = {}
    if hasattr(torch, "mps") and torch.backends.mps.is_built():
        try:
            torch.mps.empty_cache()
            report = {"active_mb": torch.mps.current_allocated_memory() // (1024 * 1024)}
            log(f"indextts-2.5: model unloaded, MPS active {report['active_mb']} MB")
        except Exception:  # noqa: BLE001
            log("indextts-2.5: model unloaded (no MPS stats available)")
    return {"unloaded": True, "memory": report}


def main() -> None:
    check_mps()
    log("indextts-2.5: MPS worker ready")
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply("WORKER_ERROR", None, {"fatal": False, "error": f"bad request json: {exc}"})
            continue
        req_id = request.get("id")
        action = request.get("action")
        try:
            if action == "synthesize":
                _reply("RESULT", req_id, _synthesize(request))
            elif action == "unload":
                _reply("RESULT", req_id, _unload())
            elif action == "shutdown":
                _reply("RESULT", req_id, {"bye": True})
                break
            else:
                _reply("WORKER_ERROR", req_id, {"fatal": False, "error": f"unknown action: {action!r}"})
        except Exception as exc:  # noqa: BLE001 - report per-request failures, keep the worker alive
            _reply("WORKER_ERROR", req_id, {"fatal": False, "error": str(exc)})


if __name__ == "__main__":
    main()
