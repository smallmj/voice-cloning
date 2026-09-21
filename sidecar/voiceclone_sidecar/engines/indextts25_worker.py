"""IndexTTS-2.5 worker on Windows + CUDA — the platform shim (issue #32).

The full worker used to live here verbatim, duplicated for MPS. Now only the
CUDA-specific deltas remain:

- the CUDA gate (spec, issue #4): on boot the worker asserts CUDA is
  available AND ``torch.version.cuda`` is non-empty. Otherwise it refuses to
  start with an actionable message — a silent CPU fallback would produce an
  "extremely slow" result the user cannot distinguish from a healthy one;
- precision: bf16 on the NVIDIA rig (the only CUDA path verified end to end);
- memory telemetry/reclamation: ``torch.cuda.mem_get_info`` / ``empty_cache``.

Everything else — the wire protocol, the load-once model cache, the
synthesize flow and the peak self-check (which now refuses 0-frame output on
CUDA too, issue #32) — lives in indextts25_common_worker.py and runs
unchanged here.
"""

from __future__ import annotations

import gc
import os
import sys
import time

try:  # package import when imported as voiceclone_sidecar.engines.*
    from . import indextts25_common_worker as common
except ImportError:  # standalone run inside the engine venv
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import indextts25_common_worker as common

CUDA_GATE_EXIT_CODE = 3

_state = {"tts": None}


def _fatal(message: str) -> None:
    common.fatal(CUDA_GATE_EXIT_CODE, message)


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


def _load(model_dir: str):

    if _state["tts"] is not None:
        return _state["tts"]
    started = time.monotonic()
    common.log(f"indextts-2.5: loading model from {model_dir}")
    # The v2.5 loader class is still named IndexTTS2 in infer_v2_5.py.
    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(cfg_path=os.path.join(model_dir, "config.yaml"), model_dir=model_dir,
                    use_bf16=True, use_cuda_kernel=False, use_torch_compile=False,
                    use_qwen_emo=True)  # 情感描述文本模式需要 QwenEmotion（issue #37）
    _state["tts"] = tts
    common.log(f"indextts-2.5: model loaded in {time.monotonic() - started:.1f}s")
    return tts


def _memory_report() -> dict:
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        return {"vram": {"free_mb": free // (1024 * 1024), "total_mb": total // (1024 * 1024)}}
    except Exception:  # noqa: BLE001 - VRAM stats are best-effort telemetry
        return {"vram": {}}


def _unload() -> dict:
    _state["tts"] = None
    try:
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        report = {"free_mb": free // (1024 * 1024), "total_mb": total // (1024 * 1024)}
        common.log(f"indextts-2.5: model unloaded, VRAM free {report['free_mb']}/{report['total_mb']} MB")
    except Exception:  # noqa: BLE001
        report = {}
        common.log("indextts-2.5: model unloaded (no CUDA stats available)")
    return {"unloaded": True, "vram": report}


def main() -> None:
    common.serve(
        check_boot=check_cuda,
        exit_code=CUDA_GATE_EXIT_CODE,
        load=_load,
        unload=_unload,
        memory_report=_memory_report,
        ready_message="indextts-2.5: worker ready",
    )


if __name__ == "__main__":
    common.run_main(main)
