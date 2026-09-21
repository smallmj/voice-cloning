"""IndexTTS-2.5 worker on Apple Silicon (MPS) — the platform shim (issue #32).

The MPS gate mirrors the CUDA gate (spec, issue #4): on boot the worker
asserts MPS is actually available. A silent CPU fallback would take minutes
per sentence and is indistinguishable from a hang to the user, so it refuses
to start instead.

Device choice: the upstream IndexTTS2 loader auto-detects MPS, but we pin
``device="mps"`` explicitly so the gate and the runtime can never disagree.
Precision stays fp32 (``use_bf16=False``) — the fp32-on-MPS path is the one
verified end to end on this machine family (2026-09-16, ComfyUI T8 node,
real generation); bf16 autocast on MPS is unverified upstream.

Everything else — the wire protocol, the load-once model cache, the
synthesize flow and the peak self-check (0-frame and silence refusal) —
lives in indextts25_common_worker.py and runs unchanged here.
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

MPS_GATE_EXIT_CODE = 3

_state = {"tts": None}


def _fatal(message: str) -> None:
    common.fatal(MPS_GATE_EXIT_CODE, message)


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


def _load(model_dir: str):

    if _state["tts"] is not None:
        return _state["tts"]
    started = time.monotonic()
    common.log(f"indextts-2.5: loading model from {model_dir} (device=mps, fp32)")
    # The v2.5 loader class is still named IndexTTS2 in infer_v2_5.py.
    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=os.path.join(model_dir, "config.yaml"),
        model_dir=model_dir,
        use_bf16=False,  # fp32 on MPS: the only precision verified on Apple Silicon
        device="mps",
        use_cuda_kernel=False,  # CUDA-only fused kernel; never on MPS
        use_torch_compile=False,
        # 情感描述文本模式需要 QwenEmotion（issue #37）；权重本就随主权重
        # 一起下载（qwen0.6bemo4-merge），构造期开启即可用。
        use_qwen_emo=True,
    )
    _state["tts"] = tts
    common.log(f"indextts-2.5: model loaded in {time.monotonic() - started:.1f}s")
    return tts


def _memory_report() -> dict:
    try:
        import torch

        return {"memory": {"active_mb": torch.mps.current_allocated_memory() // (1024 * 1024)}}
    except Exception:  # noqa: BLE001 - memory stats are best-effort telemetry
        return {"memory": {}}


def _unload() -> dict:
    import torch

    _state["tts"] = None
    gc.collect()
    report = {}
    if hasattr(torch, "mps") and torch.backends.mps.is_built():
        try:
            torch.mps.empty_cache()
            report = {"active_mb": torch.mps.current_allocated_memory() // (1024 * 1024)}
            common.log(f"indextts-2.5: model unloaded, MPS active {report['active_mb']} MB")
        except Exception:  # noqa: BLE001
            common.log("indextts-2.5: model unloaded (no MPS stats available)")
    return {"unloaded": True, "memory": report}


def main() -> None:
    common.serve(
        check_boot=check_mps,
        exit_code=MPS_GATE_EXIT_CODE,
        load=_load,
        unload=_unload,
        memory_report=_memory_report,
        ready_message="indextts-2.5: MPS worker ready",
    )


if __name__ == "__main__":
    common.run_main(main)
