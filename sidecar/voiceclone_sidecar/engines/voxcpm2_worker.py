"""VoxCPM2 worker — the platform shim for MPS (macOS) and CUDA (Windows).

Runs inside the engine's own venv (issue #32 shared-runtime pattern; the
protocol/peak-self-check live in indextts25_common_worker.py and run
unchanged here via the ``synthesis`` hook). The platform gate is chosen by
``argv[1]``: ``mps`` refuses to start without a usable MPS device, ``cuda``
without a CUDA device — a silent CPU fallback would take minutes per
sentence and is indistinguishable from a hang to the user (spec, issue #4).

Model load: ``VoxCPM.from_pretrained(<local weights dir>, load_denoiser=False,
optimize=False)`` — a local directory short-circuits the Hub entirely
(verified upstream), so generation stays fully offline; load_denoiser=False
skips the ModelScope ZipEnhancer download; optimize=False skips
torch.compile (which upstream auto-skips on MPS anyway).
"""

from __future__ import annotations

import gc
import os
import sys

try:  # package import when imported as voiceclone_sidecar.engines.*
    from . import indextts25_common_worker as common
except ImportError:  # standalone run inside the engine venv
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import indextts25_common_worker as common

GATE_EXIT_CODE = 3

_state = {"tts": None}


def _fatal(message: str) -> None:
    common.fatal(GATE_EXIT_CODE, message)


def check_mps() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - gate must produce an actionable message
        _fatal(
            f"torch is not importable in the engine venv ({exc}). "
            "Reinstall the engine: POST /engines/voxcpm2-mps/install."
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


def check_cuda() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _fatal(
            f"torch is not importable in the engine venv ({exc}). "
            "Reinstall the engine: POST /engines/voxcpm2-cuda/install."
        )
    if not torch.cuda.is_available():
        _fatal(
            "CUDA is not available (torch.cuda.is_available() is False). "
            "This engine requires an NVIDIA GPU with the CUDA torch build. Refusing to fall "
            "back to CPU — generation would take minutes per sentence."
        )


def _load(model_dir: str):
    import time

    if _state["tts"] is not None:
        return _state["tts"]
    started = time.monotonic()
    gate = sys.argv[1] if len(sys.argv) > 1 else ""
    device = "mps" if gate == "mps" else "cuda"
    common.log(f"voxcpm2: loading model from {model_dir} (device={device})")
    from voxcpm import VoxCPM

    tts = VoxCPM.from_pretrained(
        model_dir,  # a local directory: upstream short-circuits the Hub
        load_denoiser=False,  # no ZipEnhancer / ModelScope download
        optimize=False,  # no torch.compile, no warmup generate
        device=device,
    )
    _state["tts"] = tts
    common.log(f"voxcpm2: model loaded in {time.monotonic() - started:.1f}s (device={device})")
    return tts


def _synthesis(request: dict, load, memory_report, log) -> dict:
    import time

    import soundfile as sf

    output = request["output"]
    if not request.get("ref_audio"):
        raise RuntimeError(
            "VoxCPM2 是零样本复刻引擎，必须提供参考音频：请选择一个音色（或先创建音色）再生成"
        )
    started = time.monotonic()
    model = load(request["model_dir"])

    # Three cloning modes (verified upstream): prompt pair REQUIRED together,
    # reference tokens work alone; mode ③ = both (max similarity).
    prompt_text = (request.get("prompt_text") or "").strip() or None
    kwargs = {}
    if prompt_text is not None:
        kwargs["prompt_wav_path"] = request["ref_audio"]
        kwargs["prompt_text"] = prompt_text
    if request.get("cfg_value") not in (None, ""):
        kwargs["cfg_value"] = float(request["cfg_value"])
    if request.get("inference_timesteps") not in (None, ""):
        kwargs["inference_timesteps"] = int(request["inference_timesteps"])
    if request.get("seed") not in (None, ""):
        kwargs["seed"] = int(request["seed"])

    wav = model.generate(
        text=request["text"],
        reference_wav_path=request["ref_audio"],
        **kwargs,
    )
    sr = int(model.tts_model.sample_rate)
    sf.write(output, wav, sr, subtype="PCM_16")
    log(f"voxcpm2: synthesized in {time.monotonic() - started:.1f}s")
    result = {"audio_path": output, **common.verify_wav_output(output, log, "voxcpm2")}
    result.update(memory_report())
    return result


def _memory_report() -> dict:
    try:
        import torch

        gate = sys.argv[1] if len(sys.argv) > 1 else ""
        if gate == "mps":
            return {"memory": {"active_mb": torch.mps.current_allocated_memory() // (1024 * 1024)}}
        return {"memory": {"active_mb": torch.cuda.memory_allocated() // (1024 * 1024)}}
    except Exception:  # noqa: BLE001 - memory stats are best-effort telemetry
        return {"memory": {}}


def _unload() -> dict:
    import torch

    _state["tts"] = None
    gc.collect()
    report = {}
    gate = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        if gate == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_built():
            torch.mps.empty_cache()
            report = {"active_mb": torch.mps.current_allocated_memory() // (1024 * 1024)}
        elif gate == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            report = {"active_mb": torch.cuda.memory_allocated() // (1024 * 1024)}
        common.log(f"voxcpm2: model unloaded, active {report.get('active_mb', '?')} MB")
    except Exception:  # noqa: BLE001
        common.log("voxcpm2: model unloaded (no device stats available)")
    return {"unloaded": True, "memory": report}


def main() -> None:
    gate = sys.argv[1] if len(sys.argv) > 1 else ""
    if gate not in ("mps", "cuda"):
        _fatal(f"worker gate not specified (argv[1] must be 'mps' or 'cuda', got {gate!r})")
    common.serve(
        check_boot=check_mps if gate == "mps" else check_cuda,
        exit_code=GATE_EXIT_CODE,
        load=_load,
        unload=_unload,
        memory_report=_memory_report,
        ready_message=f"voxcpm2: {gate} worker ready",
        synthesis=_synthesis,
        engine_label="voxcpm2",
    )


if __name__ == "__main__":
    common.run_main(main)
