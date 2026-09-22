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
    # Seed reproducibility warmup (issue #49): on CUDA the FIRST generation
    # after model load deviates from same-seed reruns (3090 实测 maxabs≈1.36)
    # — the same reason upstream runs a warmup generate when optimize=True.
    # One throwaway generation (synthetic 2 s sine as reference, 4 timesteps)
    # makes even the first seeded user request byte-reproducible (3090 实测
    # 逐字节一致, 2026-09-22).
    try:
        import tempfile

        import numpy as np
        import torch

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sr = 48000
            t = np.linspace(0, 2, sr * 2, endpoint=False)
            sf_data = (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32")
            import soundfile as _sf

            _sf.write(tmp.name, sf_data, sr, subtype="PCM_16")
            warm_ref = tmp.name
        torch.manual_seed(0)
        tts.generate(
            text="这是一次预热。",
            reference_wav_path=warm_ref,
            inference_timesteps=4,
        )
        os.unlink(warm_ref)
        common.log("voxcpm2: seed-reproducibility warmup done")
    except Exception as exc:  # noqa: BLE001 - warmup is best-effort
        common.log(f"voxcpm2: seed warmup skipped ({exc})")
    common.log(f"voxcpm2: model loaded in {time.monotonic() - started:.1f}s (device={device})")
    return tts


def _synthesis(request: dict, load, memory_report, log, engine_label="") -> dict:
    import time

    import soundfile as sf

    output = request["output"]
    # Voice Design (issue #57) is the ONLY reference-less entry: the engine
    # base sets ``design`` for its design_voice() calls and for nothing else;
    # plain synthesize keeps enforcing ref_audio (zero-shot cloning engine).
    design = bool(request.get("design"))
    if not design and not request.get("ref_audio"):
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
    # seed is NOT an upstream generate() parameter (pinned 2.0.3 _generate has
    # no seed arg and no **kwargs — passing it raises TypeError, issue #49).
    # Reproducibility is done worker-side via torch.manual_seed (dots
    # seed_everything precedent) plus the model-load warmup in _load(): with
    # both, same seed + same params is byte-identical on MPS and CUDA from the
    # very first request (3090 + Mac MPS 真机, 2026-09-22).
    if request.get("seed") not in (None, ""):
        import torch

        torch.manual_seed(int(request["seed"]))
    if request.get("min_len") not in (None, ""):
        kwargs["min_len"] = int(request["min_len"])
    if request.get("max_len") not in (None, ""):
        kwargs["max_len"] = int(request["max_len"])
    if request.get("retry_badcase") not in (None, ""):
        kwargs["retry_badcase"] = bool(request["retry_badcase"])
    if request.get("retry_badcase_max_times") not in (None, ""):
        kwargs["retry_badcase_max_times"] = int(request["retry_badcase_max_times"])
    if request.get("retry_badcase_ratio_threshold") not in (None, ""):
        kwargs["retry_badcase_ratio_threshold"] = float(request["retry_badcase_ratio_threshold"])

    if design:
        # No-reference generation (Voice Design): text = "(description)preview",
        # no reference_wav_path / prompt pair — the model invents the timbre.
        wav = model.generate(text=request["text"], **kwargs)
    else:
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
