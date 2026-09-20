"""dots.tts worker on Windows + CUDA — issue #25.

Runs inside the engine's own venv over the shared worker runtime
(indextts25_common_worker.py, issue #32): same wire protocol, same peak
self-check. The gate refuses to start without CUDA — the upstream runtime
would silently warn and run on CPU with one thread (minutes per sentence,
indistinguishable from a hang), which the gate turns into a loud failure.

Model load: ``DotsTtsRuntime.from_pretrained(<local weights dir>,
precision="bfloat16", optimize=False)`` — a local directory short-circuits
snapshot_download (verified upstream), so generation stays fully offline;
optimize=False (the upstream default) skips the cached streaming runners.

The WeTextProcessing/pynini import has been patched out at install time
(dots_tts_patch.py, CORRECTIONS.md C-003): normalization is this app's own
pre-engine layer (ADR-0008), ``normalize_text`` stays False.
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

CUDA_GATE_EXIT_CODE = 3

_state = {"tts": None}


def _fatal(message: str) -> None:
    common.fatal(CUDA_GATE_EXIT_CODE, message)


def check_cuda() -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - gate must produce an actionable message
        _fatal(
            f"torch is not importable in the engine venv ({exc}). "
            "Reinstall the engine: POST /engines/dots-tts-cuda/install."
        )
    if not torch.cuda.is_available():
        _fatal(
            "CUDA is not available (torch.cuda.is_available() is False). "
            "This engine requires an NVIDIA GPU with the CUDA torch build. Refusing to fall "
            "back to CPU — the upstream runtime would pin one thread and take minutes per "
            "sentence."
        )


def _load(model_dir: str):
    import time

    if _state["tts"] is not None:
        return _state["tts"]
    started = time.monotonic()
    common.log(f"dots.tts: loading weights from {model_dir} (cuda, bfloat16)")
    from dots_tts.runtime import DotsTtsRuntime

    tts = DotsTtsRuntime.from_pretrained(
        model_dir,  # local dir: short-circuits snapshot_download
        precision="bfloat16",  # the vendor-default precision, CUDA-tested (H800)
        optimize=False,  # skip the cached streaming runners / CUDA warmup
    )
    _state["tts"] = tts
    common.log(f"dots.tts: model loaded in {time.monotonic() - started:.1f}s on {tts.device}")
    return tts


def _synthesis(request: dict, load, memory_report, log) -> dict:
    import time

    import soundfile as sf

    output = request["output"]
    if not request.get("ref_audio"):
        raise RuntimeError(
            "dots.tts 是零样本复刻引擎，必须提供参考音频：请选择一个音色（或先创建音色）再生成"
        )
    started = time.monotonic()
    tts = load(request["model_dir"])
    if request.get("seed") not in (None, ""):
        # The runtime itself has no seed parameter; the upstream helper
        # (dots_tts.utils.util.seed_everything) is the supported way.
        from dots_tts.utils.util import seed_everything

        seed_everything(int(request["seed"]))

    prompt_text = (request.get("prompt_text") or "").strip() or None
    kwargs = {}
    if request.get("speaker_scale") not in (None, ""):
        kwargs["speaker_scale"] = float(request["speaker_scale"])
    if request.get("num_steps") not in (None, ""):
        kwargs["num_steps"] = int(request["num_steps"])
    if request.get("guidance_scale") not in (None, ""):
        kwargs["guidance_scale"] = float(request["guidance_scale"])

    result = tts.generate(
        text=request["text"],
        prompt_audio_path=request["ref_audio"],
        prompt_text=prompt_text,  # None -> x-vector-only timbre mode (verified optional)
        normalize_text=False,  # ADR-0008: normalization happened before the engine
        **kwargs,
    )
    audio = result["audio"].detach().cpu().float().numpy()
    if audio.ndim > 1:
        audio = audio.squeeze()
    sf.write(output, audio, int(result["sample_rate"]), subtype="PCM_16")
    rtf = result.get("rtf")
    log(
        f"dots.tts: synthesized in {time.monotonic() - started:.1f}s"
        + (f" (engine RTF {rtf:.2f})" if isinstance(rtf, (int, float)) else "")
    )
    out = {"audio_path": output, **common.verify_wav_output(output, log, "dots.tts")}
    out.update(memory_report())
    return out


def _memory_report() -> dict:
    try:
        import torch

        return {"memory": {"active_mb": torch.cuda.memory_allocated() // (1024 * 1024)}}
    except Exception:  # noqa: BLE001 - memory stats are best-effort telemetry
        return {"memory": {}}


def _unload() -> dict:
    import torch

    _state["tts"] = None
    gc.collect()
    report = {}
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            report = {"active_mb": torch.cuda.memory_allocated() // (1024 * 1024)}
        common.log(f"dots.tts: model unloaded, active {report.get('active_mb', '?')} MB")
    except Exception:  # noqa: BLE001
        common.log("dots.tts: model unloaded (no CUDA stats available)")
    return {"unloaded": True, "memory": report}


def main() -> None:
    common.serve(
        check_boot=check_cuda,
        exit_code=CUDA_GATE_EXIT_CODE,
        load=_load,
        unload=_unload,
        memory_report=_memory_report,
        ready_message="dots.tts: CUDA worker ready",
        synthesis=_synthesis,
        engine_label="dots.tts",
    )


if __name__ == "__main__":
    common.run_main(main)
