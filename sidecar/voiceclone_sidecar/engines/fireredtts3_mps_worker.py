"""FireRedTTS3 worker on Apple Silicon (MPS) — issue #25 (ADR-0019).

Platform shim over the shared worker runtime (indextts25_common_worker.py,
issue #32): same wire protocol, same peak self-check (0-frame and silence
refusal), same MPS gate philosophy as the IndexTTS-2.5 MPS worker.

FireRedTTS3-specific deltas:

- the model comes from a PATCHED upstream checkout (not a pip package):
  the engine install step downloads the upstream source, applies the
  four-class patch table (fireredtts3_patch.py, ADR-0019 decision 2) and
  passes the checkout dir to the worker, which puts it on ``sys.path``;
- inference is ``FireRedTTS3.generate(language, prompt_text, prompt_audio,
  prompt_audio_sr, text, seed)`` — cloning ONLY works with the reference
  transcript (``prompt_text``), so a request without ``ref_text`` is
  refused up front (requires_reference_text=True, honest refusal);
- precision stays fp32 (autocast guarded by the patch), device pinned by
  the upstream-patch selection — the gate below refuses to start without
  a usable MPS device rather than silently falling back to CPU.

``fasttext`` is never installed/used (``use_fasttext=False``); ``wetext``
is used for text normalization (``use_wetext=True``), the pynini-free
normalizer the PoC verified (issue #26).
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
            "Reinstall the engine: POST /engines/fireredtts3-mps/install."
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
            "CPU — generation would take minutes per sentence (PoC RTF median 2.22 on GPU)."
        )


def _load(payload: dict):
    import time

    if _state["tts"] is not None:
        return _state["tts"]
    started = time.monotonic()
    upstream_dir = payload["upstream_dir"]
    weights_dir = payload["model_dir"]
    common.log(f"fireredtts3: loading Base weights from {weights_dir} (MPS, fp32, patched)")
    sys.path.insert(0, str(upstream_dir))
    from fireredtts3.core import FireRedTTS3  # upstream, after sys.path insert

    tts = FireRedTTS3(
        str(weights_dir),
        use_fasttext=False,  # optional upstream dep; never installed here
        use_wetext=True,  # pynini-free TN, verified by the issue-#26 PoC
    )
    _state["tts"] = tts
    common.log(f"fireredtts3: model loaded in {time.monotonic() - started:.1f}s on {tts.device}")
    return tts


def _synthesis(request: dict, load, memory_report, log, engine_label="") -> dict:
    import time

    import torchaudio

    ref_audio = request.get("ref_audio")
    ref_text = (request.get("ref_text") or "").strip()
    if not ref_audio:
        raise RuntimeError(
            "FireRedTTS3 是零样本复刻引擎，必须提供参考音频：请选择一个音色（或先创建音色）再生成"
        )
    if not ref_text:
        # The reference transcript is a hard requirement of the cloning
        # path (PoC: prompt_audio without prompt_text fails deep inside the
        # model). Refuse with an actionable message instead.
        raise RuntimeError(
            "FireRedTTS3 复刻需要参考文本：请为该音色提供参考音频的逐字转写（参考音频诊断可自动转写）"
        )
    output = request["output"]
    started = time.monotonic()
    tts = load(request)
    prompt_audio, prompt_sr = torchaudio.load(str(ref_audio))
    seed = request.get("seed")
    gen, sr = tts.generate(
        language=request.get("language", "Chinese"),
        prompt_text=ref_text,
        prompt_audio=prompt_audio,
        prompt_audio_sr=prompt_sr,
        text=request["text"],
        seed=int(seed) if seed not in (None, "") else None,
    )
    # Write 16-bit PCM WAV ourselves — same output contract as the other
    # local workers (the peak self-check below refuses empty/silent files).
    import soundfile as sf

    wav = gen.detach().cpu().float().numpy()
    if wav.ndim > 1:
        wav = wav.squeeze()
    sf.write(output, wav, sr, subtype="PCM_16")
    log(f"fireredtts3: synthesized in {time.monotonic() - started:.1f}s")
    result = {"audio_path": output, **common.verify_wav_output(output, log, "fireredtts3")}
    result.update(memory_report())
    return result


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
            common.log(f"fireredtts3: model unloaded, MPS active {report['active_mb']} MB")
        except Exception:  # noqa: BLE001
            common.log("fireredtts3: model unloaded (no MPS stats available)")
    return {"unloaded": True, "memory": report}


def main() -> None:
    common.serve(
        check_boot=check_mps,
        exit_code=MPS_GATE_EXIT_CODE,
        load=_load,
        unload=_unload,
        memory_report=_memory_report,
        ready_message="fireredtts3: MPS worker ready",
        synthesis=_synthesis,
        engine_label="fireredtts3",
    )


if __name__ == "__main__":
    common.run_main(main)
