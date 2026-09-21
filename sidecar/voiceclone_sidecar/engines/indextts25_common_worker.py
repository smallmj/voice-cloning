"""Shared IndexTTS-2.5 worker runtime (issue #32).

Runs inside the ENGINE'S OWN VENV (never the sidecar's environment), because
torch + the indextts dependency set are the engine's private business. The
CUDA and MPS workers used to duplicate this file wholesale; the platform
shims (``indextts25_worker.py`` / ``indextts25_mps_worker.py``) now declare
only their gate, model-precision/device choice and memory telemetry, and
delegate the request loop here.

Protocol (see worker_supervisor): one JSON request per stdin line
    {"id": "...", "action": "synthesize"|"unload"|"shutdown",
     "text": "...", "ref_audio": "...", "lang": "zh", "output": "..."}
Log lines on stdout are forwarded verbatim; each request answers with exactly
one ``RESULT: {json}`` or ``WORKER_ERROR: {json}`` line.

Output contract (ADR-0002 rule 3): the peak self-check refuses to deliver an
empty or all-zero WAV — a 0-frame file and digital silence both fail the
request, on BOTH platforms, instead of shipping a broken result.

The module must stay importable standalone (the engine venv has no
voiceclone_sidecar), so it uses the standard library only until torch is
imported lazily inside the platform hooks.
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


def log(message: str) -> None:
    print(message, flush=True)


def reply(kind: str, req_id: str | None, payload: dict) -> None:
    body = {"id": req_id, **payload}
    print(f"{kind}: " + json.dumps(body, ensure_ascii=False), flush=True)


def fatal(exit_code: int, message: str) -> None:
    reply("WORKER_ERROR", None, {"fatal": True, "error": message})
    sys.exit(exit_code)


def verify_wav_output(output: str, log=log, engine_label="indextts-2.5") -> dict:
    """Peak self-check on the freshly written WAV (ADR-0002 rule 3).

    Catches silent truncation, empty (0-frame) output, all-zero output and
    clipping instead of shipping a broken WAV. Raises RuntimeError on the
    failure modes; returns the measured telemetry on success.
    """
    import wave

    with wave.open(output, "rb") as w:
        sample_rate = w.getframerate()
        channels = w.getnchannels()
        frames = w.getnframes()
        raw = w.readframes(frames)
    if frames == 0:
        # 0 frames is a synthesizer failure, not a success: refuse it on
        # every platform (issue #32 — the CUDA variant used to deliver an
        # empty WAV as a completed generation).
        raise RuntimeError(
            f"synthesis produced an empty audio file (0 frames, {sample_rate} Hz) — "
            "refusing to deliver an empty result"
        )
    import array

    samples = array.array("h")  # the engine writes 16-bit PCM
    samples.frombytes(raw)
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max((abs(s) for s in samples), default=0) / 32767.0
    if peak == 0.0:
        raise RuntimeError("synthesis produced silence — refusing to deliver an empty result")
    clipping = peak > 0.999
    duration = frames / sample_rate
    if clipping:
        log(f"{engine_label}: WARNING peak {peak:.3f} — output is clipped at full scale")
    log(f"{engine_label}: {duration:.2f}s at {sample_rate} Hz, peak {peak:.3f}")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_s": round(duration, 2),
        "peak": round(peak, 4),
        "clipping": clipping,
    }


def run_synthesis(request: dict, load, memory_report, log=log,
                  engine_label="indextts-2.5") -> dict:
    """One synthesize request: load the model, infer, verify the output.

    ``engine_label`` is forwarded by ``serve`` on every call (the substituted
    synthesis functions in other workers accept it too); ruff's F821 caught
    this name previously being used here without ever being defined, which
    made every IndexTTS-2.5 synthesize request fail with a TypeError.
    """
    output = request["output"]
    model_dir = request["model_dir"]
    started = time.monotonic()
    tts = load(model_dir)
    # duration_factor (issue #23): the sidecar's canonical speed adapter
    # maps user speed onto this INVERSE multiplier; only forwarded when set.
    infer_kwargs = {}
    duration_factor = request.get("duration_factor")
    if duration_factor:
        infer_kwargs["duration_factor"] = duration_factor
    tts.infer(
        spk_audio_prompt=request.get("ref_audio"),
        text=request["text"],
        output_path=output,
        lang=request.get("lang", "zh"),
        verbose=False,
        **infer_kwargs,
    )
    log(f"{engine_label}: synthesized in {time.monotonic() - started:.1f}s")

    result = {"audio_path": output, **verify_wav_output(output, log, engine_label)}
    result.update(memory_report())  # platform telemetry, already fault-isolated
    return result


def serve(*, check_boot, exit_code: int, load, unload, memory_report,
          ready_message: str, synthesis=None, engine_label="indextts-2.5") -> None:
    """The request loop every local worker built on this runtime runs.

    ``check_boot`` is the platform device gate: it must call ``fatal`` and
    exit when the machine cannot run the engine on the intended device —
    a silent CPU fallback would take minutes per sentence and is
    indistinguishable from a hang to the user.

    ``synthesis`` lets a non-IndexTTS engine reuse the whole protocol and
    substitute only the infer step (issue #25: FireRedTTS3); it receives
    ``(request, load, memory_report, log)`` and returns the result dict.
    The default is this module's IndexTTS-2.5 flow.
    """
    if synthesis is None:
        synthesis = run_synthesis
    check_boot()
    log(ready_message)
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
            reply("WORKER_ERROR", None, {"fatal": False, "error": f"bad request JSON: {exc}"})
            continue
        req_id = request.get("id")
        action = request.get("action", "synthesize")
        try:
            if action == "shutdown":
                reply("RESULT", req_id, {"bye": True})
                break
            if action == "unload":
                reply("RESULT", req_id, unload())
            elif action == "synthesize":
                reply("RESULT", req_id, synthesis(request, load, memory_report, log, engine_label))
            else:
                reply("WORKER_ERROR", req_id, {"fatal": False, "error": f"unknown action: {action}"})
        except Exception as exc:  # noqa: BLE001 - report per-request failures, keep the worker alive
            reply("WORKER_ERROR", req_id, {"fatal": False, "error": str(exc)})


def run_main(main_loop) -> None:
    """Boot wrapper: boot failures must be visible, not swallowed (exit 1;
    gate rejections exit with the gate code via ``fatal``/SystemExit)."""
    try:
        main_loop()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        fatal(1, str(exc))
