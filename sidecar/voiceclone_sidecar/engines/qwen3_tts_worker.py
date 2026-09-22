"""Qwen3-TTS MLX worker: runs inside the ENGINE'S OWN VENV (never the sidecar's
environment), because mlx-audio and its dependency set are the engine's
private business.

Protocol: one JSON request on stdin ->
    {"weights_dir": "...", "text": "...", "ref_audio": "...", "ref_text": "...",
     "output": "...", "lang_code": "zh",
     "temperature": 0.9, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05}
lang_code is OPTIONAL — absence means auto, and the sidecar only sends it
when the user picked a language (issue #47; V3 verification proved mlx-audio
never consumes it, so it is normally absent). The four sampling parameters
are ALWAYS sent by the sidecar, including the declared defaults, because
mlx-audio's hardcoded silent defaults (0.6/0.8/-1/1.3) differ from the
official generation_config (0.9/1.0/50/1.05) the UI shows (issue #48).
Progress lines on stdout are forwarded to the sidecar log bus verbatim; the
final line is ``RESULT: {json}`` with {"audio_path", "sample_rate"}.

The model is loaded from a LOCAL directory with HF_HUB_OFFLINE=1 — once the
weights are on disk, generation never touches the network.
"""

from __future__ import annotations

import json
import os
import sys
import time

# Force, not setdefault: a pre-set HF_HUB_OFFLINE=0 in the environment must
# not silently break the offline guarantee. The model loads from a LOCAL
# directory, so once weights are on disk, generation never touches the network.
os.environ["HF_HUB_OFFLINE"] = "1"


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    request = json.loads(sys.stdin.read())
    started = time.monotonic()

    weights_dir = request["weights_dir"]
    text = request["text"]
    output = request["output"]
    ref_audio = request.get("ref_audio")
    ref_text = request.get("ref_text")

    log(f"qwen3-tts-mlx: loading model from {weights_dir}")
    from mlx_audio.tts.utils import load_model

    model = load_model(weights_dir)
    log(f"qwen3-tts-mlx: model loaded in {time.monotonic() - started:.1f}s, synthesizing")

    # Issue #47: the upstream qwen3 pipeline key is `lang_code` (NOT
    # `language` — that lands in **kwargs and is silently dropped); the
    # lowercase value must match the weight codec_language_id keys. Absent
    # key = auto. Issue #48: the four sampling parameters are real upstream
    # form parameters; the sidecar sends them always (declared defaults
    # included) — forward whatever arrived.
    generate_kwargs = {}
    if request.get("lang_code"):
        generate_kwargs["lang_code"] = str(request["lang_code"]).lower()
    for name in ("temperature", "top_p", "top_k", "repetition_penalty"):
        if request.get(name) is not None:
            generate_kwargs[name] = request[name]

    results = list(
        model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            verbose=False,
            **generate_kwargs,
        )
    )
    import numpy as np

    audio = np.concatenate([np.asarray(r.audio, dtype=np.float32) for r in results])
    sample_rate = int(results[0].sample_rate)
    log(f"qwen3-tts-mlx: generated {len(audio) / sample_rate:.2f}s at {sample_rate} Hz")

    # ADR-0002 rule 3: peak self-check — catch silent clipping/encoding bugs
    # instead of shipping a saturated or all-zero WAV.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        raise RuntimeError("synthesis produced silence — refusing to write an empty result")
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    import wave

    with wave.open(output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    log(f"qwen3-tts-mlx: peak {peak:.3f}, wrote {output}")

    print(
        "RESULT: " + json.dumps({"audio_path": output, "sample_rate": sample_rate}),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - surfaced to the sidecar, not swallowed
        print(f"WORKER_ERROR: {exc}", flush=True)
        sys.exit(1)
