"""Qwen3-TTS MLX worker: runs inside the ENGINE'S OWN VENV (never the sidecar's
environment), because mlx-audio and its dependency set are the engine's
private business.

Protocol: one JSON request on stdin ->
    {"weights_dir": "...", "text": "...", "ref_audio": "...", "ref_text": "...",
     "output": "...", "language": "zh"}
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

    results = list(
        model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=request.get("language", "Chinese"),
            verbose=False,
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
