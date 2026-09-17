"""Standalone transcription worker: runs inside the transcribe-local tool venv.

Protocol: one JSON request on stdin ->
    {"audio": "<path>", "weights_dir": "<local model dir>"}
The final stdout line is ``RESULT: {json}`` with {"text": "..."}.

Runs fully offline: the model is loaded from a local directory.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    req = json.loads(sys.stdin.read())
    if sys.platform == "darwin":
        import mlx_whisper

        result = mlx_whisper.transcribe(
            req["audio"], path_or_hf_repo=req["weights_dir"], verbose=False
        )
        text = result["text"].strip()
    else:
        from faster_whisper import WhisperModel

        model = WhisperModel(req["weights_dir"], device="cpu", compute_type="int8")
        segments, _info = model.transcribe(req["audio"])
        text = " ".join(s.text.strip() for s in segments).strip()
    print("RESULT: " + json.dumps({"text": text}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - surfaced to the sidecar, not swallowed
        print(f"WORKER_ERROR: {exc}", flush=True)
        sys.exit(1)
