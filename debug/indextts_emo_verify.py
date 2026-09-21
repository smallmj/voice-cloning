"""Issue #37 acceptance: real-synthesis verification that the emotion
parameters actually reach the engine and change the output.

Runs the INSTALLED IndexTTS-2.5 MPS engine through the shared pipeline's
prepare_synthesis adapter + the real worker protocol:

  run A — baseline (same-as-reference, no emotion overrides)
  run B — emo_mode=情感向量, emo_vector angry=1.0, emo_weight=0.65
  run C — emo_mode=情感描述文本, emo_text, no random

Pass criteria: every run produces a non-empty, non-silent WAV (the worker's
peak self-check already refuses those), and B/C/D outputs differ from their
baselines (parameter took effect). Upstream also logs "scaled emotion
vectors ... " when emo_vector is applied — we assert that log line appears.

Usage: uv run python ../debug/indextts_emo_verify.py  (from sidecar/)
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sidecar"))

from voiceclone_sidecar.engines.indextts25_base import prepare_synthesis  # noqa: E402
from voiceclone_sidecar.engines.indextts25_mps import IndexTts25MpsEngine  # noqa: E402

REF = "/Users/smallmj/My Projects/声音复刻软件/sidecar/data/audio/g1.wav"
TEXT = "今天天气真好，我们一起去公园散步吧。"


def wav_fingerprint(path: str) -> dict:
    with wave.open(path, "rb") as w:
        frames = w.getnframes()
        raw = w.readframes(frames)
    return {
        "frames": frames,
        "sha1": hashlib.sha1(raw).hexdigest(),
    }


def run(engine, label: str, params: dict, logs: list) -> dict:
    from voiceclone_sidecar.registry import GenerationRequest

    logs.clear()
    result = engine.synthesize(
        GenerationRequest(
            generation_id=f"emo-verify-{label}",
            text=TEXT,
            params={"ref_audio": REF, **params},
        ),        logs.append,
    )
    info = wav_fingerprint(result.audio_path)
    print(f"[{label}] {info} sample_rate={result.sample_rate}")
    return info


def main() -> int:
    engine = IndexTts25MpsEngine(output_dir=Path(tempfile.mkdtemp(prefix="emo-verify-")))
    assert engine.is_installed(), "engine must be installed for the verification run"

    logs: list[str] = []
    base = run(engine, "A-baseline", {}, logs)
    vec = run(
        engine, "B-vector",
        {"emo_mode": "情感向量", "emo_vector": "0,1,0,0,0,0,0,0", "emo_weight": "0.65"},
        logs,
    )
    scaled = [ln for ln in logs if "scaled emotion vector" in ln or "emo_vector" in ln]
    print("vector-mode log lines:", scaled[:3])
    text = run(
        engine, "C-text",
        {"emo_mode": "情感描述文本", "emo_text": "非常愤怒，大声呵斥"},
        logs,
    )
    ok = True
    for label, info in (("B", vec), ("C", text)):
        if info["sha1"] == base["sha1"]:
            print(f"FAIL: run {label} is byte-identical to baseline — parameter had no effect")
            ok = False
    if not scaled:
        print("FAIL: upstream never logged the scaled emotion vector — emo_vector not applied")
        ok = False
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
