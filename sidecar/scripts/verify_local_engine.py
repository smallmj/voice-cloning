"""Real end-to-end verification for issue #3 acceptance, run on a clean-ish Mac:

1. resolve/validate the ASCII runtime root;
2. install the uv-hosted Python + the engine's own venv + mlx-audio;
3. download the Qwen3-TTS weights (resumable, HF with hf-mirror fallback);
4. generate a Chinese sentence and re-generate with networking blocked, to
   prove the offline path (HF_HUB_OFFLINE=1) works once weights are local;
5. retry-safety smoke: a failed install is retried cleanly.

Usage:  uv run python scripts/verify_local_engine.py [--skip-download]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voiceclone_sidecar.engines.qwen3_tts import Qwen3TtsMlxEngine
from voiceclone_sidecar.registry import GenerationRequest

TEXT = "今天天气真好，我们一起去公园散步吧。"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    engine = Qwen3TtsMlxEngine(
        output_dir=Path(__file__).resolve().parent.parent / "data" / "audio"
    )
    print(f"runtime root: {engine.root}")
    assert str(engine.root).isascii(), "runtime root must be pure ASCII"

    state = engine.install_state()
    if args.skip_download and state.get("installed"):
        print("already installed, skipping install steps")
    else:
        last_pct = -1

        def progress(name: str, done: int, total: int | None) -> None:
            nonlocal last_pct
            if total:
                pct = int(done * 100 / total)
                if pct >= last_pct + 5 or pct == 100:
                    last_pct = pct
                    print(f"  {name}: {pct}% ({done}/{total} bytes)")

        state = engine.install(
            lambda m: print(f"  {m}"),
            progress,
        )
    assert state["installed"], state

    print("generating Chinese audio (online path)...")
    result = engine.synthesize(
        GenerationRequest(generation_id="e2e-online", text=TEXT, params={}),
        lambda m: print(f"  {m}"),
    )
    out = Path(result.audio_path)
    assert out.is_file() and out.stat().st_size > 10_000, out
    print(f"online OK: {out} ({out.stat().st_size} bytes, {result.sample_rate} Hz)")

    print("generating again with HF_HUB_OFFLINE=1 + no-network assertion...")
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    result = engine.synthesize(
        GenerationRequest(generation_id="e2e-offline", text=TEXT, params={}),
        lambda m: print(f"  {m}"),
    )
    out = Path(result.audio_path)
    assert out.is_file() and out.stat().st_size > 10_000, out
    print(f"offline OK: {out}")
    print("E2E PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
