"""FireRedTTS3 PoC 运行器（issue #26）。

在打好补丁的上游 checkout 上，用本地 Base 权重跑：

1. 三段固定中文样音（复刻模式：参考音频 + 参考文本）；
2. 现有 17 项中文回归集（sidecar/voiceclone_sidecar/regression.py 的
   REGRESSION_ITEMS，数字 / 日期 / 金额 / 中英混读 / 多音字）。

逐条记录音频时长、端到端墙钟时间、RTF 与 MPS 峰值分配内存，写入
``<out>/results.json``；音频写 ``<out>/wav/``。不加任何评分：回归集的
ASR 可懂度对照与人工听感判定在 PoC 之外完成（issue #26 把听感判定
留给人，这也是它标 ready-for-human 的原因）。

用法::

    python run_poc.py \
        --upstream ~/Library/Application\\ Support/VoiceClone/runtime/engines/fireredtts3-poc-upstream \
        --weights  ~/Library/Application\\ Support/VoiceClone/runtime/engines/fireredtts3-poc/weights \
        --reference sidecar/data/audio/e2e-offline.wav \
        --prompt-text "今天天气真好，我们一起去公园散步吧。" \
        --out docs/evaluation/fireredtts3_poc_run

前提：当前解释器已装 torch==2.8.0 torchaudio==2.8.0 transformers==5.6.2
einops wetext（见同目录 README.md），且已先跑 patch_fireredtts3.py。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "sidecar"))

# 三段固定样音：覆盖叙述、数字/术语、口语，供人工听感对比。
SAMPLE_TEXTS = [
    "声音是人与机器之间最古老的界面，如今它正在被重新定义。",
    "2026 年 9 月，这款模型的中文复刻相似度达到了 78.8 分。",
    "好了，今天就先聊到这里，我们下期再见。",
]


def peak_memory_mib() -> float | None:
    """MPS 当前分配峰值（MiB）；无 MPS 时返回 None。"""
    try:
        import torch

        if torch.backends.mps.is_available():
            import torch.mps

            torch.mps.synchronize()
            return round(torch.mps.current_allocated_memory() / 1024 / 1024, 1)
    except Exception:
        pass
    return None


def audio_seconds(path: Path) -> float:
    import torchaudio

    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="参考音频 wav")
    parser.add_argument("--prompt-text", required=True, help="参考音频的逐字转写")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-regression", action="store_true")
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream))

    import torch
    import torchaudio

    from fireredtts3.core import FireRedTTS3  # upstream, after sys.path insert

    from voiceclone_sidecar.regression import REGRESSION_ITEMS  # noqa: E402

    args.out.mkdir(parents=True, exist_ok=True)
    wav_dir = args.out / "wav"
    wav_dir.mkdir(exist_ok=True)

    print(f"loading model from {args.weights} ...", flush=True)
    t0 = time.perf_counter()
    tts = FireRedTTS3(str(args.weights), use_fasttext=False, use_wetext=True)
    load_s = round(time.perf_counter() - t0, 2)
    print(f"model loaded in {load_s}s on {tts.device}", flush=True)

    prompt_audio, prompt_sr = torchaudio.load(str(args.reference))

    jobs: list[dict] = [
        {"id": f"sample-{i+1}", "text": text, "kind": "sample"}
        for i, text in enumerate(SAMPLE_TEXTS)
    ]
    if not args.skip_regression:
        jobs += [{"id": item["id"], "text": item["text"], "kind": "regression", "category": item["category"]} for item in REGRESSION_ITEMS]

    results = []
    for job in jobs:
        t = time.perf_counter()
        try:
            gen, sr = tts.generate(
                language="Chinese",
                prompt_text=args.prompt_text,
                prompt_audio=prompt_audio,
                prompt_audio_sr=prompt_sr,
                text=job["text"],
                seed=1234,
            )
            wall = time.perf_counter() - t
            out_path = wav_dir / f"{job['id']}.wav"
            torchaudio.save(str(out_path), gen.cpu(), sr)
            dur = gen.shape[1] / sr
            results.append({
                **job,
                "ok": True,
                "audio_seconds": round(dur, 2),
                "wall_seconds": round(wall, 2),
                "rtf": round(wall / dur, 3) if dur > 0 else None,
                "sample_rate": sr,
                "mps_allocated_mib": peak_memory_mib(),
            })
            print(f"[ok] {job['id']}: {dur:.1f}s audio / {wall:.1f}s wall -> RTF {results[-1]['rtf']}", flush=True)
        except Exception as e:  # noqa: BLE001 - PoC: record and continue
            results.append({**job, "ok": False, "error": f"{type(e).__name__}: {e}"})
            print(f"[fail] {job['id']}: {e}", flush=True)

    ok = [r for r in results if r.get("ok")]
    summary = {
        "engine": "FireRedTTS3-Base (PoC, MPS)",
        "device": str(getattr(tts, "device", "?")),
        "precision": "fp32 (bf16 autocast disabled without CUDA)",
        "attention": "sdpa",
        "use_fasttext": False,
        "model_load_seconds": load_s,
        "reference_audio": str(args.reference),
        "prompt_text": args.prompt_text,
        "total_jobs": len(results),
        "ok_jobs": len(ok),
        "failed": [r["id"] for r in results if not r.get("ok")],
        "median_rtf": sorted(r["rtf"] for r in ok)[len(ok) // 2] if ok else None,
        "final_mps_allocated_mib": peak_memory_mib(),
        "results": results,
    }
    out_json = args.out / "results.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    return 0 if summary["ok_jobs"] == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
