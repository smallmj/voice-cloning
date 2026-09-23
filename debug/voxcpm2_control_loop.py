"""Issue: VoxCPM2 控制指令被念出来 — 红/绿反馈环（合成侧）。

在 voxcpm2-mps 引擎 venv 里直接驱动 VoxCPM.generate，复现应用的实际
拼接路径（voxcpm2_base.control_prefix -> "(instruction)text"）：

  ref-src:  debug/qwen3-verify/english.wav（现成语音，作克隆参考）
  step A:   无指令、reference-only 合成一句已知文本 -> ref.wav（自建带转写参考）
  step B:   Hi-Fi 模式（prompt_wav+prompt_text，应用在有转写时走的模式）
            + 控制指令前缀 -> hifi-control.wav   [怀疑红点]
  step C:   timbre-only 模式（reference-only，上游 CLI 允许 control 的模式）
            + 控制指令前缀 -> timbre-control.wav
  step D:   Hi-Fi 模式、无指令 -> hifi-baseline.wav（对照）

随后由 transcribe venv 的 mlx-whisper 转写（见 voxcpm2_control_asr.py），
断言指令词不出现在转写里。
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "debug", "voxcpm2-control")
os.makedirs(OUT, exist_ok=True)
REF_SRC = os.path.join(ROOT, "debug", "qwen3-verify", "english.wav")
INSTRUCTION = "slow gentle warm female voice, soft tone"
TEXT = "今天下午我们在会议室讨论了新产品的发布计划。"

WEIGHTS = os.path.join(
    os.path.expanduser("~"), "Library/Application Support/VoiceClone/runtime/engines/voxcpm2-mps/weights"
)

sys.path.insert(0, os.path.join(ROOT, "sidecar"))
from voiceclone_sidecar.engines.voxcpm2_base import control_prefix  # noqa: E402


def main() -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    print("[loop] loading model ...", flush=True)
    t0 = time.monotonic()
    model = VoxCPM.from_pretrained(WEIGHTS, load_denoiser=False, optimize=False, device="mps")
    print(f"[loop] loaded in {time.monotonic()-t0:.1f}s", flush=True)
    sr = int(model.tts_model.sample_rate)

    def synth(tag: str, text: str, **kw) -> None:
        t = time.monotonic()
        wav = model.generate(text=text, **kw)
        path = os.path.join(OUT, f"{tag}.wav")
        sf.write(path, wav, sr, subtype="PCM_16")
        print(f"[loop] {tag}: {len(wav)/sr:.1f}s  ({time.monotonic()-t:.0f}s)  text={text!r}", flush=True)

    # step A: 造一个带已知转写的参考音频（timbre-only 合成已知句）
    BASE = "这是参考音频的转写内容，用于高保真续写模式。"
    synth("ref", BASE, reference_wav_path=REF_SRC, inference_timesteps=10)

    ref = os.path.join(OUT, "ref.wav")
    jobs = [
        ("hifi-control", control_prefix(INSTRUCTION, TEXT),
         dict(prompt_wav_path=ref, prompt_text=BASE)),
        ("timbre-control", control_prefix(INSTRUCTION, TEXT),
         dict(reference_wav_path=REF_SRC)),
        ("hifi-baseline", TEXT, dict(prompt_wav_path=ref, prompt_text=BASE)),
    ]
    for tag, text, kw in jobs:
        synth(tag, text, inference_timesteps=10, **kw)

    json.dump(
        {"instruction": INSTRUCTION, "text": TEXT, "base": BASE},
        open(os.path.join(OUT, "meta.json"), "w"),
        ensure_ascii=False, indent=1,
    )
    print("[loop] done", flush=True)


if __name__ == "__main__":
    main()
