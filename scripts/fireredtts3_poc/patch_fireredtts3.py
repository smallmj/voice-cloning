"""FireRedTTS3 PoC 补丁（issue #26）。

对上游 FireRedTTS3（Apache-2.0）源码做最小改写，使其能在 Apple Silicon
MPS 上以 fp32 + SDPA 运行。补丁面共四类，与 plan.md §5 的既定范围一致：

1. 设备选择 —— `torch.device('cuda')` 硬编码改为 cuda → mps → cpu 依次回退；
2. SDPA 替换 —— 所有 `attn_implementation='flash_attention_2'` 改为 `'sdpa'`
   （两个 core 已声明 `_supports_sdpa=True`，见 issue 记录）；
3. autocast 守卫 —— `@torch.autocast(device_type='cuda', dtype=torch.bfloat16)`
   追加 `enabled=torch.cuda.is_available()`：无 CUDA 时退化为 no-op（fp32），
   而不是在 MPS 上报错；
4. seed 守卫 —— `fix_seed` 的 `torch.cuda.manual_seed*` 仅在 CUDA 可用时调用。

fasttext 不改代码：`FireRedTTS3.__init__` 本就接受 `use_fasttext=False`
（README 标为可选，且初始化在 try/except 内），PoC 直接传 False、环境里
不装 fasttext。rotary_embedding.py 的 `autocast('cuda', enabled=False)`
本就是 no-op，不动。

补丁以「精确字符串替换」表达：每条补丁记录期望出现的原文，找不到视为
上游漂移，立即报错退出——这是有界 PoC 的锚点，防止静默打出无效补丁。
幂等：已经打过的补丁会被跳过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# (relative path, old, new, 期望出现次数)
PATCHES: list[tuple[str, str, str, int]] = [
    # 1. 设备选择（Base 与 Instruct 两个 core）
    (
        "fireredtts3/llm/fireredtts3_base.py",
        "self.device = torch.device('cuda')",
        "self.device = torch.device(\n"
        "            'cuda' if torch.cuda.is_available()\n"
        "            else 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())\n"
        "            else 'cpu'\n"
        "        )  # PoC patch (issue #26): device selection",
        1,
    ),
    (
        "fireredtts3/llm/fireredtts3_instruct.py",
        "self.device = torch.device('cuda')",
        "self.device = torch.device(\n"
        "            'cuda' if torch.cuda.is_available()\n"
        "            else 'mps' if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())\n"
        "            else 'cpu'\n"
        "        )  # PoC patch (issue #26): device selection",
        1,
    ),
    # 2. SDPA 替换
    (
        "fireredtts3/llm/fireredtts3_base.py",
        '"attn_implementation": "flash_attention_2",',
        '"attn_implementation": "sdpa",  # PoC patch (issue #26): flash_attn -> SDPA',
        1,
    ),
    (
        "fireredtts3/redae/redae.py",
        "attn_implementation='flash_attention_2',",
        "attn_implementation='sdpa',  # PoC patch (issue #26): flash_attn -> SDPA",
        2,
    ),
    # 3. autocast 守卫（Base / Instruct / RedAE 各一处）
    (
        "fireredtts3/llm/fireredtts3_base.py",
        "@torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
        "@torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available())  # PoC patch (issue #26)",
        1,
    ),
    (
        "fireredtts3/llm/fireredtts3_instruct.py",
        "@torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
        "@torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available())  # PoC patch (issue #26)",
        1,
    ),
    (
        "fireredtts3/redae/redae.py",
        "@torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
        "@torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available())  # PoC patch (issue #26)",
        1,
    ),
    # 4. seed 守卫
    (
        "fireredtts3/utils/utils.py",
        "    torch.cuda.manual_seed(seed)\n    torch.cuda.manual_seed_all(seed)",
        "    if torch.cuda.is_available():  # PoC patch (issue #26): guard on MPS/CPU\n"
        "        torch.cuda.manual_seed(seed)\n"
        "        torch.cuda.manual_seed_all(seed)",
        1,
    ),
]

PATCH_MARKER = "PoC patch (issue #26)"


def apply_patches(upstream: Path) -> list[str]:
    """Apply all patches to ``upstream`` in place. Returns applied patch descriptions."""
    applied: list[str] = []
    for rel, old, new, expected in PATCHES:
        path = upstream / rel
        text = path.read_text(encoding="utf-8")
        if PATCH_MARKER in text and old not in text and rel != "fireredtts3/redae/redae.py":
            # rough idempotency: patch marker present and the original pattern is gone
            applied.append(f"skip (already patched): {rel}")
            continue
        count = text.count(old)
        if count == 0:
            if new in text:
                applied.append(f"skip (already patched): {rel}")
                continue
            raise SystemExit(
                f"patch anchor not found in {rel}: {old[:60]!r}... "
                "上游代码已漂移，PoC 补丁需要重新核对。"
            )
        if count != expected:
            raise SystemExit(
                f"patch anchor in {rel} occurs {count} times, expected {expected}; "
                "上游代码已漂移，PoC 补丁需要重新核对。"
            )
        path.write_text(text.replace(old, new), encoding="utf-8")
        applied.append(f"patched: {rel} ({count} site(s))")
    return applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="FireRedTTS3 upstream checkout directory")
    args = parser.parse_args()
    for line in apply_patches(args.upstream.resolve()):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
