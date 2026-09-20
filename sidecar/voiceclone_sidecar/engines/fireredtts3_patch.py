"""FireRedTTS3 MPS patches (issue #26 PoC, adopted by ADR-0019 for issue #25).

Single source of truth for the four upstream rewrites that make FireRedTTS3
run on Apple Silicon (MPS, fp32 + SDPA). The PoC CLI
(``scripts/fireredtts3_poc/patch_fireredtts3.py``) delegates here; the
engine installer applies the same table during its ``engine`` step —
the PoC and the product can never drift apart.

Patch surface (four classes):

1. Device selection — hardcoded ``torch.device('cuda')`` becomes
   cuda → mps → cpu fallback;
2. SDPA — every ``attn_implementation='flash_attention_2'`` becomes
   ``'sdpa'`` (both cores declare ``_supports_sdpa=True``);
3. autocast guard — bf16 autocast gets ``enabled=torch.cuda.is_available()``
   so it degrades to fp32 (no-op) on MPS instead of erroring;
4. seed guard — ``torch.cuda.manual_seed*`` only when CUDA is available.

``fasttext`` is never installed and never patched:
``FireRedTTS3(..., use_fasttext=False)`` is upstream-supported.

Every patch is an exact-string replacement anchored on the upstream text:
a missing anchor means upstream drifted and a human must re-check the patch
surface (ADR-0019 decision 2 — accepting upstream maintenance means the
anchor check fails loudly, never silently produces a no-op patch).
Idempotent: already-patched files are skipped.
"""

from __future__ import annotations

from pathlib import Path

# (relative path, old, new, expected occurrence count)
PATCHES: list[tuple[str, str, str, int]] = [
    # 1. Device selection (Base and Instruct cores)
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
    # 2. SDPA replacement
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
    # 3. autocast guards (Base / Instruct / RedAE)
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
    # 4. Seed guard
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


class PatchAnchorError(RuntimeError):
    """An upstream anchor disappeared or moved: the patch surface must be
    re-checked by a human. Never silently skip."""


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
            raise PatchAnchorError(
                f"patch anchor not found in {rel}: {old[:60]!r}... "
                "上游代码已漂移，补丁需要重新核对（ADR-0019 决策 2）。"
            )
        if count != expected:
            raise PatchAnchorError(
                f"patch anchor in {rel} occurs {count} times, expected {expected}; "
                "上游代码已漂移，补丁需要重新核对（ADR-0019 决策 2）。"
            )
        path.write_text(text.replace(old, new), encoding="utf-8")
        applied.append(f"patched: {rel} ({count} site(s))")
    return applied
