"""dots.tts WeTextProcessing-optional patch (issue #25).

Verification record (docs/research/CORRECTIONS.md C-003): dots.tts declares
``WeTextProcessing`` as a hard, unpinned dependency and imports the ``tn.*``
normalizers at MODULE TOP LEVEL in ``src/dots_tts/utils/text.py`` — the
package cannot even be imported without pynini, which has zero official
win_amd64 wheels. That dependency is exactly what plan.md §5 used to reject
dots.tts for Windows in the first place.

This repo's architecture makes engine-side text normalization redundant:
normalization happens in the app's OWN layer before any engine runs
(ADR-0008), and dots.tts's ``normalize_text`` defaults to False anyway.

So the integration choice is the FireRedTTS3 precedent (issue #26 /
ADR-0019): a minimal, anchor-checked patch makes the two imports optional —
the normalizer singletons raise a clear error if anyone ever calls them,
instead of failing at import time. ``normalize_text`` stays closed in this
app (declared ``breaks-pipeline``, never exposed).

The patch is applied to the INSTALLED package inside the engine venv, at
install time, with exact-string anchors: a missing anchor means upstream
drifted and the patch surface must be re-checked by a human — never a
silent no-op. Idempotent.
"""

from __future__ import annotations

from pathlib import Path

# (old, new, expected occurrence count) — against dots_tts/utils/text.py
PATCHES: list[tuple[str, str, int]] = [
    (
        "from tn.chinese.normalizer import Normalizer as ZhNormalizer\n"
        "from tn.english.normalizer import Normalizer as EnNormalizer",
        "try:  # dots.tts patch (issue #25): WeTextProcessing/pynini optional here\n"
        "    from tn.chinese.normalizer import Normalizer as ZhNormalizer\n"
        "    from tn.english.normalizer import Normalizer as EnNormalizer\n"
        "except ImportError:  # pynini ships no win_amd64 wheel; normalization is\n"
        "    # this app's own pre-engine layer (ADR-0008), normalize_text stays off\n"
        "    ZhNormalizer = None\n"
        "    EnNormalizer = None",
        1,
    ),
    (
        "def get_chinese_text_normalizer() -> ZhNormalizer:\n    return ZhNormalizer()",
        "def get_chinese_text_normalizer() -> ZhNormalizer:\n"
        "    if ZhNormalizer is None:  # dots.tts patch (issue #25)\n"
        "        raise RuntimeError(\"WeTextProcessing is not installed; engine-side text normalization is disabled in this app (ADR-0008).\")\n"
        "    return ZhNormalizer()",
        1,
    ),
    (
        "def get_english_text_normalizer() -> EnNormalizer:\n    return EnNormalizer()",
        "def get_english_text_normalizer() -> EnNormalizer:\n"
        "    if EnNormalizer is None:  # dots.tts patch (issue #25)\n"
        "        raise RuntimeError(\"WeTextProcessing is not installed; engine-side text normalization is disabled in this app (ADR-0008).\")\n"
        "    return EnNormalizer()",
        1,
    ),
]

PATCH_MARKER = "dots.tts patch (issue #25)"


class PatchAnchorError(RuntimeError):
    """An upstream anchor disappeared or moved: re-check the patch surface."""


def apply_patches(text: str) -> str:
    """Apply all patches to the text of ``dots_tts/utils/text.py``."""
    for old, new, expected in PATCHES:
        count = text.count(old)
        if count == 0:
            if new in text:
                continue  # already patched
            raise PatchAnchorError(
                f"patch anchor not found in dots_tts/utils/text.py: {old[:60]!r}... "
                "上游代码已漂移，补丁需要重新核对（CORRECTIONS.md C-003）。"
            )
        if count != expected:
            raise PatchAnchorError(
                f"patch anchor occurs {count} times, expected {expected}; "
                "上游代码已漂移，补丁需要重新核对（CORRECTIONS.md C-003）。"
            )
        text = text.replace(old, new)
    return text


def patched_text_py_path(venv: Path) -> Path:
    """Locate the installed ``dots_tts/utils/text.py`` inside an engine venv
    (POSIX and Windows layouts)."""
    candidates = sorted(venv.glob("lib/python3.*/site-packages/dots_tts/utils/text.py"))
    candidates += [venv / "Lib" / "site-packages" / "dots_tts" / "utils" / "text.py"]
    for path in candidates:
        if path.is_file():
            return path
    raise PatchAnchorError(
        f"dots_tts/utils/text.py not found in venv {venv} — was the dots.tts package installed?"
    )


def apply_to_venv(venv: Path, log) -> list[str]:
    """Apply the patch to the installed package. Returns applied descriptions."""
    path = patched_text_py_path(venv)
    text = path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return [f"skip (already patched): {path}"]
    patched = apply_patches(text)
    path.write_text(patched, encoding="utf-8")
    return [f"patched: {path}"]
