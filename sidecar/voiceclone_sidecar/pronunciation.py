"""Pronunciation control: canonical input + per-engine grammar adapters.

ADR-0018 / issue #23: three engines speak three INCOMPATIBLE pronunciation
grammars —

- IndexTTS: inline angle-bracket annotations, pinyin letters UPPERCASE and
  the tone digit kept as a plain digit: ``他在银<行|XING2>里<行|HANG2>走了半天``
- MiniMax: parenthesized pinyin tags, lowercase letters + digit:
  ``(chu3)(li3)``
- dots.tts: tone-marked pinyin written directly in the text: ``hào``

The canonical input (what the user sees and types) is one declaration per
line, ``汉字=拼音`` with a lowercase tone digit (``行=xing2``); an optional
trailing ``5`` or no digit means neutral tone. Everything here is pure text
transformation — unit-testable without any model.
"""

from __future__ import annotations

import re

_PINYIN_RE = re.compile(r"^[a-zü]+[1-5]?$")
_ANN_LINE_RE = re.compile(r"^\s*(\S)\s*[=:]\s*([a-zü]+[1-5]?)\s*$")

# Standard tone-mark placement: mark the first 'a'; else first 'o'; else
# first 'e'; else the 'u' of a trailing "iu" / the 'i' of a trailing "ui";
# else the first vowel.
_MARK_TABLE = {
    ("a", 1): "ā", ("a", 2): "á", ("a", 3): "ǎ", ("a", 4): "à",
    ("e", 1): "ē", ("e", 2): "é", ("e", 3): "ě", ("e", 4): "è",
    ("i", 1): "ī", ("i", 2): "í", ("i", 3): "ǐ", ("i", 4): "ì",
    ("o", 1): "ō", ("o", 2): "ó", ("o", 3): "ǒ", ("o", 4): "ò",
    ("u", 1): "ū", ("u", 2): "ú", ("u", 3): "ǔ", ("u", 4): "ù",
    ("ü", 1): "ǖ", ("ü", 2): "ǘ", ("ü", 3): "ǚ", ("ü", 4): "ǜ",
}


def parse_annotations(spec: str) -> tuple[dict[str, str], list[str]]:
    """Parse the canonical ``汉字=拼音`` block.

    Returns ``(annotations, errors)``; errors are user-facing strings. An
    entry is skipped (never half-applied) when it is malformed, when the
    pinyin fails validation, or when the key is not a single character.
    """
    annotations: dict[str, str] = {}
    errors: list[str] = []
    for lineno, line in enumerate((spec or "").splitlines(), start=1):
        if not line.strip():
            continue
        m = _ANN_LINE_RE.match(line)
        if not m:
            errors.append(f"第 {lineno} 行无法解析：{line.strip()}")
            continue
        char, pinyin = m.group(1), m.group(2)
        if not _PINYIN_RE.match(pinyin):
            errors.append(f"第 {lineno} 行拼音无效：{pinyin}")
            continue
        annotations[char] = pinyin
    return annotations, errors


def _split_tone(pinyin: str) -> tuple[str, int]:
    if pinyin and pinyin[-1] in "12345":
        return pinyin[:-1], int(pinyin[-1])
    return pinyin, 5


def mark_pinyin(pinyin: str) -> str:
    """``hao4`` -> ``hào``; ``xing2`` -> ``xíng``; ``lü3`` -> ``lǚ``."""
    body, tone = _split_tone(pinyin)
    if tone == 5:
        return body
    def _mark(ch: str) -> str:
        return _MARK_TABLE.get((ch, tone), ch)

    if "a" in body:
        idx = body.index("a")
    elif "o" in body:
        idx = body.index("o")
    elif "e" in body:
        idx = body.index("e")
    elif body.endswith("iu"):
        idx = len(body) - 1  # liu4 -> liù: the mark goes on the last vowel (u)
    elif body.endswith("ui"):
        idx = len(body) - 1  # gui4 -> guì: the mark goes on the last vowel (i)
    else:
        idx = next((i for i, ch in enumerate(body) if ch in "aeiouü"), None)
    if idx is None:
        return body
    return body[:idx] + _mark(body[idx]) + body[idx + 1:]


def to_indextts(text: str, annotations: dict[str, str]) -> str:
    """IndexTTS grammar: ``行`` + ``xing2`` -> ``<行|XING2>`` (uppercase
    letters, plain tone digit — as upstream ``apply_pronunciation_annotations``
    expects)."""
    out = text
    for char, pinyin in annotations.items():
        body, tone = _split_tone(pinyin)
        tag = f"<{char}|{body.upper()}{'' if tone == 5 else tone}>"
        out = out.replace(char, tag)
    return out


def to_minimax(text: str, annotations: dict[str, str]) -> str:
    """MiniMax grammar: the annotated character is replaced by a lowercase
    parenthesized pinyin tag: ``吃`` + ``chu3`` -> ``(chu3)``."""
    out = text
    for char, pinyin in annotations.items():
        out = out.replace(char, f"({pinyin})")
    return out


def to_dots(text: str, annotations: dict[str, str]) -> str:
    """dots.tts grammar: the annotated character is replaced by tone-marked
    pinyin written inline: ``号`` + ``hao4`` -> ``hào``."""
    out = text
    for char, pinyin in annotations.items():
        out = out.replace(char, mark_pinyin(pinyin))
    return out


def rewrite(text: str, spec: str, grammar: str, log=None) -> str:
    """Parse a canonical annotation block and rewrite ``text`` in one of the
    three engine grammars. Invalid lines are logged and skipped — a bad
    annotation degrades to unannotated text, never to a failed generation."""
    annotations, errors = parse_annotations(spec)
    for err in errors:
        if log:
            log(f"发音标注：{err}（已跳过）")
    if not annotations:
        return text
    if grammar == "indextts":
        return to_indextts(text, annotations)
    if grammar == "minimax":
        return to_minimax(text, annotations)
    if grammar == "dots":
        return to_dots(text, annotations)
    raise ValueError(f"unknown pronunciation grammar: {grammar!r}")
