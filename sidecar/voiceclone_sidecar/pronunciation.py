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

import gzip
import re
from functools import lru_cache
from pathlib import Path

_PINYIN_RE = re.compile(r"^[a-zü]+[1-5]?$")
# Line separator is ``=`` / ``:`` (canonical) or ``|`` (issue #56 English
# form ``词|ARPAbet``). The key is a single CJK character or an English word.
_ANN_LINE_RE = re.compile(r"^\s*(\S+)\s*([=:|])\s*(.+?)\s*$")
_ENGLISH_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")
# The closed ARPAbet phone inventory (CMUdict symbols; stress digits 0/1/2
# are carried by the token suffix). Validating against the SET — not just
# the shape — is what makes "NOT A PHONEME" a bad annotation instead of
# three fake phones.
_ENGLISH_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")

_ARPABET_PHONES = frozenset(
    ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
)


def _is_arpabet_tokens(tokens: list[str]) -> bool:
    return bool(tokens) and all(
        (base := t.rstrip("012")) in _ARPABET_PHONES
        and (t == base or t[len(base):] in ("0", "1", "2"))
        for t in tokens
    )


# Bundled slim CMUDict (CMUdict 0.7b subset, offline — see
# voiceclone_sidecar/data/cmudict-mini.dict.gz.README.md). Runtime NEVER
# downloads anything: the file ships with the repository.
_CMUDICT_PATH = Path(__file__).with_name("data") / "cmudict-mini.dict.gz"


@lru_cache(maxsize=1)
def _cmudict() -> dict[str, tuple[str, ...]]:
    d: dict[str, tuple[str, ...]] = {}
    with gzip.open(_CMUDICT_PATH, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            word, phones = line.split(" ", 1)
            d[word] = tuple(phones.split())
    return d


def cmudict_lookup(word: str) -> tuple[str, ...] | None:
    """ARPAbet phones for an English word from the bundled offline CMUDict,
    or None when unknown (the annotation degrades to the original text)."""
    return _cmudict().get(word.lower().strip())

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
    """Parse the canonical annotation block (issue #23 + #56).

    Line forms, all sharing this one entry point:

    - ``汉字=拼音`` (or ``汉字:拼音``) — the Chinese pinyin route;
    - ``词|ARPAbet`` — the issue-#56 English route, explicit phones;
    - ``词=词2`` / ``词=词2|ARPAbet`` / ``词=ARPAbet`` — English key with a
      dictionary-lookup word, an explicit ``word|ARPAbet`` pair, or bare
      ARPAbet tokens.

    Returns ``(annotations, errors)``; errors are user-facing strings. An
    entry is skipped (never half-applied) when it is malformed or fails
    validation. Values are stored raw; the per-grammar ``to_*`` functions
    route them (pinyin vs. CMUDict) by content.
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
        key, _sep, value = m.group(1), m.group(2), m.group(3)
        if _is_pinyin_entry(key, value):
            annotations[key] = value
        elif _is_english_entry(key, value):
            annotations[key.lower()] = value
        else:
            errors.append(f"第 {lineno} 行标注无效：{line.strip()}")
    return annotations, errors


def _is_pinyin_entry(key: str, value: str) -> bool:
    """Single CJK character with a valid pinyin value (canonical #23 form)."""
    return len(key) == 1 and not key.isascii() and bool(_PINYIN_RE.match(value))


def _is_english_entry(key: str, value: str) -> bool:
    if not _ENGLISH_KEY_RE.match(key):
        return False
    phones = value.split("|", 1)[1] if "|" in value else value
    tokens = [t.upper() for t in phones.split()]
    if _is_arpabet_tokens(tokens):
        return True  # explicit ARPAbet (``world|W ER1 L D`` or bare tokens)
    return bool(_ENGLISH_WORD_RE.match(value))  # dictionary-lookup word


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
    expects). Neutral tone (digit 5 / absent) emits a digitless tag — that
    form is UNVERIFIED upstream; all verified examples carry a tone digit."""
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
    pinyin written inline: ``号`` + ``hao4`` -> ``hào``. Only accented
    syllables are recognized upstream — digit forms like ``hao4`` are not."""
    out = text
    for char, pinyin in annotations.items():
        out = out.replace(char, mark_pinyin(pinyin))
    return out


def _voxcpm_phoneme_tag(key: str, value: str, log) -> str | None:
    """Route one annotation entry to a VoxCPM ``{...}`` phoneme tag.

    Routing is by CONTENT (issue #56): a Chinese entry keeps the pinyin form
    (lowercase + tone digit); an English entry goes through the CMUDict
    route — explicit ARPAbet tokens (``world|W ER1 L D`` / bare ``W ER1 L
    D``) pass through uppercased, a plain word is looked up in the bundled
    offline dictionary. Returns None for a bad annotation — the caller
    degrades to the original text.
    """
    if len(key) == 1 and not key.isascii():
        return f"{{{value}}}"  # parse_annotations already validated the pinyin
    phones = value.split("|", 1)[1] if "|" in value else value
    tokens = [t.upper() for t in phones.split()]
    if _is_arpabet_tokens(tokens):
        return "{" + " ".join(tokens) + "}"
    hit = cmudict_lookup(value)
    if hit:
        return "{" + " ".join(hit) + "}"
    if log:
        log(f"发音标注：CMUDict 未收录「{value}」（该词保留原文）")
    return None


def to_voxcpm(text: str, annotations: dict[str, str], log=None) -> str:
    """VoxCPM2 grammar (issue #23 + #56).

    Chinese: the annotated character is replaced by a braced pinyin syllable
    with lowercase tone digit: ``你`` + ``ni3`` -> ``{ni3}``. English: the
    annotated word is replaced by braced ARPAbet phones via the CMUDict
    route (``world|W ER1 L D`` -> ``{W ER1 L D}``). Both share the same
    annotation entry and brace syntax (official phoneme input; only valid
    while the engine's ``normalize`` stays False, which this engine's
    adapter guarantees). Bad annotations degrade to the original text.
    """
    out = text
    for key, value in annotations.items():
        tag = _voxcpm_phoneme_tag(key, value, log)
        if tag is None:
            continue  # degrade: original text stays untouched
        if len(key) == 1 and not key.isascii():
            out = out.replace(key, tag)
        else:
            # English words replace whole-word occurrences only ("world"
            # must never eat "worldwide").
            out = re.sub(rf"\b{re.escape(key)}\b", tag, out, flags=re.IGNORECASE)
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
    if grammar == "voxcpm":
        return to_voxcpm(text, annotations, log)
    raise ValueError(f"unknown pronunciation grammar: {grammar!r}")
