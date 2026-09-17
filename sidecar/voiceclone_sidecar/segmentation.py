"""Long-text segmentation and segment-audio concatenation (issue #13).

Two pure helpers, no sidecar state:

``split_text``      — cut an over-length script into engine-sized segments,
                      preferring sentence boundaries so a segment never ends
                      mid-sentence unless a single sentence alone exceeds the
                      limit (then it is hard-split at clause/fixed size).
``concat_wav_files`` — stitch the per-segment WAVs into ONE complete audio
                      file. Segments come from the same engine and normally
                      share the WAV format; a differing sample rate is
                      resampled (linear interpolation) instead of failing the
                      whole job, while a differing channel count or sample
                      width is a hard error (there is no honest way to guess).
"""

from __future__ import annotations

import array
import wave
from pathlib import Path

# Sentence-ending punctuation. The delimiter stays attached to the sentence
# before it so re-joined text round-trips byte-for-byte.
_SENTENCE_ENDS = "。！？!?…\n"
# Clause-level fallback for a single over-long sentence.
_CLAUSE_BREAKS = "，,、；;：:（）()\"“”"


def _split_sentences(text: str) -> list[str]:
    """Split into sentences, each keeping its trailing punctuation."""
    sentences: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _SENTENCE_ENDS:
            sentences.append("".join(buf))
            buf.clear()
    if buf:
        sentences.append("".join(buf))
    return [s for s in sentences if s]


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Split one over-long sentence at clause marks, then at fixed size."""
    pieces: list[str] = []
    buf = ""
    for ch in sentence:
        buf += ch
        if len(buf) >= max_chars and ch in _CLAUSE_BREAKS:
            pieces.append(buf)
            buf = ""
    if buf:
        pieces.append(buf)
    # Any piece still over the limit (no clause marks at all): fixed-size cut.
    out: list[str] = []
    for piece in pieces:
        while len(piece) > max_chars:
            out.append(piece[:max_chars])
            piece = piece[max_chars:]
        if piece:
            out.append(piece)
    return out


def split_text(text: str, max_chars: int) -> list[str]:
    """Split ``text`` into segments of at most ``max_chars`` characters.

    Sentence boundaries are preferred; the concatenation of the returned
    segments equals the input exactly (round-trip guarantee the UI relies
    on when it previews the plan). ``max_chars`` < 1 is rejected.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if len(text) <= max_chars:
        return [text]
    segments: list[str] = []
    buf = ""
    for sentence in _split_sentences(text):
        if len(sentence) > max_chars:
            # Flush what we have, then hard-split the monster sentence.
            if buf:
                segments.append(buf)
                buf = ""
            segments.extend(_hard_split(sentence, max_chars))
            continue
        if buf and len(buf) + len(sentence) > max_chars:
            segments.append(buf)
            buf = sentence
        else:
            buf += sentence
    if buf:
        segments.append(buf)
    return segments


def _resample_linear(samples: array.array, src_rate: int, dst_rate: int) -> array.array:
    if src_rate == dst_rate or not samples:
        return samples
    # Naive linear interpolation: enough to keep concatenation playable when
    # one engine returns a slightly different rate mid-job; never used as a
    # general-purpose resampler.
    duration = len(samples) / src_rate
    dst_len = max(1, int(duration * dst_rate))
    out = array.array("h", bytes(2 * dst_len))
    step = (len(samples) - 1) / max(1, dst_len - 1)
    for i in range(dst_len):
        pos = i * step
        i0 = int(pos)
        i1 = min(i0 + 1, len(samples) - 1)
        frac = pos - i0
        out[i] = int(samples[i0] * (1 - frac) + samples[i1] * frac)
    return out


def concat_wav_files(paths: list[Path], out_path: Path) -> int:
    """Concatenate WAV files into ``out_path``; returns the sample rate.

    All inputs must share channels and sample width (16-bit mono in
    practice). Differing sample rates are resampled to the first file's
    rate. Returns without writing anything when ``paths`` is empty.
    """
    if not paths:
        raise ValueError("no segments to concatenate")

    with wave.open(str(paths[0]), "rb") as first:
        channels = first.getnchannels()
        sampwidth = first.getsampwidth()
        rate = first.getframerate()
        frames: list[array.array] = [array.array("h", first.readframes(first.getnframes()))]

    for path in paths[1:]:
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != channels or w.getsampwidth() != sampwidth:
                raise ValueError(
                    f"分段音频格式不一致，无法拼接：{path.name} 的声道数/位宽与首段不同"
                )
            data = array.array("h", w.readframes(w.getnframes()))
            frames.append(_resample_linear(data, w.getframerate(), rate))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        for data in frames:
            w.writeframes(data.tobytes())
    return rate
