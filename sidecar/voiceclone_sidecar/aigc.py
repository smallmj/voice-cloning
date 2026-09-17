"""AI-generated-content (AIGC) metadata marker for synthesized audio (issue #15).

Every audio artifact this app synthesizes must carry a machine-readable
statement that it is AI-generated — embedded IN the file, so the marker
survives download, sharing and re-encoding pipelines that keep RIFF
metadata. WAV files get a RIFF ``LIST``/``INFO`` chunk with:

- ``ICMT`` — a human-readable comment naming VoiceClone Workbench, the
  voice/engine/model lineage and the "AI 生成音频" statement;
- ``ISFT`` — the producing software name + version.

Deliberately dependency-free (like diagnostics.py and loudness.py): pure
stdlib ``pathlib``/``struct`` over the RIFF container. Non-WAV input is
reported honestly (``False``) rather than half-marked.

The marker is re-attached wherever the sidecar rewrites a marked file
(loudness normalization copies the source's INFO chunks through — see
``loudness.normalize_wav_lufs``), so a derived artifact never loses the
disclosure its source carried.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

from ._version import SIDECAR_VERSION as SOFTWARE_VERSION

SOFTWARE = "VoiceClone Workbench"

# Marker chunks we write; ``copy_info_chunks`` moves exactly these.
MARKER_IDS = (b"ICMT", b"ISFT")


def build_comment(fields: dict[str, str]) -> str:
    """One-line ICMT payload: fixed AIGC statement + lineage fields."""
    lineage = "; ".join(f"{k}={v}" for k, v in fields.items() if v)
    return (
        f"AI 生成音频 / AI-generated audio. Produced by {SOFTWARE}"
        + (f" v{SOFTWARE_VERSION}" if SOFTWARE_VERSION else "")
        + (f"; {lineage}" if lineage else "")
    )


def _chunk_bytes(cid: bytes, payload: bytes) -> bytes:
    return cid + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")


def _info_chunk_bytes(comment: str) -> bytes:
    sub = _chunk_bytes(b"ICMT", comment.encode("utf-8")) + _chunk_bytes(
        b"ISFT", SOFTWARE.encode("utf-8")
    )
    return _chunk_bytes(b"LIST", b"INFO" + sub)


def is_wav(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"RIFF" and f.read(8)[4:8] == b"WAVE"
    except OSError:
        return False


def _split_chunks(raw: bytes) -> list[tuple[bytes, bytes]]:
    """Parse ``(id, payload)`` pairs from the chunk stream after 'WAVE'."""
    chunks: list[tuple[bytes, bytes]] = []
    pos = 0
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        size = struct.unpack("<I", raw[pos + 4 : pos + 8])[0]
        payload = raw[pos + 8 : pos + 8 + size]
        chunks.append((cid, payload))
        pos += 8 + size + (size % 2)
    return chunks


def _rewrite_wav(path: Path, chunks: list[tuple[bytes, bytes]]) -> None:
    """Atomically write ``chunks`` back into the RIFF/WAVE ``path``.

    Shared by embed and copy paths: tmp file + ``os.replace`` (repo atomic-
    write convention), and the RIFF size field is recomputed here so every
    caller gets it right.
    """
    rebuilt = bytearray(path.read_bytes()[:12])
    for cid, payload in chunks:
        rebuilt += _chunk_bytes(cid, payload)
    rebuilt[4:8] = struct.pack("<I", len(rebuilt) - 8)
    tmp = path.with_name(path.name + f".aigc-{os.getpid()}")
    tmp.write_bytes(bytes(rebuilt))
    os.replace(tmp, path)


def embed_aigc_marker(path: Path, fields: dict[str, str]) -> bool:
    """Insert (or replace) the AIGC INFO chunks of a WAV file in place.

    Returns ``True`` when written, ``False`` when the file is not a
    RIFF/WAVE container (nothing is guessed, nothing is broken).
    """
    path = Path(path)
    if not is_wav(path):
        return False
    body = path.read_bytes()[12:]  # after RIFF header + 'WAVE'
    marker = _info_chunk_bytes(build_comment(fields))
    # The LIST chunk payload is 'INFO' + its subchunks; re-add as one chunk.
    list_payload = marker[8 : 8 + struct.unpack("<I", marker[4:8])[0]]
    kept = [
        (cid, payload)
        for cid, payload in _split_chunks(body)
        if cid not in MARKER_IDS and not (cid == b"LIST" and payload[:4] == b"INFO")
    ]
    # Place the marker right after 'fmt ' when present (players read INFO
    # anywhere; keeping audio data first is the least surprising layout).
    fmt_index = next((i for i, (cid, _) in enumerate(kept) if cid == b"fmt "), -1)
    kept.insert(fmt_index + 1, (b"LIST", list_payload))
    _rewrite_wav(path, kept)
    return True


def read_info_comment(path: Path) -> str | None:
    """Return the ICMT payload of a WAV file, or ``None`` when absent."""
    path = Path(path)
    if not is_wav(path):
        return None
    # ICMT lives as a subchunk inside LIST/INFO — descend into LIST too.
    stack = list(_split_chunks(Path(path).read_bytes()[12:]))
    while stack:
        cid, payload = stack.pop()
        if cid == b"ICMT":
            return payload.decode("utf-8", errors="replace")
        if cid == b"LIST":
            stack.extend(_split_chunks(payload[4:]))
    return None


def has_marker(path: Path) -> bool:
    """A file is considered marked when its ICMT carries the AIGC statement."""
    comment = read_info_comment(path)
    return bool(comment) and "AI-generated audio" in (comment or "")


def copy_info_chunks(src: Path, dst: Path) -> bool:
    """Move the INFO chunks of ``src`` onto the rewritten WAV ``dst``.

    Used after any sidecar-side rewrite (loudness normalization) so derived
    artifacts keep the disclosure their source carried. Replaces any INFO
    chunks ``dst`` already has. Returns ``False`` when ``src`` carries none.
    """
    src, dst = Path(src), Path(dst)
    if not is_wav(src) or not is_wav(dst):
        return False
    info = [
        (cid, payload)
        for cid, payload in _split_chunks(src.read_bytes()[12:])
        if cid == b"LIST" and payload[:4] == b"INFO"
    ]
    if not info:
        return False
    kept = [
        (cid, payload)
        for cid, payload in _split_chunks(dst.read_bytes()[12:])
        if not (cid == b"LIST" and payload[:4] == b"INFO")
    ]
    _rewrite_wav(dst, kept + info)
    return True
