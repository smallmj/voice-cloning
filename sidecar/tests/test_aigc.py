"""AIGC metadata marker unit tests (issue #15): embed / read / replace,
non-WAV refusal, and survival through loudness normalization."""

from __future__ import annotations

import wave

from voiceclone_sidecar import aigc
from voiceclone_sidecar.loudness import normalize_wav_lufs


def _make_wav(path, seconds: float = 0.5, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x10\x20" * int(seconds * rate))


def test_embed_and_read_roundtrip(tmp_path):
    path = tmp_path / "a.wav"
    _make_wav(path)
    assert aigc.embed_aigc_marker(path, {"voice": "小明", "engine": "fake"})
    comment = aigc.read_info_comment(path)
    assert comment is not None
    assert "AI 生成音频" in comment
    assert "AI-generated audio" in comment
    assert "voice=小明" in comment
    assert aigc.has_marker(path)


def test_embed_replaces_previous_marker(tmp_path):
    path = tmp_path / "a.wav"
    _make_wav(path)
    aigc.embed_aigc_marker(path, {"voice": "第一次"})
    aigc.embed_aigc_marker(path, {"voice": "第二次"})
    comment = aigc.read_info_comment(path)
    assert "第二次" in comment
    assert "第一次" not in comment


def test_marked_file_still_parses_as_wav(tmp_path):
    path = tmp_path / "a.wav"
    _make_wav(path, seconds=1.0)
    aigc.embed_aigc_marker(path, {"voice": "v"})
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000
        assert w.getnframes() == 16000


def test_non_wav_is_refused(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"not a riff file")
    assert aigc.embed_aigc_marker(path, {}) is False
    assert aigc.read_info_comment(path) is None
    assert aigc.has_marker(path) is False


def test_normalization_keeps_marker(tmp_path):
    src = tmp_path / "src.wav"
    dst = tmp_path / "norm.wav"
    _make_wav(src, seconds=2.0)
    aigc.embed_aigc_marker(src, {"voice": "小明", "engine": "fake"})
    normalize_wav_lufs(src, dst)
    # The derived copy carries the disclosure its source had (issue #15).
    assert aigc.has_marker(dst)


def test_unmarked_source_normalizes_without_marker(tmp_path):
    src = tmp_path / "src.wav"
    dst = tmp_path / "norm.wav"
    _make_wav(src, seconds=2.0)
    normalize_wav_lufs(src, dst)
    assert not aigc.has_marker(dst)


def test_copy_info_chunks_moves_disclosure(tmp_path):
    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"
    _make_wav(src)
    _make_wav(dst)
    aigc.embed_aigc_marker(src, {"voice": "v"})
    assert aigc.copy_info_chunks(src, dst)
    assert aigc.has_marker(dst)
