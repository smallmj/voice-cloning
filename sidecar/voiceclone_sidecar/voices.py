"""Voices: the engine-independent sound identity (ADR-0001).

A voice owns its metadata (name, description, avatar) and its reference
sample. The reference sample is the source of truth; a voice's realisation
on a specific engine is an **engine binding** — a rebuildable cache that
records how the reference was handed to that engine. Deleting a voice
deletes every binding and every local file.

Storage layout (all under the data dir):

    <data>/voices.json            the index (atomic writes)
    <data>/voices/<id>/ref.<ext>  the reference sample (source of truth)
    <data>/voices/<id>/avatar.<ext>  optional avatar image
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path

from .storage import write_json_atomic

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
AVATAR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_AVATAR_BYTES = 5 * 1024 * 1024
MIN_DURATION_SECONDS = 3.0
MAX_DURATION_SECONDS = 120.0
MAX_NAME_CHARS = 100
MAX_DESCRIPTION_CHARS = 500

AUDIO_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
}
AVATAR_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class VoiceValidationError(ValueError):
    """Raised when an upload fails a hard check; message is user-facing."""


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_audio_duration(path: Path) -> float:
    """Return the audio duration in seconds.

    WAV is probed with the stdlib (no external dependency). Every other
    accepted format needs ffprobe on PATH; without it the duration is
    unknowable, so the upload fails with an explicit reason instead of
    silently skipping the check.
    """
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
        except wave.Error as exc:
            raise VoiceValidationError(f"WAV 文件无法解析：{exc}") from exc
        if rate <= 0:
            raise VoiceValidationError("WAV 文件采样率异常，无法确定时长")
        return frames / rate

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise VoiceValidationError(
            f"无法确定 {path.suffix[1:].upper()} 音频时长（该格式需要系统安装 ffprobe）；"
            "请上传 WAV 文件或安装 ffmpeg 后重试"
        )
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.SubprocessError, ValueError) as exc:
        raise VoiceValidationError(f"音频文件无法解析：{exc}") from exc


class VoiceStore:
    """Persistence for voices and their engine bindings."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "voices"
        self.index_path = self.data_dir / "voices.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self._voices: dict[str, dict] = {}
        # Mutations come from both request handlers and the synthesis worker
        # thread — serialize them so voices.json is never written torn.
        self._lock = threading.Lock()
        self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for record in raw.get("voices", []):
            if isinstance(record, dict) and record.get("id"):
                self._voices[record["id"]] = record

    def _save(self) -> None:
        write_json_atomic(self.index_path, {"voices": list(self._voices.values())})

    # -- CRUD ----------------------------------------------------------------

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(v) for v in self._voices.values()]

    def get(self, voice_id: str) -> dict | None:
        with self._lock:
            record = self._voices.get(voice_id)
            return dict(record) if record else None

    def reference_path(self, voice_id: str) -> Path:
        record = self._voices[voice_id]
        return self.root / voice_id / record["reference"]["filename"]

    def avatar_path(self, voice_id: str) -> Path | None:
        record = self._voices[voice_id]
        if not record.get("avatar"):
            return None
        return self.root / voice_id / record["avatar"]["filename"]

    def create(
        self, name: str, description: str, audio_file: Path, avatar_file: Path | None = None
    ) -> dict:
        name = (name or "").strip()
        if not name:
            raise VoiceValidationError("音色名称不能为空")
        if len(name) > MAX_NAME_CHARS:
            raise VoiceValidationError(f"音色名称过长（最多 {MAX_NAME_CHARS} 字符）")
        if description and len(description) > MAX_DESCRIPTION_CHARS:
            raise VoiceValidationError(f"说明过长（最多 {MAX_DESCRIPTION_CHARS} 字符）")

        ext = audio_file.suffix.lower()
        if ext not in AUDIO_EXTENSIONS:
            allowed = ", ".join(sorted(e[1:] for e in AUDIO_EXTENSIONS))
            raise VoiceValidationError(f"不支持的音频格式 {ext or '(无后缀)'}；支持：{allowed}")
        size = audio_file.stat().st_size
        if size == 0:
            raise VoiceValidationError("参考音频是空文件")
        if size > MAX_AUDIO_BYTES:
            raise VoiceValidationError(
                f"参考音频过大（{size / 1024 / 1024:.1f} MB，上限 {MAX_AUDIO_BYTES // 1024 // 1024} MB）"
            )
        duration = probe_audio_duration(audio_file)
        if duration < MIN_DURATION_SECONDS:
            raise VoiceValidationError(
                f"参考音频太短（{duration:.1f} 秒，至少 {MIN_DURATION_SECONDS:g} 秒）"
            )
        if duration > MAX_DURATION_SECONDS:
            raise VoiceValidationError(
                f"参考音频太长（{duration:.1f} 秒，上限 {MAX_DURATION_SECONDS:g} 秒）"
            )

        voice_id = uuid.uuid4().hex
        voice_dir = self.root / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        ref_name = f"ref{ext}"
        shutil.copyfile(audio_file, voice_dir / ref_name)

        record: dict = {
            "id": voice_id,
            "name": name,
            "description": description or "",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reference": {
                "filename": ref_name,
                "format": ext[1:],
                "duration_seconds": round(duration, 3),
                "size_bytes": size,
                "sha256": _file_sha256(voice_dir / ref_name),
            },
            "avatar": None,
            "bindings": {},
        }

        if avatar_file is not None and avatar_file.stat().st_size > 0:
            avatar = self._store_avatar(voice_id, avatar_file)
            record["avatar"] = avatar

        with self._lock:
            self._voices[voice_id] = record
            self._save()
        return dict(record)

    def _store_avatar(self, voice_id: str, avatar_file: Path) -> dict:
        ext = avatar_file.suffix.lower()
        if ext not in AVATAR_EXTENSIONS:
            allowed = ", ".join(sorted(e[1:] for e in AVATAR_EXTENSIONS))
            raise VoiceValidationError(f"不支持的头像格式 {ext}；支持：{allowed}")
        size = avatar_file.stat().st_size
        if size > MAX_AVATAR_BYTES:
            raise VoiceValidationError(
                f"头像过大（{size / 1024 / 1024:.1f} MB，上限 {MAX_AVATAR_BYTES // 1024 // 1024} MB）"
            )
        filename = f"avatar{ext}"
        shutil.copyfile(avatar_file, self.root / voice_id / filename)
        return {"filename": filename, "format": ext[1:], "size_bytes": size}

    def update(self, voice_id: str, name: str | None = None,
               description: str | None = None) -> dict:
        record = self._voices.get(voice_id)
        if record is None:
            raise KeyError(voice_id)
        with self._lock:
            if name is not None:
                name = name.strip()
                if not name:
                    raise VoiceValidationError("音色名称不能为空")
                if len(name) > MAX_NAME_CHARS:
                    raise VoiceValidationError(f"音色名称过长（最多 {MAX_NAME_CHARS} 字符）")
                record["name"] = name
            if description is not None:
                if len(description) > MAX_DESCRIPTION_CHARS:
                    raise VoiceValidationError(f"说明过长（最多 {MAX_DESCRIPTION_CHARS} 字符）")
                record["description"] = description
            self._save()
        return dict(record)

    def set_avatar(self, voice_id: str, avatar_file: Path) -> dict:
        with self._lock:
            record = self._voices.get(voice_id)
            if record is None:
                raise KeyError(voice_id)
            old = record.get("avatar")
            avatar = self._store_avatar(voice_id, avatar_file)
            if old and old["filename"] != avatar["filename"]:
                try:
                    (self.root / voice_id / old["filename"]).unlink()
                except OSError:
                    pass
            record["avatar"] = avatar
            self._save()
            return dict(record)

    def delete(self, voice_id: str) -> None:
        """Remove the voice, all of its bindings and all of its local files."""
        with self._lock:
            if voice_id not in self._voices:
                raise KeyError(voice_id)
            del self._voices[voice_id]
            self._save()
        shutil.rmtree(self.root / voice_id, ignore_errors=True)

    # -- engine bindings -------------------------------------------------------

    def set_transcript(self, voice_id: str, text: str) -> dict:
        """Store the reference's transcript on the voice record.

        The transcript is metadata about the reference, never a modification
        of the audio itself (issue #8: the app does not touch user material).
        """
        with self._lock:
            record = self._voices.get(voice_id)
            if record is None:
                raise KeyError(voice_id)
            record["reference"]["transcript"] = text
            self._save()
        return dict(record)

    def bind(self, voice_id: str, engine_id: str, status: str = "ready",
             extra: dict | None = None) -> dict:
        """Create or refresh the voice's binding on one engine.

        Bindings are a cache keyed by the reference's checksum: if the
        reference ever changes, a stale binding is simply replaced here —
        nothing in the model assumes permanence (ADR-0001).
        """
        with self._lock:
            record = self._voices.get(voice_id)
            if record is None:
                raise KeyError(voice_id)
            binding = {
                "status": status,
                "reference_sha256": record["reference"]["sha256"],
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if extra:
                binding.update(extra)
            record["bindings"][engine_id] = binding
            self._save()
        return dict(binding)
