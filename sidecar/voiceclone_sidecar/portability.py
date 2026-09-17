"""Portability: voice packages + whole-library backup/restore (issue #14).

Two manual actions, deliberately nothing automatic (ADR-0005: no remote
inference, no cross-machine sync):

**Voice package** — one voice exported as a ``.zip`` a user can hand to
someone else: the voice record (metadata, design prompt, reference meta)
plus the reference sample and avatar files. The exporter's engine bindings
travel inside the record as information only; on import they are dropped —
a binding is a rebuildable cache (ADR-0001), and a binding minted on
someone else's machine is meaningless here, so the importing user rebinds
on demand.

**Library backup / restore** — a ``.zip`` of the entire data directory.
The data dir is self-contained by construction: JSON indexes (voices,
generations, compare sessions, settings) and all audio artifacts live
under one portable root. BYOK API keys are excluded by design — they
never touch disk (ADR-0003) and stay in the OS key store.

Zip layouts::

    voice package:            library backup:
      manifest.json             manifest.json
      files/<ref|avatar>...     <every file of the data dir>
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

PACKAGE_KIND = "voice-clone-package"
BACKUP_KIND = "voice-clone-library-backup"
FORMAT_VERSION = 1


class PortabilityError(ValueError):
    """Raised when an export/import/backup/restore fails; user-facing."""


def _manifest(kind: str, extra: dict | None = None) -> dict:
    manifest = {
        "kind": kind,
        "version": FORMAT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        manifest.update(extra)
    return manifest


def _read_manifest(zf: zipfile.ZipFile, expected_kind: str) -> dict:
    try:
        raw = json.loads(zf.read("manifest.json").decode("utf-8"))
    except KeyError as exc:
        raise PortabilityError("压缩包缺少 manifest.json，不是有效的导出文件") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PortabilityError(f"manifest.json 无法解析：{exc}") from exc
    if not isinstance(raw, dict) or raw.get("kind") != expected_kind:
        raise PortabilityError("压缩包类型不匹配，不是有效的导出文件")
    if raw.get("version") != FORMAT_VERSION:
        raise PortabilityError(
            f"压缩包格式版本 {raw.get('version')} 不受支持（当前 {FORMAT_VERSION}）"
        )
    return raw


def _safe_member_name(name: str) -> str:
    # Zip members are user-supplied material: reject anything that could
    # escape the extraction root (zip-slip).
    if not name or name.startswith("/") or "\\" in name or ".." in Path(name).parts:
        raise PortabilityError(f"压缩包含有不安全的路径：{name!r}")
    return name


# -- voice package -----------------------------------------------------------


def export_voice_package(record: dict, voice_dir: Path, out_path: Path) -> Path:
    """Zip one voice: its record (bindings kept as information) + files.

    The importer decides what to do with bindings — the package never
    assumes they are usable on another machine.
    """
    voice = dict(record)
    reference = voice.get("reference") or {}
    avatar = voice.get("avatar") or {}

    files: list[Path] = []
    if reference.get("filename"):
        files.append(voice_dir / reference["filename"])
    if avatar.get("filename"):
        files.append(voice_dir / avatar["filename"])
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        raise PortabilityError(f"音色缺少文件，无法导出：{', '.join(missing)}")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(
                _manifest(PACKAGE_KIND, {"voice": voice}),
                ensure_ascii=False, indent=2,
            ))
            for f in files:
                zf.write(f, f"files/{f.name}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return out_path


def import_voice_package(zip_path: Path, voice_store) -> dict:
    """Import a voice package. Returns the new voice record.

    The imported voice always gets a fresh id and empty bindings; the user
    rebuilds engine bindings on demand (acceptance criteria).
    """
    with tempfile.TemporaryDirectory(prefix="voicepkg-") as td:
        td_path = Path(td)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                manifest = _read_manifest(zf, PACKAGE_KIND)
                voice = manifest.get("voice")
                if not isinstance(voice, dict):
                    raise PortabilityError("manifest.json 缺少音色数据")
                for info in zf.infolist():
                    name = _safe_member_name(info.filename)
                    if not name.startswith("files/"):
                        continue
                    target = td_path / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with open(target, "wb") as f:
                        f.write(zf.read(info))
        except zipfile.BadZipFile as exc:
            raise PortabilityError(f"压缩包无法解析：{exc}") from exc
        return voice_store.import_record(voice, td_path / "files")


# -- library backup / restore -------------------------------------------------


def create_library_backup(data_dir: Path, out_path: Path) -> Path:
    """Zip the whole data directory (indexes + audio artifacts) with a manifest."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise PortabilityError("数据目录不存在，无法备份")
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(
                _manifest(BACKUP_KIND, {"notes": [
                    "BYOK API keys are not included: they live in the OS key store.",
                ]}),
                ensure_ascii=False, indent=2,
            ))
            for path in sorted(data_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix == ".tmp":
                    continue
                zf.write(path, path.relative_to(data_dir).as_posix())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return out_path


def restore_library_backup(zip_path: Path, data_dir: Path) -> dict:
    """Replace the data directory with the backup's contents and return counts.

    The extraction lands in a staging dir next to the data dir; only after
    the archive validates completely is the live directory swapped out, so a
    broken archive can never leave the library half-restored. Callers must
    reload their stores afterwards.
    """
    data_dir = Path(data_dir)
    staging = data_dir.parent / f"{data_dir.name}.restore-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True, exist_ok=True)
    old = data_dir.parent / f"{data_dir.name}.restore-old-{uuid.uuid4().hex[:8]}"
    try:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                _read_manifest(zf, BACKUP_KIND)
                for info in zf.infolist():
                    name = _safe_member_name(info.filename)
                    if name == "manifest.json":
                        continue
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with open(target, "wb") as f:
                        f.write(zf.read(info))
        except zipfile.BadZipFile as exc:
            raise PortabilityError(f"备份包无法解析：{exc}") from exc

        restored = {"voices": 0, "generations": 0}
        # A backup whose indexes no longer parse cannot honestly claim a
        # complete restore — fail loudly instead of reporting zero voices.
        voices_index = staging / "voices.json"
        if voices_index.is_file():
            try:
                restored["voices"] = len(json.loads(
                    voices_index.read_text(encoding="utf-8")).get("voices", []))
            except (json.JSONDecodeError, OSError) as exc:
                raise PortabilityError(f"备份内的 voices.json 无法解析：{exc}") from exc
        generations_index = staging / "generations.json"
        if generations_index.is_file():
            try:
                restored["generations"] = len(json.loads(
                    generations_index.read_text(encoding="utf-8")).get("generations", []))
            except (json.JSONDecodeError, OSError) as exc:
                raise PortabilityError(f"备份内的 generations.json 无法解析：{exc}") from exc

        # Swap: live dir -> old, staging -> live. If anything below fails we
        # roll back; the swap itself is two renames on the same filesystem.
        if data_dir.exists():
            data_dir.rename(old)
        try:
            staging.rename(data_dir)
        except OSError:
            if old.exists():
                old.rename(data_dir)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
    return restored
