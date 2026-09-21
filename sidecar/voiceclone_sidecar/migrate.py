"""One-time migration: JSON indexes → SQLite (ADR-0017 decision 5).

On startup, if the library database holds no index rows and the legacy JSON
index files exist, the originals are FIRST copied into
``<data_root>/migration-backup-<utcstamp>/`` and only then imported. The
original files are removed only after the import committed successfully —
a failed migration never deletes the user's data. A corrupt or unparseable
legacy index fails the migration loudly and the originals stay untouched —
the "unparseable JSON = empty library" behaviour this ADR eliminates must
not sneak back in through the migration path itself.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .db import (
    JSON_INDEX_NAMES,
    T_APP_STATE,
    T_COMPARE,
    T_GENERATIONS,
    T_REGRESSION,
    T_SETTINGS,
    T_VOICES,
    Database,
    DatabaseError,
    library_db_path,
)

MIGRATION_MARKER = "migrated_at"

_BACKUP_PREFIX = "migration-backup-"


def migration_backup_dir(data_root: Path) -> Path | None:
    """The most recent migration backup directory, if any (rollback aid)."""
    root = Path(data_root)
    if not root.is_dir():
        return None
    backups = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(_BACKUP_PREFIX))
    return backups[-1] if backups else None


def _load_json(path: Path) -> dict:
    """Parse a legacy index; a corrupt file fails the migration loudly.

    The originals are never deleted on failure, so aborting here loses
    nothing — silently skipping a corrupt index could (a stray control
    character in voices.json) and would then delete it as "migrated".
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise DatabaseError(f"旧索引文件无法解析，迁移中止（原文件未改动）：{path.name}：{exc}") from exc
    if not isinstance(raw, dict):
        raise DatabaseError(f"旧索引文件内容不是 JSON 对象，迁移中止（原文件未改动）：{path.name}")
    return raw


def _import_voice_records(db: Database, path: Path) -> int:
    raw = _load_json(path)
    if raw is None:
        return 0
    n = 0
    for record in raw.get("voices", []):
        if not (isinstance(record, dict) and record.get("id")):
            continue
        # Records written before issue #11 predate the origin field; a voice
        # without one is a reference-audio clone.
        record.setdefault("origin", "cloned")
        record.setdefault("design", None)
        db.put_payload(T_VOICES, record["id"], record.get("created_at", ""), record)
        n += 1
    return n


def _import_generation_records(db: Database, path: Path) -> int:
    raw = _load_json(path)
    if raw is None:
        return 0
    n = 0
    for record in raw.get("generations", []):
        if not (isinstance(record, dict) and record.get("id")):
            continue
        db.put_payload(
            T_GENERATIONS,
            record["id"],
            record.get("created_at", ""),
            record,
            {
                "engine_id": record.get("engine_id") or "",
                "voice_id": record.get("voice_id") or "",
                "status": record.get("status") or "",
                "text": record.get("text") or "",
                "voice_name": record.get("voice_name") or "",
            },
        )
        n += 1
    return n


def _import_session_records(db: Database, table: str, path: Path) -> int:
    raw = _load_json(path)
    if raw is None:
        return 0
    n = 0
    key = "sessions"
    for session in raw.get(key, []):
        if not (isinstance(session, dict) and session.get("id")):
            continue
        db.put_payload(
            table,
            session["id"],
            session.get("created_at", ""),
            session,
            {"status": session.get("status") or ""} if table == T_REGRESSION else None,
        )
        n += 1
    return n


def _import_kv_file(db: Database, table: str, path: Path) -> int:
    raw = _load_json(path)
    if raw is None:
        return 0
    for key, value in raw.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue
        db.put_kv(table, str(key), value)
    return len(raw)


def _import_compare_records(db: Database, path: Path) -> int:
    return _import_session_records(db, T_COMPARE, path)


def _import_regression_records(db: Database, path: Path) -> int:
    return _import_session_records(db, T_REGRESSION, path)


def _import_settings_file(db: Database, path: Path) -> int:
    return _import_kv_file(db, T_SETTINGS, path)


def _import_app_state_file(db: Database, path: Path) -> int:
    return _import_kv_file(db, T_APP_STATE, path)


# (table, file name, importer) — importer callables are looked up fresh at
# call time so tests can monkeypatch them via the module namespace.
_IMPORTERS = (
    (T_VOICES, "voices.json", _import_voice_records),
    (T_GENERATIONS, "generations.json", _import_generation_records),
    (T_COMPARE, "compare.json", _import_compare_records),
    (T_REGRESSION, "regression.json", _import_regression_records),
    (T_SETTINGS, "settings.json", _import_settings_file),
    (T_APP_STATE, "app_state.json", _import_app_state_file),
)


def ensure_migrated(data_root: Path) -> str:
    """Migrate legacy JSON indexes into the SQLite library, once.

    Returns ``"migrated"``, ``"already"`` (marker present or database
    already populated) or ``"fresh"`` (nothing to migrate). Raises on an
    unexpected import failure — after rolling back, the original JSON
    files are guaranteed untouched.
    """
    data_root = Path(data_root)
    db_path = library_db_path(data_root)

    if db_path.exists() and not _db_is_blank_or_absent_ok(db_path):
        # The database file exists but cannot be opened: loud failure, the
        # user must see a broken index instead of a silently empty library.
        raise DatabaseError(f"索引数据库无法打开：{db_path}")

    db = Database(db_path, migrate=False)
    try:
        if db.get_kv("meta", MIGRATION_MARKER) is not None:
            return "already"
        if not db.is_empty():
            # A populated database without a marker predates the marker or
            # was restored from a snapshot — either way there is nothing to
            # migrate from JSON.
            return "already"

        present = [
            (table, data_root / name, importer)
            for table, name, importer in _IMPORTERS
            if (data_root / name).exists()
        ]
        if not present:
            return "fresh"

        # Backup FIRST (ADR-0017 decision 5): originals are copied, never
        # moved, so a failure below cannot lose data.
        backup_dir = data_root / f"{_BACKUP_PREFIX}{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        for _, path, _ in present:
            shutil.copyfile(path, backup_dir / path.name)

        try:
            with db.tx():
                counts = {}
                for table, path, importer in present:
                    counts[table] = importer(db, path)
                db.put_kv("meta", MIGRATION_MARKER, {
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "backup_dir": backup_dir.name,
                    "counts": counts,
                })
        except Exception:
            # Rollback above already undid the import; originals remain in
            # place and the marker was never written. Re-raise loudly.
            shutil.rmtree(backup_dir, ignore_errors=True)
            raise

        # Committed: the originals live on in the backup dir, so the data
        # root keeps only the database as the readable index.
        for _, path, _ in present:
            path.unlink(missing_ok=True)
        return "migrated"
    finally:
        db.close()


def _db_is_blank_or_absent_ok(db_path: Path) -> bool:
    """True when the database either opens fine or doesn't exist yet."""
    if not db_path.exists():
        return True
    try:
        probe = Database(db_path, migrate=False)
        probe.close()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not blank-or-absent ok"
        return False


def json_indexes_present(data_root: Path) -> list[str]:
    return [n for n in JSON_INDEX_NAMES if (Path(data_root) / n).exists()]
