"""One-time JSON→SQLite migration (ADR-0017 decision 5, issue #24)."""

from __future__ import annotations

import json

import pytest

from voiceclone_sidecar import migrate
from voiceclone_sidecar.db import (
    T_COMPARE,
    T_GENERATIONS,
    T_REGRESSION,
    T_VOICES,
    Database,
    library_db_path,
)
from voiceclone_sidecar.migrate import MIGRATION_MARKER, ensure_migrated


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def sample_indexes(root):
    write_json(root / "voices.json", {"voices": [
        {"id": "v1", "name": "老王", "created_at": "2026-01-01T00:00:00Z"},
        # Pre-#11 record without origin/design fields.
        {"id": "v2", "name": "老李", "created_at": "2026-01-02T00:00:00Z"},
    ]})
    write_json(root / "generations.json", {"generations": [
        {"id": "g1", "text": "你好世界", "voice_id": "v1", "engine_id": "fake",
         "status": "ok", "voice_name": "老王", "created_at": "2026-02-01T00:00:00Z"},
    ]})
    write_json(root / "compare.json", {"sessions": [{"id": "c1", "created_at": "t"}]})
    write_json(root / "regression.json", {"sessions": [
        {"id": "r1", "status": "completed", "created_at": "t"}]})
    write_json(root / "settings.json", {
        "engines": {"fake": {"env": {"A": "1"}}},
        "transcription_provider": "local",
    })
    write_json(root / "app_state.json", {"ui_prefs": {"theme": "dark"}})


def test_migration_imports_every_index_into_sqlite(tmp_path):
    sample_indexes(tmp_path)
    assert ensure_migrated(tmp_path) == "migrated"

    db = Database(library_db_path(tmp_path), migrate=False)
    try:
        assert db.count(T_VOICES) == 2
        assert db.get_payload(T_VOICES, "v2")["origin"] == "cloned"
        assert db.get_payload(T_VOICES, "v2")["design"] is None
        assert db.get_payload(T_GENERATIONS, "g1")["text"] == "你好世界"
        assert db.get_payload("compare_sessions", "c1")["id"] == "c1"
        assert db.get_payload("regression_sessions", "r1")["status"] == "completed"
        assert db.get_kv("settings", "engines") == {"fake": {"env": {"A": "1"}}}
        assert db.get_kv("settings", "transcription_provider") == "local"
        assert db.get_kv("app_state", "ui_prefs") == {"theme": "dark"}
        assert db.get_kv("meta", MIGRATION_MARKER)["counts"]["voices"] == 2
    finally:
        db.close()


def test_migration_backs_up_originals_then_removes_them_from_data_root(tmp_path):
    sample_indexes(tmp_path)
    ensure_migrated(tmp_path)

    backup = migrate.migration_backup_dir(tmp_path)
    assert backup is not None and backup.name.startswith("migration-backup-")
    assert json.loads((backup / "voices.json").read_text(encoding="utf-8"))["voices"][0]["id"] == "v1"
    assert (backup / "settings.json").is_file()

    # Success: the data root no longer carries the legacy files.
    assert not (tmp_path / "voices.json").exists()
    assert not (tmp_path / "generations.json").exists()


def test_migration_is_idempotent(tmp_path):
    sample_indexes(tmp_path)
    assert ensure_migrated(tmp_path) == "migrated"
    assert ensure_migrated(tmp_path) == "already"
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("migration-backup-")]
    assert len(backups) == 1


def test_populated_database_skips_migration_even_without_marker(tmp_path):
    db = Database(library_db_path(tmp_path), migrate=False)
    db.put_payload(T_VOICES, "v9", "", {"id": "v9"})
    db.close()
    sample_indexes(tmp_path)
    assert ensure_migrated(tmp_path) == "already"
    # The legacy files were never touched.
    assert (tmp_path / "voices.json").exists()


def test_fresh_data_root_reports_fresh(tmp_path):
    assert ensure_migrated(tmp_path) == "fresh"
    assert not (tmp_path / "library.db").exists() or Database(
        library_db_path(tmp_path), migrate=False).is_empty()


def test_corrupt_index_fails_migration_loudly_and_keeps_originals(tmp_path):
    """ADR-0017 eliminates "parse-failure = empty library": a corrupt legacy
    index must abort the migration loudly instead of being silently skipped
    and then deleted as "migrated"."""
    from voiceclone_sidecar.db import DatabaseError

    (tmp_path / "voices.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DatabaseError, match="voices.json"):
        ensure_migrated(tmp_path)
    assert (tmp_path / "voices.json").exists()
    assert not list(tmp_path.glob("migration-backup-*"))
    db = Database(library_db_path(tmp_path), migrate=False)
    try:
        assert db.is_empty()
    finally:
        db.close()


def test_failed_import_rolls_back_and_keeps_originals(tmp_path, monkeypatch):
    sample_indexes(tmp_path)

    def boom(db, path):
        raise RuntimeError("disk exploded")

    import voiceclone_sidecar.migrate as m
    patched = list(m._IMPORTERS)
    patched[0] = (patched[0][0], patched[0][1], boom)
    monkeypatch.setattr(m, "_IMPORTERS", tuple(patched))
    with pytest.raises(RuntimeError):
        ensure_migrated(tmp_path)

    # Originals untouched, no marker, no partial rows, no backup leftovers.
    assert (tmp_path / "voices.json").exists()
    assert (tmp_path / "generations.json").exists()
    db = Database(library_db_path(tmp_path), migrate=False)
    try:
        assert db.is_empty()
        assert db.get_kv("meta", MIGRATION_MARKER) is None
    finally:
        db.close()
    assert not list(tmp_path.glob("migration-backup-*"))
