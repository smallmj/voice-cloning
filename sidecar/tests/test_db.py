"""SQLite index layer (ADR-0017, issue #24): unit tests."""

from __future__ import annotations

import json
import sqlite3

import pytest

from voiceclone_sidecar.db import (
    Database,
    DatabaseError,
    T_APP_STATE,
    T_COMPARE,
    T_GENERATIONS,
    T_REGRESSION,
    T_SETTINGS,
    T_VOICES,
    library_db_path,
    read_kv_block,
    write_kv_block,
)


# -- open / schema -------------------------------------------------------------


def test_memory_database_roundtrips_payload():
    db = Database(None)
    record = {"id": "g1", "text": "你好", "created_at": "2026-01-01T00:00:00Z"}
    db.put_payload(T_GENERATIONS, "g1", record["created_at"], record,
                   {"engine_id": "fake", "voice_id": "v1", "status": "ok",
                    "text": "你好", "voice_name": "小王"})
    assert db.get_payload(T_GENERATIONS, "g1") == record
    assert db.get_payload(T_VOICES, "g1") is None
    db.close()


def test_payload_read_returns_a_fresh_object_not_a_stored_reference():
    db = Database(None)
    record = {"id": "v1", "name": "音色"}
    db.put_payload(T_VOICES, "v1", "", record)
    first = db.get_payload(T_VOICES, "v1")
    first["name"] = "改过的名字"
    assert db.get_payload(T_VOICES, "v1")["name"] == "音色"


def test_put_payload_upserts_existing_id():
    db = Database(None)
    db.put_payload(T_VOICES, "v1", "t1", {"id": "v1", "name": "旧"})
    db.put_payload(T_VOICES, "v1", "t2", {"id": "v1", "name": "新"})
    assert db.count(T_VOICES) == 1
    assert db.get_payload(T_VOICES, "v1")["name"] == "新"


def test_all_payloads_orders_generations_newest_first():
    db = Database(None)
    db.put_payload(T_GENERATIONS, "a", "2026-01-01T00:00:00Z", {"id": "a"})
    db.put_payload(T_GENERATIONS, "b", "2026-02-01T00:00:00Z", {"id": "b"})
    db.put_payload(T_VOICES, "v", "", {"id": "v"})
    gens = db.all_payloads(T_GENERATIONS)
    assert [r["id"] for r in gens] == ["b", "a"]


def test_delete_payload_removes_row_and_reports_presence():
    db = Database(None)
    db.put_payload(T_COMPARE, "s1", "", {"id": "s1"})
    assert db.delete_payload(T_COMPARE, "s1") is True
    assert db.delete_payload(T_COMPARE, "s1") is False
    assert db.get_payload(T_COMPARE, "s1") is None


# -- persistence across reopen ---------------------------------------------------


def test_on_disk_database_survives_close_and_reopen(tmp_path):
    path = tmp_path / "sub" / library_db_path(tmp_path).name
    db = Database(path)
    db.put_payload(T_VOICES, "v1", "t", {"id": "v1"})
    db.close()
    db2 = Database(path, migrate=False)
    assert db2.get_payload(T_VOICES, "v1") == {"id": "v1"}
    db2.close()


def test_wal_mode_is_active_on_disk_database(tmp_path):
    db = Database(library_db_path(tmp_path))
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    db.close()


def test_corrupt_database_file_fails_loudly_not_as_empty_library(tmp_path):
    path = library_db_path(tmp_path)
    path.write_bytes(b"this is not a sqlite database at all")
    with pytest.raises(DatabaseError):
        Database(path)


# -- emptiness / counts -----------------------------------------------------------


def test_is_empty_true_only_when_no_index_rows(tmp_path):
    db = Database(None)
    assert db.is_empty()
    db.put_kv(T_SETTINGS, "engines", {"a": {}})
    assert db.is_empty(), "settings rows are not index records"
    db.put_payload(T_REGRESSION, "r1", "", {"id": "r1"})
    assert not db.is_empty()


def test_counts_reflect_stores(tmp_path):
    db = Database(None)
    db.put_payload(T_VOICES, "v1", "", {"id": "v1"})
    db.put_payload(T_GENERATIONS, "g1", "", {"id": "g1"})
    assert db.count(T_VOICES) == 1 and db.count(T_GENERATIONS) == 1
    assert db.count(T_COMPARE) == 0


# -- key/value blocks ---------------------------------------------------------------


def test_kv_roundtrip_and_all_kv(tmp_path):
    db = Database(None)
    db.put_kv(T_APP_STATE, "ui_prefs", {"theme": "dark"})
    db.put_kv(T_APP_STATE, "voice_consent", {"accepted": True})
    assert db.get_kv(T_APP_STATE, "ui_prefs") == {"theme": "dark"}
    assert db.all_kv(T_APP_STATE) == {
        "ui_prefs": {"theme": "dark"},
        "voice_consent": {"accepted": True},
    }


def test_read_write_kv_block_helpers_over_a_path(tmp_path):
    write_kv_block(tmp_path, T_SETTINGS, "engines", {"fake": {"env": {"A": "1"}}})
    assert read_kv_block(tmp_path, T_SETTINGS, "engines") == {"fake": {"env": {"A": "1"}}}
    assert read_kv_block(None, T_SETTINGS, "engines") is None


# -- snapshot / integrity --------------------------------------------------------------


def test_vacuum_into_produces_a_readable_consistent_snapshot(tmp_path):
    live = Database(library_db_path(tmp_path))
    live.put_payload(T_VOICES, "v1", "", {"id": "v1"})
    snapshot = live.vacuum_into(tmp_path / "snap.db")
    live.put_payload(T_VOICES, "v2", "", {"id": "v2"})

    raw = sqlite3.connect(snapshot)
    try:
        rows = raw.execute("SELECT id FROM voices ORDER BY id").fetchall()
        assert rows == [("v1",)], "snapshot must freeze the state at VACUUM time"
    finally:
        raw.close()
    live.close()


def test_integrity_check_true_on_healthy_db():
    db = Database(None)
    assert db.integrity_check()


def test_transaction_rolls_back_on_error():
    db = Database(None)
    db.put_payload(T_VOICES, "v1", "", {"id": "v1"})
    with pytest.raises(RuntimeError):
        with db.tx() as conn:
            conn.execute("DELETE FROM voices")
            raise RuntimeError("boom")
    assert db.count(T_VOICES) == 1


def test_concurrent_threads_share_one_connection_safely(tmp_path):
    db = Database(library_db_path(tmp_path))
    import threading

    def write(i: int) -> None:
        for j in range(20):
            db.put_payload(T_GENERATIONS, f"g{i}-{j}", "", {"id": f"g{i}-{j}"})

    threads = [threading.Thread(target=write, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert db.count(T_GENERATIONS) == 80
    db.close()
