"""SQLite index layer (ADR-0017, issue #24).

The database lives at ``<data_root>/library.db`` with WAL journaling. Every
index table stores one JSON payload per row plus the columns the query paths
filter on, so searches and pagination run in SQL instead of a full in-lock
linear scan. Reads always return freshly parsed objects — a record handed to
a caller is never aliased into the store.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

LIBRARY_DB_NAME = "library.db"

# The JSON index files the one-time migration consumes (ADR-0017). Kept here
# so the migration and the backup skip-lists share one source of truth.
JSON_INDEX_NAMES = (
    "voices.json",
    "generations.json",
    "compare.json",
    "regression.json",
    "settings.json",
    "app_state.json",
)

T_META = "meta"
T_VOICES = "voices"
T_GENERATIONS = "generations"
T_COMPARE = "compare_sessions"
T_REGRESSION = "regression_sessions"
T_SETTINGS = "settings"
T_APP_STATE = "app_state"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS voices (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS generations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT '',
    engine_id TEXT NOT NULL DEFAULT '',
    voice_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    voice_name TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_generations_created_at ON generations(created_at);
CREATE INDEX IF NOT EXISTS idx_generations_engine ON generations(engine_id);
CREATE INDEX IF NOT EXISTS idx_generations_voice ON generations(voice_id);
CREATE TABLE IF NOT EXISTS compare_sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS regression_sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_PAYLOAD_TABLES = (T_VOICES, T_GENERATIONS, T_COMPARE, T_REGRESSION)
_KV_TABLES = (T_META, T_SETTINGS, T_APP_STATE)


class DatabaseError(RuntimeError):
    """Raised when the SQLite index is unusable. Loud, never silent (ADR-0017)."""


def library_db_path(data_dir: Path) -> Path:
    return Path(data_dir) / LIBRARY_DB_NAME


class Database:
    """One SQLite connection to the library index.

    ``path=None`` opens an in-memory database (tests). A corrupted on-disk
    file raises :class:`DatabaseError` at open time — a broken index is a
    loud failure, not an empty library.
    """

    def __init__(self, path: Path | None, migrate: bool = True) -> None:
        self.path = Path(path) if path is not None else None
        self._migrate = migrate
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._open()

    def _connect(self) -> sqlite3.Connection:
        target = str(self.path) if self.path is not None else ":memory:"
        try:
            conn = sqlite3.connect(target, check_same_thread=False, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise DatabaseError(f"索引数据库无法打开（{target}）：{exc}") from exc
        return conn

    def _open(self) -> None:
        self._conn = self._connect()
        self._tx_depth = 0
        # ADR-0017 decision 5: opening a store-backed database is the
        # migration trigger. A legacy JSON-only data root is imported (after
        # an automatic backup) exactly once, before any store reads it.
        if self.path is not None and self._migrate:
            from .migrate import ensure_migrated

            ensure_migrated(self.path.parent)

    def close(self) -> None:
        with self._lock, contextlib.suppress(sqlite3.Error):
            self._conn.close()

    def reconnect(self) -> None:
        """Reopen the same path after the underlying file was swapped out
        (library restore)."""
        with self._lock:
            self.close()
            self._open()

    @contextmanager
    def tx(self):
        """A write transaction: commit on success, roll back on error.

        Reentrant: nested ``tx()`` blocks join the outer transaction, so an
        importer can wrap many ``put_payload`` calls in one atomic commit.
        """
        with self._lock:
            if getattr(self, "_tx_depth", 0) > 0:
                self._tx_depth += 1
                try:
                    yield self._conn
                except BaseException:
                    # Mark the outer transaction poisoned without swallowing
                    # the exception chain.
                    self._tx_poisoned = True
                    raise
                finally:
                    self._tx_depth -= 1
                return
            try:
                self._conn.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise DatabaseError(f"无法开启写事务：{exc}") from exc
            self._tx_depth = 1
            self._tx_poisoned = False
            try:
                yield self._conn
            except BaseException:
                self._conn.rollback()
                self._tx_depth = 0
                raise
            else:
                if self._tx_poisoned:
                    self._conn.rollback()
                    self._tx_depth = 0
                    raise DatabaseError("事务中某个子操作已失败，已回滚")
                self._conn.commit()
                self._tx_depth = 0

    # -- generic row access ---------------------------------------------------

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    # -- payload records --------------------------------------------------------

    def _check_table(self, table: str, allowed: tuple) -> None:
        # Table names are interpolated into SQL, so every entry point validates
        # against its allow-list with a real exception (never a bare `assert`,
        # which would vanish under `python -O`).
        if table not in allowed:
            raise DatabaseError(f"未知的索引表：{table!r}")

    def put_payload(
        self,
        table: str,
        record_id: str,
        created_at: str,
        payload: dict,
        columns: dict | None = None,
    ) -> None:
        self._check_table(table, _PAYLOAD_TABLES)
        cols = {"created_at": created_at or "", **(columns or {})}
        col_names = ", ".join(cols)
        placeholders = ", ".join("?" for _ in cols)
        assignments = ", ".join(f"{c} = ?" for c in cols)
        encoded = json.dumps(payload, ensure_ascii=False)
        with self.tx() as conn:
            conn.execute(
                f"INSERT INTO {table} (id, {col_names}, payload) VALUES (?, {placeholders}, ?) "
                f"ON CONFLICT(id) DO UPDATE SET {assignments}, payload = excluded.payload",
                (record_id, *cols.values(), encoded, *cols.values()),
            )

    def get_payload(self, table: str, record_id: str) -> dict | None:
        self._check_table(table, _PAYLOAD_TABLES)
        row = self.execute(
            f"SELECT payload FROM {table} WHERE id = ?", (record_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def all_payloads(self, table: str) -> list[dict]:
        self._check_table(table, _PAYLOAD_TABLES)
        order = "DESC" if table != T_VOICES else "ASC"
        rows = self.execute(
            f"SELECT payload FROM {table} ORDER BY created_at {order}, id"
        ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def delete_payload(self, table: str, record_id: str) -> bool:
        self._check_table(table, _PAYLOAD_TABLES)
        with self.tx() as conn:
            cur = conn.execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))
        return cur.rowcount > 0

    def count(self, table: str) -> int:
        return self.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

    def is_empty(self) -> bool:
        """True when no real index rows exist (migration trigger, ADR-0017)."""
        return all(self.count(t) == 0 for t in _PAYLOAD_TABLES)

    # -- key/value blocks (settings, app_state, meta) ---------------------------

    def get_kv(self, table: str, key: str):
        self._check_table(table, _KV_TABLES)
        row = self.execute(
            f"SELECT value FROM {table} WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row["value"]) if row else None

    def put_kv(self, table: str, key: str, value) -> None:
        self._check_table(table, _KV_TABLES)
        with self.tx() as conn:
            conn.execute(
                f"INSERT INTO {table} (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value, ensure_ascii=False)),
            )

    def all_kv(self, table: str) -> dict:
        self._check_table(table, _KV_TABLES)
        rows = self.execute(f"SELECT key, value FROM {table}").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    # -- snapshot / integrity (ADR-0017 decision 4) ------------------------------

    def vacuum_into(self, dest: Path) -> Path:
        """Write a consistent single-file snapshot of the live database."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        try:
            self.execute("VACUUM INTO ?", (str(dest),))
        except sqlite3.Error as exc:
            raise DatabaseError(f"无法生成数据库快照：{exc}") from exc
        return dest

    def integrity_check(self) -> bool:
        row = self.execute("PRAGMA integrity_check").fetchone()
        return bool(row) and row[0] == "ok"


# -- module-level key/value access for path-based callers ----------------------
#
# Settings blocks (engine config, transcription provider) and app state
# (consent, UI prefs) are read from several modules that only know the data
# root. They open a short-lived connection per access: these calls are
# low-frequency, and a fresh connection can never go stale after a restore.


def read_kv_block(data_dir: Path | None, table: str, key: str):
    if data_dir is None:
        return None
    db = Database(library_db_path(data_dir), migrate=False)
    try:
        return db.get_kv(table, key)
    finally:
        db.close()


def write_kv_block(data_dir: Path, table: str, key: str, value) -> None:
    db = Database(library_db_path(data_dir), migrate=False)
    try:
        db.put_kv(table, key, value)
    finally:
        db.close()
