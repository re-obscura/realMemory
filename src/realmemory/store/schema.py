"""Схема БД realMemory: DDL, миграционная цепочка, FTS5-индекс.

Самый хрупкий и наименее посещаемый код хранилища собран здесь одной кучей,
чтобы будущие миграции ревьюились в одном месте. Правило: каждая мутация
схемы предваряется страховочной копией базы (MemoryStore.backup).
"""
from __future__ import annotations

import sqlite3
import threading

from ._sql import _upsert_meta

# Версия схемы; растёт при несовместимых изменениях. Миграции выполняются
# при открытии по цепочке от сохранённой версии, каждая мутация схемы
# предваряется страховочной копией базы.
SCHEMA_VERSION = 3

# Цепочка миграций по целевым версиям. Индексы по колонкам, которых могло не
# быть раньше, живут здесь и в быстром пути свежей схемы — не в базовом
# CREATE-скрипте, иначе он ломает открытие унаследованных баз.
_MIGRATION_V1 = (
    "ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'global'",
    "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)",
)

_MIGRATION_V2 = (
    "ALTER TABLE memories ADD COLUMN author TEXT NOT NULL DEFAULT ''",
    """
    CREATE TABLE IF NOT EXISTS publications (
        id TEXT PRIMARY KEY,
        trace_id INTEGER NOT NULL,
        project TEXT NOT NULL DEFAULT '',
        author TEXT NOT NULL DEFAULT '',
        published_at REAL NOT NULL,
        revoked_at REAL,
        content_hash TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_author ON memories(author)",
    "CREATE INDEX IF NOT EXISTS idx_publications_trace ON publications(trace_id)",
)

_MIGRATION_V3 = (
    "ALTER TABLE publications ADD COLUMN synced_at REAL",
    "CREATE INDEX IF NOT EXISTS idx_publications_synced ON publications(synced_at)",
)

_SCHEMA_CHAIN: list[tuple[int, tuple[str, ...]]] = [
    (1, _MIGRATION_V1), (2, _MIGRATION_V2), (3, _MIGRATION_V3),
]


def _is_duplicate_column(exc: sqlite3.OperationalError) -> bool:
    return "duplicate column name" in str(exc).lower()


def _migrate_schema(conn: sqlite3.Connection, lock: threading.Lock,
                    backup_factory) -> str:
    """Довести схему до SCHEMA_VERSION по цепочке миграций.

    Свежая база рождается с полной схемой и получает метку версии без мутаций;
    унаследованные проходят недостающие шаги, каждой мутации предшествует
    страховочная копия (после успешного шага). Повторный запуск частично
    обновлённой базы терпит «колонка уже существует». Возвращает версию."""
    with lock:
        row = conn.execute(
            "SELECT value FROM db_meta WHERE key='schema_version'"
        ).fetchone()
        stored = int(row[0]) if row else None

    def mark(target: int) -> None:
        with lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _upsert_meta(conn, "schema_version", str(target))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    if stored == SCHEMA_VERSION:
        return str(SCHEMA_VERSION)
    if stored is None:
        with lock:
            cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(memories)").fetchall()}
            pubs_cols = {r[1] for r in conn.execute(
                "PRAGMA table_info(publications)").fetchall()}
            complete = ("author" in cols and "synced_at" in pubs_cols
                        and bool(conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' "
                            "AND name='publications'").fetchone()))
        if complete:
            # свежая база: достроить идемпотентные индексы «мигрирующих»
            # колонок (их нет в базовом скрипте) и зафиксировать версию
            with lock:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for idx in (
                        "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)",
                        "CREATE INDEX IF NOT EXISTS idx_memories_author ON memories(author)",
                        "CREATE INDEX IF NOT EXISTS idx_publications_trace ON publications(trace_id)",
                        "CREATE INDEX IF NOT EXISTS idx_publications_synced ON publications(synced_at)",
                    ):
                        conn.execute(idx)
                    # версия фиксируется той же транзакцией: mark() здесь нельзя —
                    # его внутренний lock нерекурсивен
                    _upsert_meta(conn, "schema_version", str(SCHEMA_VERSION))
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
            return str(SCHEMA_VERSION)
        # базы до появления нумерации: наличие scope отличает v0.3+ от v0.2-
        stored = 1 if "scope" in cols else 0

    version = max(stored, 0)
    for target, statements in _SCHEMA_CHAIN:
        if version >= target:
            continue
        backup_factory()
        with lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                for stmt in statements:
                    try:
                        conn.execute(stmt)
                    except sqlite3.OperationalError as exc:
                        if not (_is_duplicate_column(exc)
                                and stmt.lstrip().upper().startswith("ALTER")):
                            raise
                # версия фиксируется той же транзакцией, что и DDL: атомарно
                _upsert_meta(conn, "schema_version", str(target))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            version = target
    return str(version)


# Схема списком стейтментов: executescript() внутри BEGIN не годится
# (он неявно коммитит транзакцию).
_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        meta TEXT NOT NULL,
        embedding BLOB NOT NULL,
        sdr BLOB NOT NULL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        reinforced_count INTEGER NOT NULL DEFAULT 0,
        last_reinforced_at REAL NOT NULL,
        base_strength REAL NOT NULL DEFAULT 1.0,
        valid_from REAL NOT NULL,
        valid_to REAL,
        superseded_by INTEGER,
        scope TEXT NOT NULL DEFAULT 'global',
        author TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS publications (
        id TEXT PRIMARY KEY,
        trace_id INTEGER NOT NULL,
        project TEXT NOT NULL DEFAULT '',
        author TEXT NOT NULL DEFAULT '',
        published_at REAL NOT NULL,
        revoked_at REAL,
        content_hash TEXT NOT NULL DEFAULT '',
        synced_at REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)",
    "CREATE INDEX IF NOT EXISTS idx_publications_trace ON publications(trace_id)",
    """
    CREATE TABLE IF NOT EXISTS edges (
        key INTEGER PRIMARY KEY,
        w REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eligibility (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        src BLOB NOT NULL,
        dst BLOB NOT NULL,
        strength REAL NOT NULL,
        created_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS elig_sources (
        seq INTEGER NOT NULL REFERENCES eligibility(seq) ON DELETE CASCADE,
        mem_id INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_elig_sources_mem ON elig_sources(mem_id)",
    """
    CREATE TABLE IF NOT EXISTS events (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        type TEXT NOT NULL,
        data TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)",
    """
    CREATE TABLE IF NOT EXISTS db_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)

# Гибридный поиск: внешне-содержимый FTS5 поверх текстов следа.
# Создаётся отдельно от основной схемы: сборка SQLite без FTS5
# не должна ломать открытие базы — ядро просто теряет keyword-канал.
_FTS_STATEMENTS = (
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
        text,
        content='memories',
        content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
        INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF text ON memories BEGIN
        INSERT INTO memories_fts(memories_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
        INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
)
