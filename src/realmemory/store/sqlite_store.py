"""SQLite-хранилище: следы, рёбра L2, eligibility, события. Контракт: docs/CONTRACTS.md.

Единый источник истины для всех процессов: долгоживущий MCP-сервер и
краткоживущие хуки читают и пишут одну базу. WAL + busy_timeout +
BEGIN IMMEDIATE на мутациях исключают потерю состояния «кто последний
записал снапшот» — снапшотных файлов больше нет.

Схема:
  memories      следы (embedding -> float32 blob, sdr -> int32 blob отсортирован)
  edges         стабильные рёбра L2, key = src*n_units + dst (layout dict AssemblyNetwork)
  eligibility   незакоммиченные bind'ы (src/dst — int32 blob одного события)
  elig_sources  следы-источники события (для reward-матчинга)
  events        журнал пластичности (бывший journal.jsonl): аудит и статистика
  db_meta       маркер эмбеддера, конфиг, служебные счётчики/метки
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..types import STATUS_ACTIVE, MemoryRecord

# Версия схемы; растёт при несовместимых изменениях. Миграции выполняются
# при открытии, перед изменением схемы делается автоматический бэкап.
SCHEMA_VERSION = 1

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
        scope TEXT NOT NULL DEFAULT 'global'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)",
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

_INSERT_SQL = (
    "INSERT INTO memories(text,kind,status,meta,embedding,sdr,"
    "created_at,updated_at,reinforced_count,last_reinforced_at,base_strength,"
    "valid_from,valid_to,superseded_by,scope) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

_COLS = (
    "id,text,kind,status,meta,embedding,sdr,created_at,updated_at,"
    "reinforced_count,last_reinforced_at,base_strength,valid_from,valid_to,superseded_by,scope"
)


class StorageError(Exception):
    """Повреждение или недоступность хранилища."""


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

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Токены для keyword-канала: слова в нижнем регистре (unicode)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def search_tokens(text: str) -> set[str]:
    """Токены, значимые для сопоставления запроса и следа: 1-символьные
    отбрасываются синхронно с build_fts_query, иначе каналы определяют
    «токен» по-разному и full_match ведёт себя несогласованно."""
    return {t for t in tokenize(text) if len(t) > 1}


def build_fts_query(text: str, max_terms: int = 12) -> str | None:
    """Выражение MATCH из токенов запроса: "tok1" OR "tok2" ..."""
    seen: dict[str, None] = {}
    for t in tokenize(text):
        if len(t) > 1:
            seen.setdefault(t)
        if len(seen) >= max_terms:
            break
    if not seen:
        return None
    return " OR ".join(f'"{t}"' for t in seen)


def _pack_f32(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _pack_i32(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.int32).tobytes()


def _row_to_record(row: tuple, dim: int) -> MemoryRecord:
    (rid, text, kind, status, meta_json, emb_b, sdr_b, created, updated,
     rcount, rlast, base, vfrom, vto, sby, scope) = row
    return MemoryRecord(
        id=int(rid),
        text=text,
        kind=kind,
        status=status,
        meta=json.loads(meta_json),
        embedding=np.frombuffer(emb_b, dtype=np.float32, count=dim).copy(),
        sdr=np.frombuffer(sdr_b, dtype=np.int32).copy(),
        created_at=float(created),
        updated_at=float(updated),
        reinforced_count=int(rcount),
        last_reinforced_at=float(rlast),
        base_strength=float(base),
        valid_from=float(vfrom),
        valid_to=None if vto is None else float(vto),
        superseded_by=None if sby is None else int(sby),
        scope=str(scope),
    )


class MemoryStore:
    def __init__(self, path: str | Path, dim: int) -> None:
        self.path = Path(path)
        self.dim = int(dim)
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                         isolation_level=None)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            with self._txn() as con:
                for stmt in _SCHEMA_STATEMENTS:
                    con.execute(stmt)
            # миграция баз, созданных до появления scope; индекс строго после
            # гарантии колонки — иначе на legacy-базе нет такого столбца.
            # Изменению схемы всегда предшествует страховочная копия.
            with self._lock:
                cols = {
                    r[1] for r in self._conn.execute(
                        "PRAGMA table_info(memories)"
                    ).fetchall()
                }
            scope_missing = "scope" not in cols
            if scope_missing:
                self.backup()
            with self._txn() as con:
                if scope_missing:
                    con.execute(
                        "ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL "
                        "DEFAULT 'global'"
                    )
                con.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)"
                )
                con.execute(
                    "INSERT INTO db_meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO NOTHING",
                    (str(SCHEMA_VERSION),),
                )
            self._fts_error: str | None = None
            try:
                with self._txn() as con:
                    for stmt in _FTS_STATEMENTS:
                        con.execute(stmt)
                self._fts_enabled = True
            except sqlite3.OperationalError as exc:
                # редкая сборка SQLite без FTS5: ядро работает, keyword-канала нет
                self._fts_enabled = False
                self._fts_error = str(exc)
        except sqlite3.DatabaseError as exc:
            raise StorageError(f"не удалось открыть {self.path}: {exc}") from exc

    @contextmanager
    def _txn(self):
        """Мутации в BEGIN IMMEDIATE: захват блокировки записи до конца блока.
        Конкурентные процессы выстраиваются по busy_timeout вместо потери правок."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def __enter__(self) -> MemoryStore:  # noqa: PYI034
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

    # -- следы: запись -----------------------------------------------------------

    def insert(self, rec: MemoryRecord) -> int:
        with self._txn() as con:
            cur = con.execute(
                _INSERT_SQL,
                (
                    rec.text, rec.kind, rec.status, json.dumps(rec.meta, ensure_ascii=False),
                    _pack_f32(rec.embedding), _pack_i32(rec.sdr),
                    float(rec.created_at), float(rec.updated_at), int(rec.reinforced_count),
                    float(rec.last_reinforced_at), float(rec.base_strength),
                    float(rec.valid_from),
                    None if rec.valid_to is None else float(rec.valid_to),
                    rec.superseded_by,
                    str(rec.scope or "global"),
                ),
            )
            rowid = cur.lastrowid
        if rowid is None:  # pragma: no cover - не бывает после успешного INSERT
            raise StorageError("INSERT не вернул rowid")
        return int(rowid)

    def update_trace(
        self,
        memory_id: int,
        base_strength: float,
        reinforced_count: int,
        last_reinforced_at: float,
        kind: str | None = None,
    ) -> None:
        sets = "base_strength=?, reinforced_count=?, last_reinforced_at=?, updated_at=?"
        params: list = [float(base_strength), int(reinforced_count),
                        float(last_reinforced_at), float(last_reinforced_at)]
        if kind is not None:
            sets += ", kind=?"
            params.append(kind)
        params.append(int(memory_id))
        with self._txn() as con:
            con.execute(f"UPDATE memories SET {sets} WHERE id=?", params)

    def mark_superseded(self, memory_id: int, by_id: int, when: float) -> None:
        with self._txn() as con:
            con.execute(
                "UPDATE memories SET status='superseded', valid_to=?, superseded_by=? WHERE id=?",
                (float(when), int(by_id), int(memory_id)),
            )

    def adjust_base(self, memory_id: int, base_strength: float, updated_at: float) -> None:
        """Ослабление следа без сброса таймера подкрепления (негативный feedback)."""
        with self._txn() as con:
            con.execute(
                "UPDATE memories SET base_strength=?, updated_at=? WHERE id=?",
                (float(base_strength), float(updated_at), int(memory_id)),
            )

    def max_updated_at(self) -> float | None:
        with self._lock:
            row = self._conn.execute("SELECT MAX(updated_at) FROM memories").fetchone()
        return None if row is None or row[0] is None else float(row[0])

    def backup(self, dest_dir: str | Path | None = None, keep: int = 10,
               min_interval_s: float = 0.0) -> Path | None:
        """Консистентная копия базы через sqlite backup API + ротация.

        Копии складываются в <каталог базы>/backups/memory-<timestamp>.db;
        остаются последние `keep` штук (keep=0 — копия без ротации).
        min_interval_s > 0 включает троттлинг по wall-clock (метка
        db_meta.last_backup_at): если с последней копии прошло меньше времени,
        копия не делается и возвращается None — «сон» после каждого ответа
        агента не обязан копировать всю базу.
        """
        now = time.time()
        if min_interval_s > 0:
            last = self.get_meta("last_backup_at")
            if last is not None and (now - float(last)) < min_interval_s:
                return None
        dest_dir = Path(dest_dir) if dest_dir else self.path.parent / "backups"
        dest_dir.mkdir(parents=True, exist_ok=True)
        # миллисекунды в суффиксе: две консолидации в одну секунду не делят имя
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(now * 1000) % 1000:03d}"
        dest = dest_dir / f"memory-{stamp}.db"
        with self._lock:
            out = sqlite3.connect(str(dest))
            try:
                self._conn.backup(out)
            finally:
                out.close()
        self.set_meta("last_backup_at", repr(float(now)))
        if keep > 0:
            olds = sorted(dest_dir.glob("memory-*.db"))
            for old in olds[:-keep]:
                old.unlink(missing_ok=True)
        return dest

    # -- мета базы -----------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM db_meta WHERE key=?", (key,)
            ).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        with self._txn() as con:
            con.execute(
                "INSERT INTO db_meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )

    def bump_meta_int(self, key: str, delta: int = 1) -> int:
        """Атомарный инкремент целочисленного счётчика в db_meta; возвращает новое значение."""
        with self._txn() as con:
            con.execute(
                "INSERT INTO db_meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE "
                "SET value=CAST(CAST(value AS INTEGER)+? AS TEXT)",
                (key, str(delta), int(delta)),
            )
            (val,) = con.execute("SELECT value FROM db_meta WHERE key=?", (key,)).fetchone()
        return int(val)

    def consume_meta_int(self, key: str) -> int:
        """Прочитать и обнулить счётчик (pending_rewards между «снами»)."""
        with self._txn() as con:
            row = con.execute("SELECT value FROM db_meta WHERE key=?", (key,)).fetchone()
            if row is None:
                return 0
            con.execute("UPDATE db_meta SET value='0' WHERE key=?", (key,))
        return int(row[0])

    # -- чтение следов ---------------------------------------------------------------

    def get(self, memory_id: int) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLS} FROM memories WHERE id=?", (int(memory_id),)
            ).fetchone()
        return _row_to_record(row, self.dim) if row else None

    def get_many(self, memory_ids: Sequence[int]) -> list[MemoryRecord]:
        ids = [int(i) for i in memory_ids]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM memories WHERE id IN ({placeholders})", ids
            ).fetchall()
        by_id = {int(r[0]): _row_to_record(r, self.dim) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def iter_active(self, batch: int = 256) -> Iterator[MemoryRecord]:
        """Порционная итерация активных следов по курсору id.

        Каждая порция читается под замком и материализуется до yield: открытый
        курсор не удерживается между порциями, поэтому конкурентные мутации
        (BEGIN IMMEDIATE/ROLLBACK того же соединения из другого треда MCP-сервера)
        не рвут итерацию. Следы, вставленные после старта итерации, видны —
        это скан «живого» состояния, а не снапшот.
        """
        last_id = 0
        while True:
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT {_COLS} FROM memories WHERE status=? AND id>? ORDER BY id LIMIT ?",
                    (STATUS_ACTIVE, last_id, int(batch)),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                yield _row_to_record(row, self.dim)
                last_id = int(row[0])

    def all_active_ids(self) -> np.ndarray:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM memories WHERE status=?", (STATUS_ACTIVE,)
            ).fetchall()
        return np.asarray([r[0] for r in rows], dtype=np.int64)

    def count(self, status: str | None = None, kind: str | None = None,
              scope: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM memories"
        conds, params = [], []
        if status is not None:
            conds.append("status=?")
            params.append(status)
        if kind is not None:
            conds.append("kind=?")
            params.append(kind)
        if scope is not None:
            conds.append("scope=?")
            params.append(scope)
        if conds:
            query += " WHERE " + " AND ".join(conds)
        with self._lock:
            (n,) = self._conn.execute(query, params).fetchone()
        return int(n)

    def scope_counts(self) -> dict[str, int]:
        """Активные следы по скоупам (для introspect/отчёта)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope, COUNT(*) FROM memories WHERE status='active' GROUP BY scope"
            ).fetchall()
        return {str(s): int(n) for s, n in rows}

    def top_by_reinforcements(self, limit: int = 10) -> list[MemoryRecord]:
        """Самые подкреплённые активные следы (для отчёта)."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM memories WHERE status='active' "
                "ORDER BY reinforced_count DESC, last_reinforced_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_row_to_record(r, self.dim) for r in rows]

    def stale_episodic(self, limit: int = 10) -> list[MemoryRecord]:
        """Активные эпизоды дольше всего без подкрепления — кандидаты на забывание."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM memories WHERE status='active' AND kind='episodic' "
                "ORDER BY last_reinforced_at ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_row_to_record(r, self.dim) for r in rows]

    # -- keyword-канал (FTS5) ------------------------------------------------------

    @property
    def fts_enabled(self) -> bool:
        return getattr(self, "_fts_enabled", False)

    @property
    def fts_error(self) -> str | None:
        return getattr(self, "_fts_error", None)

    def fts_match(self, expr: str, limit: int = 32) -> list[tuple[int, float]]:
        """ID следа + bm25-ранг (меньше = лучше) по выражению MATCH."""
        if not self.fts_enabled or not expr:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT rowid, bm25(memories_fts) FROM memories_fts "
                "WHERE memories_fts MATCH ? ORDER BY 2 LIMIT ?",
                (expr, int(limit)),
            ).fetchall()
        return [(int(r), float(b)) for r, b in rows]

    # -- рёбра L2 ---------------------------------------------------------------------

    def edges_rev(self) -> int:
        """Версия рёбер: растёт при любой их мутации (инвалидация CSR-кэша фасада)."""
        val = self.get_meta("edges_rev")
        return int(val) if val is not None else 0

    def edges_load(self) -> tuple[np.ndarray, np.ndarray]:
        """Все рёбра как отсортированные (keys int64, weights float32) для CSR-кэша."""
        with self._lock:
            rows = self._conn.execute("SELECT key, w FROM edges ORDER BY key").fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        keys = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        ws = np.fromiter((r[1] for r in rows), dtype=np.float32, count=len(rows))
        return keys, ws

    def edges_apply(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        w: np.ndarray,
        now: float,
        tau: float,
        min_weight: float,
        stride: int,
    ) -> tuple[int, int]:
        """Консолидационная запись рёбер одним транзакционным шагом: распад всех
        стабильных весов за (now - last_edge_tick), обрезка слабых, вливание батча
        (аккумуляция в существующие ключи), тик часов.

        Возвращает (число влитых пар, число обрезанных рёбер). Вызовы разных
        процессов сериализуются BEGIN IMMEDIATE; Δt считается от last_edge_tick
        после захвата блокировки — двойного распада нет.
        """
        pairs = [
            (int(i) * int(stride) + int(j), float(wi))
            for i, j, wi in zip(np.asarray(src).tolist(), np.asarray(dst).tolist(),
                                np.asarray(w).tolist())
        ]
        with self._txn() as con:
            tick_row = con.execute(
                "SELECT value FROM db_meta WHERE key='last_edge_tick'"
            ).fetchone()
            pruned = 0
            if tick_row is not None:
                tick = float(tick_row[0])
                if now > tick:
                    factor = float(np.exp(-(now - tick) / float(tau)))
                    if factor < 1.0:
                        con.execute("UPDATE edges SET w = w * ?", (factor,))
                    con.execute("DELETE FROM edges WHERE w < ?", (float(min_weight),))
                    pruned = int(con.execute("SELECT changes()").fetchone()[0])
            if pairs:
                con.executemany(
                    "INSERT INTO edges(key, w) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET w = w + excluded.w",
                    pairs,
                )
            con.execute(
                "INSERT INTO db_meta(key, value) VALUES('last_edge_tick', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (repr(float(now)),),
            )
            con.execute(
                "INSERT INTO db_meta(key, value) VALUES('edges_rev', '1') "
                "ON CONFLICT(key) DO UPDATE SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT)"
            )
        return len(pairs), pruned

    def edges_import(self, keys: np.ndarray, ws: np.ndarray, last_tick: float | None) -> None:
        """Прямая вставка рёбер без распада (миграция legacy-снапшота).
        Часы сети переносятся из снапшота, чтобы старые веса не омолодились."""
        pairs = [
            (int(k), float(wv))
            for k, wv in zip(np.asarray(keys).reshape(-1).tolist(),
                             np.asarray(ws).reshape(-1).tolist())
        ]
        with self._txn() as con:
            if pairs:
                con.executemany("INSERT OR REPLACE INTO edges(key, w) VALUES(?, ?)", pairs)
            if last_tick is not None:
                con.execute(
                    "INSERT INTO db_meta(key, value) VALUES('last_edge_tick', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (repr(float(last_tick)),),
                )
            con.execute(
                "INSERT INTO db_meta(key, value) VALUES('edges_rev', '1') "
                "ON CONFLICT(key) DO UPDATE SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT)"
            )

    def edges_stats(self, now: float, tau: float) -> tuple[int, float]:
        """(число рёбер, суммарный эффективный вес с учётом ленивого распада)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(w), 0.0) FROM edges"
            ).fetchone()
        tick_val = self.get_meta("last_edge_tick")
        factor = 1.0
        if tick_val is not None and now > float(tick_val):
            factor = float(np.exp(-(now - float(tick_val)) / float(tau)))
        return int(row[0]), round(float(row[1]) * factor, 4)

    # -- eligibility -------------------------------------------------------------------

    def elig_add(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        strength: float,
        created_at: float,
        source_ids: Sequence[int],
    ) -> None:
        """Write-through staging bind'а: событие сразу в БД, потерять его нельзя."""
        src_arr = np.asarray(src, dtype=np.int32)
        dst_arr = np.asarray(dst, dtype=np.int32)
        ids = sorted({int(i) for i in source_ids})
        with self._txn() as con:
            cur = con.execute(
                "INSERT INTO eligibility(src, dst, strength, created_at) VALUES(?,?,?,?)",
                (_pack_i32(src_arr), _pack_i32(dst_arr), float(strength), float(created_at)),
            )
            seq = cur.lastrowid
            con.executemany(
                "INSERT INTO elig_sources(seq, mem_id) VALUES(?, ?)",
                ((seq, mid) for mid in ids),
            )

    def elig_reward(self, mem_ids: Sequence[int], factor: float) -> int:
        """Умножить strength событий, затрагивающих данные следы. Возвращает #событий."""
        ids = sorted({int(i) for i in mem_ids})
        if not ids or factor == 1.0:
            return 0
        ph = ",".join("?" * len(ids))
        with self._txn() as con:
            cur = con.execute(
                "UPDATE eligibility SET strength = strength * ? WHERE seq IN ("
                f"SELECT DISTINCT seq FROM elig_sources WHERE mem_id IN ({ph}))",
                (float(factor), *ids),
            )
            return int(cur.rowcount)

    def elig_pending(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM eligibility").fetchone()
        return int(n)

    def elig_drain(self) -> list[tuple]:
        """Выкачать все события (с удалением) в формате EligibilityLog.load_state_dict."""
        with self._txn() as con:
            rows = con.execute(
                "SELECT seq, src, dst, strength, created_at FROM eligibility ORDER BY seq"
            ).fetchall()
            if rows:
                seqs = [r[0] for r in rows]
                ph = ",".join("?" * len(seqs))
                src_rows = con.execute(
                    f"SELECT seq, mem_id FROM elig_sources WHERE seq IN ({ph})", seqs
                ).fetchall()
                by_seq: dict[int, list[int]] = {}
                for seq, mid in src_rows:
                    by_seq.setdefault(int(seq), []).append(int(mid))
                con.execute("DELETE FROM eligibility")
            else:
                by_seq = {}
        events = []
        for seq, src_b, dst_b, strength, created in rows:
            src = np.frombuffer(src_b, dtype=np.int32)
            dst = np.frombuffer(dst_b, dtype=np.int32)
            events.append((src.tolist(), dst.tolist(), float(strength), float(created),
                           sorted(by_seq.get(int(seq), []))))
        return events

    # -- события (журнал пластичности) ---------------------------------------------------

    def event_append(self, event_type: str, fields: dict | None = None, ts: float = 0.0) -> None:
        data = dict(fields or {})
        with self._txn() as con:
            con.execute(
                "INSERT INTO events(ts, type, data) VALUES(?,?,?)",
                (float(ts), str(event_type), json.dumps(data, ensure_ascii=False)),
            )

    def event_count(self) -> int:
        with self._lock:
            (n,) = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return int(n)

    def iter_events(self) -> Iterator[dict]:
        """Порционная итерация журнала (замок на порцию — см. iter_active).
        Внешняя форма события прежняя: {ts, type, **data}; seq используется
        только как курсор и наружу не выдаётся."""
        last_seq = 0
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT seq, ts, type, data FROM events WHERE seq>? ORDER BY seq LIMIT 256",
                    (last_seq,),
                ).fetchall()
            if not rows:
                return
            for seq, ts, etype, data in rows:
                last_seq = int(seq)
                event = {"ts": float(ts), "type": etype}
                event.update(json.loads(data))
                yield event

    def events_prune(self, keep_rows: int) -> int:
        """Оставить последние `keep_rows` событий журнала; вернуть число удалённых.
        keep_rows <= 0 трактуется как «ничего не удалять»."""
        if keep_rows <= 0:
            return 0
        with self._txn() as con:
            con.execute(
                "DELETE FROM events WHERE seq NOT IN "
                "(SELECT seq FROM events ORDER BY seq DESC LIMIT ?)",
                (int(keep_rows),),
            )
            deleted = int(con.execute("SELECT changes()").fetchone()[0])
        return deleted

    def gate_decisions(self) -> dict[str, int]:
        """Счётчики решений гейта из write-событий (устойчиво к рестартам процессов)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT COALESCE(json_extract(data,'$.action'),'unknown'), COUNT(*) "
                "FROM events WHERE type='write' GROUP BY 1"
            ).fetchall()
        return {str(a): int(n) for a, n in rows}

    def recall_stats(self) -> tuple[int, float]:
        """(число recall-запросов, средняя латентность мс) из журнала событий."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(AVG(json_extract(data,'$.latency_ms')), 0.0) "
                "FROM events WHERE type='recall'"
            ).fetchone()
        return int(row[0]), round(float(row[1]), 3)
