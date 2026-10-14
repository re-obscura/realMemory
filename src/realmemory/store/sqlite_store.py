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

Схема/DDL и миграции — store/schema.py, мэппинг строк и FTS-утилиты —
store/rows.py; ниже ре-экспорты для совместимости (тесты и внешние
импорты ссылаются на имена из этого модуля).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..types import STATUS_ACTIVE, MemoryRecord
from ._sql import _bump_meta, _chunked, _qmarks, _upsert_meta
from .rows import (  # noqa: F401 - ре-экспорт для совместимости (тесты, hippocampus)
    _COLS,
    _COLS_LIGHT,
    _INSERT_SQL,
    _pack_f32,
    _pack_i32,
    _row_to_record,
    build_fts_query,
    scope_key,
    search_tokens,
    tokenize,
)
from .schema import (  # noqa: F401 - ре-экспорт для совместимости
    _FTS_STATEMENTS,
    _SCHEMA_STATEMENTS,
    SCHEMA_VERSION,
    _migrate_schema,
)


class StorageError(Exception):
    """Повреждение или недоступность хранилища."""


# батч журнала событий в iter_events (замок на порцию — как в iter_active)
_EVENTS_BATCH = 256


class MemoryStore:
    def __init__(self, path: str | Path, dim: int) -> None:
        self.path = Path(path)
        self.dim = int(dim)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                         isolation_level=None)
        except sqlite3.DatabaseError as exc:
            raise StorageError(f"не удалось открыть {self.path}: {exc}") from exc
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            with self._txn() as con:
                for stmt in _SCHEMA_STATEMENTS:
                    con.execute(stmt)
            # миграции цепочкой до SCHEMA_VERSION (v2: author + publications);
            # каждой мутации схемы предшествует страховочная копия базы
            self._schema_version = _migrate_schema(
                self._conn, self._lock, lambda: self.backup()
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
            self._conn.close()
            raise StorageError(f"не удалось открыть {self.path}: {exc}") from exc
        except BaseException:
            # битая миграция/бэкап не должны оставлять открытое соединение:
            # на Windows оно держит файловые локи и портит повторные открытия
            self._conn.close()
            raise

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
                    str(rec.author or ""),
                ),
            )
            self.bump_memories_rev(con)
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
            self.bump_memories_rev(con)

    def mark_superseded(self, memory_id: int, by_id: int, when: float) -> None:
        with self._txn() as con:
            con.execute(
                "UPDATE memories SET status='superseded', valid_to=?, superseded_by=? WHERE id=?",
                (float(when), int(by_id), int(memory_id)),
            )
            self.bump_memories_rev(con)

    def forget_traces(self, memory_ids: Sequence[int]) -> int:
        """Физическое удаление следов, забытых до потери осмысленности
        (retention ниже recall-пола и grace без подкреплений).

        Убирает строки memories (FTS-триггеры синхронно чистят индекс) и ссылки
        elig_sources, чтобы reward-матчинг не бился о мёртвые id. Рёбра L2 живут
        на уровне юнитов SDR и не атрибутируются по следам — они угасают сами.
        Журнал событий сохраняется (аудит). Возвращает число удалённых строк."""
        ids = sorted({int(i) for i in memory_ids})
        if not ids:
            return 0
        with self._txn() as con:
            for part in _chunked(ids):
                con.execute(
                    f"DELETE FROM elig_sources WHERE mem_id IN ({_qmarks(len(part))})",
                    part,
                )
            self.bump_memories_rev(con)
            remaining = 0
            for part in _chunked(ids):
                con.executemany(
                    "DELETE FROM memories WHERE id=?", ((i,) for i in part)
                )
                remaining += int(con.execute(
                    f"SELECT COUNT(*) FROM memories WHERE id IN ({_qmarks(len(part))})",
                    part,
                ).fetchone()[0])
        return len(ids) - remaining

    def adjust_base(self, memory_id: int, base_strength: float, updated_at: float) -> None:
        """Ослабление следа без сброса таймера подкрепления (негативный feedback)."""
        with self._txn() as con:
            con.execute(
                "UPDATE memories SET base_strength=?, updated_at=? WHERE id=?",
                (float(base_strength), float(updated_at), int(memory_id)),
            )
            self.bump_memories_rev(con)

    def update_trace_atomic(
        self,
        memory_id: int,
        compute: Callable[[MemoryRecord], tuple[float, int, float, str | None]],
        *,
        updated_at: float | None = None,
        reset_timer: bool = True,
    ) -> bool:
        """Чтение записи и запись новой динамики в ОДНОЙ BEGIN IMMEDIATE-транзакции.

        compute получает актуальную строку и возвращает
        (base_strength, reinforced_count, last_reinforced_at, kind|None);
        kind=None — тип не менять. Параметры формы записи:
        reset_timer=True (подкрепление/промоция) пишет last_reinforced_at из
        compute; reset_timer=False (негативный feedback) таймер не трогает;
        updated_at по умолчанию следует за last_reinforced_at.

        Атомарность здесь — про деньги: чтение вне транзакции с записью
        абсолютных значений теряло подкрепления двух одновременно
        сработавших процессов (об(rec).reinforced_count + 1 вместо N+1).
        False — след не существует."""
        with self._txn() as con:
            row = con.execute(
                f"SELECT {_COLS} FROM memories WHERE id=?", (int(memory_id),)
            ).fetchone()
            if row is None:
                return False
            rec = _row_to_record(row, self.dim)
            base, count, last_reinforced, kind = compute(rec)
            # updated_at по умолчанию следует за ВЫЧИСЛЕННЫМ таймером (как
            # update_trace), а не за старым значением строки
            stamp = float(last_reinforced if updated_at is None else updated_at)
            sets = "base_strength=?, reinforced_count=?, updated_at=?"
            params: list = [float(base), int(count), stamp]
            if reset_timer:
                sets += ", last_reinforced_at=?"
                params.append(float(last_reinforced))
            if kind is not None and kind != rec.kind:
                sets += ", kind=?"
                params.append(kind)
            params.append(int(memory_id))
            con.execute(f"UPDATE memories SET {sets} WHERE id=?", params)
            self.bump_memories_rev(con)
        return True

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
            _upsert_meta(con, key, str(value))

    def bump_meta_int(self, key: str, delta: int = 1) -> int:
        """Атомарный инкремент целочисленного счётчика в db_meta; возвращает новое значение."""
        with self._txn() as con:
            _bump_meta(con, key, delta)
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
        by_id: dict[int, MemoryRecord] = {}
        with self._lock:
            for part in _chunked(ids):
                rows = self._conn.execute(
                    f"SELECT {_COLS} FROM memories WHERE id IN ({_qmarks(len(part))})",
                    part,
                ).fetchall()
                for r in rows:
                    by_id[int(r[0])] = _row_to_record(r, self.dim)
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

    def iter_active_light(self, batch: int = 512) -> Iterator[tuple]:
        """Порционная итерация (id, kind, created_at, last_reinforced_at,
        base_strength, reinforced_count) активных следов — без декодирования
        блобов эмбеддинга и SDR. Проходам «сна» (промоция/GC/метрики) нужна
        только динамика следа; полный декодирует iter_active."""
        last_id = 0
        while True:
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT {_COLS_LIGHT} FROM memories WHERE status=? AND id>? "
                    "ORDER BY id LIMIT ?",
                    (STATUS_ACTIVE, last_id, int(batch)),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                yield (int(row[0]), str(row[1]), float(row[2]), float(row[3]),
                       float(row[4]), int(row[5]))
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

    def canonical_scope(self, name: str) -> str:
        """Существующий скоуп с тем же каноническим ключом, иначе имя как есть.

        При нескольких совпадающих вариантах выигрывает самый населённый
        (стабильно и предсказуемо) — так «cpm_backend» сходится к уже
        живущему «CPM.Backend», а не плодит второй вариант."""
        key = scope_key(name)
        if not key:
            return name
        with self._lock:
            rows = self._conn.execute(
                "SELECT scope, COUNT(*) AS n FROM memories"
                " WHERE status='active' GROUP BY scope"
            ).fetchall()
        best: tuple[int, str] | None = None
        for scope, n in rows:
            if scope_key(str(scope)) == key:
                cand = (int(n), str(scope))
                if best is None or cand > best:
                    best = cand
        return best[1] if best else name

    def wal_checkpoint(self) -> None:
        """Пассивный чекпойнт WAL: не блокирует никого, но возвращает
        журнал в основную базу, когда читателей нет. Без давления снаружи
        (один долгоживущий MCP-процесс) WAL рос вдвое больше самой базы."""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

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
        return self._fts_enabled

    def fts_error(self) -> str | None:
        return self._fts_error

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

    # -- синхронизация registry -------------------------------------------------

    def publications_unsynced(self) -> list[tuple]:
        """Строки registry, ещё не подтверждённые сетью (и активные, и
        tombstones — отзыв тоже обязан доехать). Порядок по времени."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, trace_id, project, author, published_at,"
                " revoked_at, content_hash FROM publications"
                " WHERE synced_at IS NULL ORDER BY published_at"
            ).fetchall()
        return [tuple(r) for r in rows]

    def publications_synced_active(self) -> list[tuple]:
        """Доставленные и всё ещё активные публикации (для детекции
        «след локально забыт после доставки» — GC-автоотзыва)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, trace_id, project, author, published_at,"
                " revoked_at, content_hash FROM publications"
                " WHERE synced_at IS NOT NULL AND revoked_at IS NULL"
                " ORDER BY published_at"
            ).fetchall()
        return [tuple(r) for r in rows]

    def publication_mark_synced_force(self, publication_ids: Sequence[str],
                                      when: float) -> int:
        """Проставить synced_at БЕЗ условия «был NULL» (повторная отметка
        после форс-отзыва)."""
        ids = [str(i) for i in publication_ids]
        if not ids:
            return 0
        touched = 0
        with self._txn() as con:
            for part in _chunked(ids):
                cur = con.execute(
                    f"UPDATE publications SET synced_at=? WHERE id IN ({_qmarks(len(part))})",
                    (float(when), *part),
                )
                touched += int(cur.rowcount)
        return touched

    def publication_mark_synced(self, publication_ids: Sequence[str],
                                when: float) -> int:
        ids = [str(i) for i in publication_ids]
        if not ids:
            return 0
        touched = 0
        with self._txn() as con:
            for part in _chunked(ids):
                cur = con.execute(
                    f"UPDATE publications SET synced_at=? "
                    f"WHERE id IN ({_qmarks(len(part))}) AND synced_at IS NULL",
                    (float(when), *part),
                )
                touched += int(cur.rowcount)
        return touched

    def memories_rev(self) -> int:
        """Версия набора следов (вставки/правки/отзывы/забывание): дешёвая
        детекция чужих изменений для кэшей процесса — аналог edges_rev.
        -1, пока счётчик не заводился (legacy-базы)."""
        val = self.get_meta("memories_rev")
        return int(val) if val is not None else -1

    @staticmethod
    def bump_memories_rev(con: sqlite3.Connection) -> None:
        """Поднять счётчик ВНУТРИ текущей транзакции мутации следов."""
        _bump_meta(con, "memories_rev")

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
            _upsert_meta(con, "last_edge_tick", repr(float(now)))
            _bump_meta(con, "edges_rev")
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
                _upsert_meta(con, "last_edge_tick", repr(float(last_tick)))
            _bump_meta(con, "edges_rev")

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
                by_seq: dict[int, list[int]] = {}
                for part in _chunked(seqs):
                    src_rows = con.execute(
                        f"SELECT seq, mem_id FROM elig_sources WHERE seq IN ({_qmarks(len(part))})",
                        part,
                    ).fetchall()
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

    # Счётчики наблюдаемости растут в транзакции каждого события, чтобы
    # stats()/«сон» не делали полный json_extract-скан журнала. Ключи начинают
    # существовать с первого bump; legacy-базы засеивает seed_obs_counters().
    _OBS_ACTIONS = ("create", "reinforce", "link")

    def _bump_obs_counters(self, con: sqlite3.Connection, event_type: str,
                           data: dict) -> None:
        if event_type == "write":
            _bump_meta(con, "obs_writes_total")
            action = str(data.get("action", "unknown"))
            if action in self._OBS_ACTIONS:
                _bump_meta(con, f"obs_action_{action}_total")
        elif event_type == "recall":
            _bump_meta(con, "obs_recalls_total")
            latency = data.get("latency_ms")
            if latency is not None:
                _bump_meta(con, "obs_recall_ms_sum",
                        float(latency), is_float=True)

    def event_append(self, event_type: str, fields: dict | None = None, ts: float = 0.0) -> None:
        data = dict(fields or {})
        with self._txn() as con:
            con.execute(
                "INSERT INTO events(ts, type, data) VALUES(?,?,?)",
                (float(ts), str(event_type), json.dumps(data, ensure_ascii=False)),
            )
            self._bump_obs_counters(con, str(event_type), data)

    def obs_counters(self) -> dict | None:
        """Инкрементальные счётчики наблюдаемости; None — база ещё не засеяна."""
        writes = self.get_meta("obs_writes_total")
        recalls = self.get_meta("obs_recalls_total")
        ms_sum = self.get_meta("obs_recall_ms_sum")
        if writes is None or recalls is None or ms_sum is None:
            return None
        n_recalls = int(recalls)
        return {
            "writes": int(writes),
            "recalls": n_recalls,
            "avg_recall_ms": round(float(ms_sum) / n_recalls, 3) if n_recalls else 0.0,
            "actions": {a: int(self.get_meta(f"obs_action_{a}_total") or 0)
                        for a in self._OBS_ACTIONS},
        }

    def seed_obs_counters(self) -> dict:
        """Однократный посев счётчиков сканом журнала (базы до их введения).

        Гонка с параллельным bump стоит максимум пару событий точности —
        это наблюдаемость, а не деньги; дальше счётчики монотонны.
        """
        actions = self.gate_decisions()
        recalls, avg_ms = self.recall_stats()
        rows = [
            ("obs_writes_total", sum(actions.values())),
            ("obs_recalls_total", recalls),
            ("obs_recall_ms_sum", round(avg_ms * recalls, 3)),
        ] + [(f"obs_action_{a}_total", actions.get(a, 0)) for a in self._OBS_ACTIONS]
        with self._txn() as con:
            for key, val in rows:
                _upsert_meta(con, key, str(val))
        result = self.obs_counters()
        if result is None:  # посев выше обязан завести все ключи
            raise StorageError("seed_obs_counters не засеял счётчики наблюдаемости")
        return result

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
                    "SELECT seq, ts, type, data FROM events "
                    f"WHERE seq>? ORDER BY seq LIMIT {_EVENTS_BATCH}",
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

    @property
    def schema_version(self) -> str:
        return str(self._schema_version)

    # -- командный слой: registry публикаций ----------------------------------------

    def publication_add(self, trace_id: int, project: str, author: str,
                        published_at: float, content_hash: str) -> str:
        """Зарегистрировать публикацию следа; возвращает publication_id.
        Отзыв — отдельная запись с revoked_at; повторная публикация того же
        следа создаёт новую строку (история решении сохраняется)."""
        pid = uuid.uuid4().hex
        with self._txn() as con:
            con.execute(
                "INSERT INTO publications(id, trace_id, project, author,"
                " published_at, revoked_at, content_hash) VALUES(?,?,?,?,?,?,?)",
                (pid, int(trace_id), str(project or ""), str(author or ""),
                 float(published_at), None, str(content_hash)),
            )
        return pid

    def publication_retract(self, now: float,
                            publication_ids: Sequence[str] = (),
                            trace_ids: Sequence[int] = ()) -> int:
        """Поставить tombstone на активные публикации указанных id либо следов."""
        pub_ids = [str(i) for i in publication_ids if str(i)]
        tr_ids = [int(i) for i in trace_ids]
        touched = 0
        with self._txn() as con:
            for col, params in (("id", pub_ids), ("trace_id", tr_ids)):
                if not params:
                    continue
                for part in _chunked(params):
                    cur = con.execute(
                        f"UPDATE publications SET revoked_at=?, synced_at=NULL "
                        f"WHERE revoked_at IS NULL AND {col} IN ({_qmarks(len(part))})",
                        (float(now), *part),
                    )
                    touched += int(cur.rowcount)
        return touched

    def publications_active(self) -> list[tuple]:
        """Активные (не отозванные) публикации в хронологическом порядке."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, trace_id, project, author, published_at, content_hash "
                "FROM publications WHERE revoked_at IS NULL ORDER BY published_at"
            ).fetchall()
        return [tuple(r) for r in rows]

    def publications_tombstones(self) -> list[tuple]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, trace_id, project, author, published_at, revoked_at "
                "FROM publications WHERE revoked_at IS NOT NULL "
                "ORDER BY revoked_at"
            ).fetchall()
        return [tuple(r) for r in rows]
