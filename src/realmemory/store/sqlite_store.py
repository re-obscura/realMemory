"""SQLite-хранилище следов. Контракт: docs/CONTRACTS.md.

WAL-режим, один процесс на базу (блокировка потоков внутри класса).
Сериализация: embedding -> float32 blob, sdr -> int32 blob (отсортирован).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from ..types import STATUS_ACTIVE, MemoryRecord

_SCHEMA = """
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
    superseded_by INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);

CREATE TABLE IF NOT EXISTS db_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_INSERT_SQL = (
    "INSERT INTO memories(text,kind,status,meta,embedding,sdr,"
    "created_at,updated_at,reinforced_count,last_reinforced_at,base_strength,"
    "valid_from,valid_to,superseded_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

_COLS = (
    "id,text,kind,status,meta,embedding,sdr,created_at,updated_at,"
    "reinforced_count,last_reinforced_at,base_strength,valid_from,valid_to,superseded_by"
)


class StorageError(Exception):
    """Повреждение или недоступность хранилища."""


def _pack_f32(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _pack_i32(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.int32).tobytes()



def _row_to_record(row: tuple, dim: int) -> MemoryRecord:
    (rid, text, kind, status, meta_json, emb_b, sdr_b, created, updated,
     rcount, rlast, base, vfrom, vto, sby) = row
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
    )


class MemoryStore:
    def __init__(self, path: str | Path, dim: int) -> None:
        self.path = Path(path)
        self.dim = int(dim)
        self._lock = threading.Lock()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.DatabaseError as exc:
            raise StorageError(f"не удалось открыть {self.path}: {exc}") from exc

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

    # -- запись ---------------------------------------------------------------

    def insert(self, rec: MemoryRecord) -> int:
        with self._lock:
            try:
                cur = self._conn.execute(
                    _INSERT_SQL,
                    (
                        rec.text, rec.kind, rec.status, json.dumps(rec.meta, ensure_ascii=False),
                        _pack_f32(rec.embedding), _pack_i32(rec.sdr),
                        float(rec.created_at), float(rec.updated_at), int(rec.reinforced_count),
                        float(rec.last_reinforced_at), float(rec.base_strength),
                        float(rec.valid_from),
                        None if rec.valid_to is None else float(rec.valid_to),
                        rec.superseded_by,
                    ),
                )
                rowid = cur.lastrowid
                self._conn.commit()
                if rowid is None:  # pragma: no cover - не бывает после успешного INSERT
                    raise StorageError("INSERT не вернул rowid")
                return int(rowid)
            except sqlite3.DatabaseError as exc:
                raise StorageError(f"вставка не удалась: {exc}") from exc

    def update_trace(
        self,
        memory_id: int,
        base_strength: float,
        reinforced_count: int,
        last_reinforced_at: float,
        kind: str | None = None,
    ) -> None:
        with self._lock:
            sets = ("base_strength=?, reinforced_count=?, last_reinforced_at=?, updated_at=?")
            params: list = [float(base_strength), int(reinforced_count),
                            float(last_reinforced_at), float(last_reinforced_at)]
            if kind is not None:
                sets += ", kind=?"
                params.append(kind)
            params.append(int(memory_id))
            try:
                self._conn.execute(f"UPDATE memories SET {sets} WHERE id=?", params)
                self._conn.commit()
            except sqlite3.DatabaseError as exc:
                raise StorageError(f"обновление не удалось: {exc}") from exc

    def mark_superseded(self, memory_id: int, by_id: int, when: float) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE memories SET status='superseded', valid_to=?, superseded_by=? WHERE id=?",
                    (float(when), int(by_id), int(memory_id)),
                )
                self._conn.commit()
            except sqlite3.DatabaseError as exc:
                raise StorageError(f"суперсед не удался: {exc}") from exc

    def adjust_base(self, memory_id: int, base_strength: float, updated_at: float) -> None:
        """Ослабление следа без сброса таймера подкрепления (негативный feedback)."""
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE memories SET base_strength=?, updated_at=? WHERE id=?",
                    (float(base_strength), float(updated_at), int(memory_id)),
                )
                self._conn.commit()
            except sqlite3.DatabaseError as exc:
                raise StorageError(f"ослабление не удалось: {exc}") from exc

    # -- мета базы --------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM db_meta WHERE key=?", (key,)
            ).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO db_meta(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, str(value)),
                )
                self._conn.commit()
            except sqlite3.DatabaseError as exc:
                raise StorageError(f"запись db_meta не удалась: {exc}") from exc

    # -- чтение -----------------------------------------------------------------

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
        cur = self._conn.execute(
            f"SELECT {_COLS} FROM memories WHERE status=? ORDER BY id", (STATUS_ACTIVE,)
        )
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            for row in rows:
                yield _row_to_record(row, self.dim)

    def all_active_ids(self) -> np.ndarray:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM memories WHERE status=?", (STATUS_ACTIVE,)
            ).fetchall()
        return np.asarray([r[0] for r in rows], dtype=np.int64)

    def count(self, status: str | None = None, kind: str | None = None) -> int:
        query = "SELECT COUNT(*) FROM memories"
        conds, params = [], []
        if status is not None:
            conds.append("status=?")
            params.append(status)
        if kind is not None:
            conds.append("kind=?")
            params.append(kind)
        if conds:
            query += " WHERE " + " AND ".join(conds)
        with self._lock:
            (n,) = self._conn.execute(query, params).fetchone()
        return int(n)

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
