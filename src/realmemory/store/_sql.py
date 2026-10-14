"""Общие SQL-идиомы хранилища: upsert db_meta, бампы счётчиков, чанкинг IN."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence

# SQLite ограничивает число параметров запроса: длинные IN-списки режем
_SQL_CHUNK = 500


def _upsert_meta(con: sqlite3.Connection, key: str, value: str) -> None:
    """Вставить/заменить ключ db_meta внутри текущей транзакции."""
    con.execute(
        "INSERT INTO db_meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def _bump_meta(con: sqlite3.Connection, key: str, delta: float = 1,
               *, is_float: bool = False) -> None:
    """Атомарно прибавить delta к счётчику db_meta внутри текущей транзакции."""
    if is_float:
        con.execute(
            "INSERT INTO db_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value=CAST(CAST(value AS REAL)+? AS TEXT)",
            (key, repr(float(delta)), float(delta)),
        )
    else:
        con.execute(
            "INSERT INTO db_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value=CAST(CAST(value AS INTEGER)+? AS TEXT)",
            (key, str(int(delta)), int(delta)),
        )


def _chunked(ids: Sequence[int | str]) -> Iterator[list[int | str]]:
    """Порции по _SQL_CHUNK: SQLite ограничивает число параметров запроса,
    длинные IN-списки режутся на части."""
    for start in range(0, len(ids), _SQL_CHUNK):
        yield list(ids[start:start + _SQL_CHUNK])


def _qmarks(n: int) -> str:
    return ",".join("?" * n)
