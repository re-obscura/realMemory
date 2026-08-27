"""Эксплуатационные гарантии: бэкапы, schema_version, видимость отказов хуков."""
import sqlite3

import numpy as np
import pytest

from realmemory import Hippocampus
from realmemory.store.sqlite_store import MemoryStore
from realmemory.types import MemoryRecord


def _rec(i: int, dim=8, k=4) -> MemoryRecord:
    rng = np.random.default_rng(200 + i)
    return MemoryRecord(
        id=None,
        text=f"ops fact {i}",
        kind="episodic",
        status="active",
        meta={},
        embedding=rng.standard_normal(dim).astype(np.float32),
        sdr=np.sort(rng.choice(dim, size=k, replace=False)).astype(np.int32),
        created_at=float(i),
        updated_at=float(i),
        reinforced_count=0,
        last_reinforced_at=float(i),
        base_strength=0.5,
        valid_from=float(i),
    )


def test_backup_readable_and_rotates(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    store.insert(_rec(1))
    bdir = tmp_path / "backups"

    first = store.backup(keep=3)
    assert first.exists()
    # подсаживаем «старые» копии для проверки ротации по имени-времени
    for name in ("memory-20200101-000000.db", "memory-20200101-000001.db",
                 "memory-20200101-000002.db"):
        (bdir / name).write_bytes(b"fake")
    store.backup(keep=3)

    files = sorted(p.name for p in bdir.glob("memory-*.db"))
    assert len(files) == 3
    assert "memory-20200101-000000.db" not in files  # самая старая удалена

    # свежая копия читается и содержит данные
    con = sqlite3.connect(str(first))
    try:
        (n,) = con.execute("SELECT COUNT(*) FROM memories").fetchone()
    finally:
        con.close()
    assert n == 1
    store.close()


def test_backup_keep_zero_disables_rotation(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    store.insert(_rec(1))
    bdir = tmp_path / "backups"
    bdir.mkdir()
    fake = bdir / "memory-20190101-000000.db"
    fake.write_bytes(b"old")
    p1 = store.backup(keep=0)
    p2 = store.backup(keep=0)
    names = sorted(p.name for p in (tmp_path / "backups").glob("*.db"))
    assert "memory-20190101-000000.db" in names  # ротация не трогает чужие файлы
    # keep=0 ничего не удаляет; два вызова в одну секунду делят имя файла
    if p1 != p2:
        assert {p1.name, p2.name} <= set(names)
    for p in (p1, p2):
        con = sqlite3.connect(str(p))
        try:
            (n,) = con.execute("SELECT COUNT(*) FROM memories").fetchone()
        finally:
            con.close()
        assert n == 1
    store.close()


def test_schema_version_recorded(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    assert store.get_meta("schema_version") == "3"
    # повторное открытие не понижает версию
    reopened = MemoryStore(tmp_path / "m.db", dim=8)
    assert reopened.get_meta("schema_version") == "3"
    # v1 -> v2: колонка author и registry публикаций на месте
    assert {"author", "scope"} <= {
        r[1] for r in sqlite3.connect(str(tmp_path / "m.db")).execute(
            "PRAGMA table_info(memories)").fetchall()}
    reopened.close()
    store.close()


def test_consolidate_makes_backup(hippo):
    hippo.remember("korvex kv001 qixa")
    hippo.consolidate()
    backups = list((hippo.path / "backups").glob("memory-*.db"))
    assert len(backups) >= 1
    con = sqlite3.connect(str(backups[-1]))
    try:
        (n,) = con.execute("SELECT COUNT(*) FROM memories").fetchone()
    finally:
        con.close()
    assert n >= 1


def test_hook_failure_visible_in_stderr(tmp_path, capsys):
    """Отказ хука не ломает сессию агента (код 0), но виден в stderr."""
    from realmemory.hook_cli import main as hook_main

    blocker = tmp_path / "notadir"
    blocker.write_text("занято файлом")
    with pytest.raises(SystemExit) as e:
        hook_main(["brief", "--path", str(blocker)])
    assert e.value.code == 0
    assert "failed" in capsys.readouterr().err


def test_sleep_failure_reported_as_event(tmp_path, tiny_cfg, clock, capsys, monkeypatch):
    """Если консолидация упала — событие hook_error попадает в журнал."""
    import json as _json

    from realmemory.hook_cli import main as hook_main

    h = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    h.remember("korvex kv9 qixa")
    path = str(h.path)
    h.close()

    def broken_open(*a, **kw):
        raise RuntimeError("эмуляция падения сна")

    monkeypatch.setattr("realmemory.hook_cli._open", broken_open)
    with pytest.raises(SystemExit) as e:
        hook_main(["sleep", "--path", path])
    assert e.value.code == 0
    assert "эмуляция падения сна" in capsys.readouterr().err

    con = sqlite3.connect(f"{path}/memory.db")
    try:
        rows = con.execute(
            "SELECT data FROM events WHERE type='hook_error'"
        ).fetchall()
    finally:
        con.close()
    assert any("эмуляция падения сна" in r[0] for r in rows)
    assert _json.loads(rows[0][0])["cmd"] == "sleep"
