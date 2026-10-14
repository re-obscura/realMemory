"""Регрессионные тесты модернизационного прохода v0.9.0.

Фаза 1: утёкшее соединение при неудачном открытии хранилища;
атомарный read-modify-write динамики следа (два процесса, подкрепляющие
один след, не должны терять инкременты друг друга)."""
import numpy as np
import pytest

from realmemory import Hippocampus
from realmemory.store.sqlite_store import MemoryStore, StorageError
from realmemory.types import MemoryRecord


def _rec(i: int, dim=8, k=4) -> MemoryRecord:
    rng = np.random.default_rng(100 + i)
    return MemoryRecord(
        id=0,  # id назначит store при вставке
        text=f"fact number {i}",
        kind="episodic",
        status="active",
        meta={"n": i},
        embedding=rng.standard_normal(dim).astype(np.float32),
        sdr=np.sort(rng.choice(dim, size=k, replace=False)).astype(np.int32),
        created_at=float(i),
        updated_at=float(i),
        reinforced_count=0,
        last_reinforced_at=float(i),
        base_strength=0.5,
        valid_from=float(i),
    )


def test_failed_open_closes_connection(tmp_path):
    db = tmp_path / "broken.db"
    db.write_bytes(b"this is definitely not a sqlite database" * 20)
    with pytest.raises(StorageError):
        MemoryStore(db, dim=8)
    # утёкшее соединение держало бы файл: на Windows unlink поднял бы
    # PermissionError и отравил повторные открытия каталога
    db.unlink()


def test_atomic_reinforce_keeps_both_increments(tmp_path):
    db = tmp_path / "m.db"
    with MemoryStore(db, dim=8) as a, MemoryStore(db, dim=8) as b:
        rid = a.insert(_rec(1))
        # оба процесса прочитали след ДО чужой записи (устаревший снимок,
        # reinforced_count=0); форма «прочитал снаружи — записал абсолют»
        # теряла второй инкремент: оба писали 1
        a.get(rid)
        b.get(rid)

        def bump(rec: MemoryRecord) -> tuple[float, int, float, str | None]:
            return (rec.base_strength, rec.reinforced_count + 1,
                    rec.last_reinforced_at, None)

        assert a.update_trace_atomic(rid, bump)
        assert b.update_trace_atomic(rid, bump)
        assert b.get(rid).reinforced_count == 2


def test_atomic_update_promotion_shape(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    rid = store.insert(_rec(1))
    # промоция: тип меняется, таймер и updated_at следуют за last_reinforced_at
    touched = store.update_trace_atomic(
        rid,
        lambda rec: (rec.base_strength, rec.reinforced_count,
                     rec.last_reinforced_at, "semantic"),
    )
    assert touched
    got = store.get(rid)
    assert got.kind == "semantic"
    assert got.updated_at == got.last_reinforced_at == 1.0
    store.close()


def test_atomic_weaken_keeps_timer(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    rid = store.insert(_rec(1))
    # негативный feedback: base ослабляется, updated_at свежий,
    # таймер подкрепления не сбрасывается, счётчик не трогается
    store.update_trace_atomic(
        rid,
        lambda rec: (rec.base_strength * 0.5, rec.reinforced_count,
                     rec.last_reinforced_at, None),
        updated_at=99.0,
        reset_timer=False,
    )
    got = store.get(rid)
    assert got.base_strength == 0.25
    assert got.updated_at == 99.0
    assert got.last_reinforced_at == 1.0
    assert got.reinforced_count == 0
    store.close()


def test_atomic_update_missing_trace_returns_false(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    assert store.update_trace_atomic(
        999, lambda rec: (1.0, 1, 1.0, None)
    ) is False
    store.close()


def test_atomic_reinforce_updates_timestamp_fresh(tmp_path):
    """updated_at следует за ВЫЧИСЛЕННЫМ таймером подкрепления, а не за старым
    значением строки: обратное сделало бы MAX(updated_at) слепым к подкреплениям
    (троттлинг «сна» в hook_cli перестал бы их замечать)."""
    store = MemoryStore(tmp_path / "m.db", dim=8)
    rid = store.insert(_rec(1))  # last_reinforced_at = 1.0
    store.update_trace_atomic(
        rid,
        lambda rec: (rec.base_strength, rec.reinforced_count + 1, 500.0, None),
    )
    got = store.get(rid)
    assert got.last_reinforced_at == 500.0
    assert got.updated_at == 500.0
    store.close()


def test_feedback_two_instances_no_loss(tmp_path, tiny_cfg, clock):
    h1 = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    h2 = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    try:
        res = h1.remember("факт про атомарность подкреплений", scope="proj")
        assert h1.feedback([res.memory_id], 1.0) == 1
        assert h2.feedback([res.memory_id], 1.0) == 1
        rec = h2.store.get(res.memory_id)
        assert rec.reinforced_count == 2
        # несуществующий след не считается затронутым
        assert h1.feedback([10_000], 1.0) == 0
    finally:
        h1.close()
        h2.close()
