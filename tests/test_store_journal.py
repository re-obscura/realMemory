"""Контракты хранилища: SQLite roundtrip, суперсед-цепочки, JSONL-журнал."""
import numpy as np

from realmemory.store.journal import Journal
from realmemory.store.sqlite_store import MemoryStore
from realmemory.types import MemoryRecord


def _rec(i: int, dim=8, k=4) -> MemoryRecord:
    rng = np.random.default_rng(100 + i)
    return MemoryRecord(
        id=None,
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


def test_insert_get_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    rec = _rec(1)
    rid = store.insert(rec)
    assert rid >= 1
    got = store.get(rid)
    assert got.text == "fact number 1"
    assert np.array_equal(got.embedding, rec.embedding)
    assert np.array_equal(got.sdr, rec.sdr)
    assert got.meta == {"n": 1}
    assert got.base_strength == 0.5 and got.status == "active"
    store.close()


def test_get_missing_returns_none(tmp_path):
    store = MemoryStore(tmp_path / "m.db", 8)
    assert store.get(42) is None
    store.close()


def test_get_many_preserves_order_skips_missing(tmp_path):
    store = MemoryStore(tmp_path / "m.db", 8)
    ids = [store.insert(_rec(i)) for i in range(5)]
    got = store.get_many([ids[3], 999, ids[0]])
    assert [r.id for r in got] == [ids[3], ids[0]]
    store.close()


def test_update_trace_and_supersede(tmp_path):
    store = MemoryStore(tmp_path / "m.db", 8)
    a = store.insert(_rec(1))
    b = store.insert(_rec(2))
    store.update_trace(a, base_strength=0.8, reinforced_count=3,
                       last_reinforced_at=55.0, kind="semantic")
    ra = store.get(a)
    assert ra.base_strength == 0.8 and ra.reinforced_count == 3
    assert ra.kind == "semantic" and ra.last_reinforced_at == 55.0
    store.mark_superseded(a, by_id=b, when=99.0)
    ra = store.get(a)
    assert ra.status == "superseded" and ra.superseded_by == b and ra.valid_to == 99.0
    assert [r.id for r in store.iter_active()] == [b]
    assert store.count(status="active") == 1
    assert store.count(kind="semantic") == 1
    assert store.all_active_ids().tolist() == [b]
    store.close()


def test_journal_roundtrip(tmp_path):
    j = Journal(tmp_path / "j.jsonl")
    j.append("write", id=1, t=1.5)
    j.append("link", ids=[1, 2])
    events = list(j.events())
    assert events[0]["type"] == "write" and events[0]["id"] == 1
    assert events[1]["ids"] == [1, 2]
    assert j.count() == 2
    reopened = Journal(tmp_path / "j.jsonl")
    assert list(reopened.events()) == events
