"""Контракты хранилища: SQLite roundtrip, суперсед-цепочки, события,
eligibility write-through, рёбра L2, целочисленные счётчики db_meta."""
import json

import numpy as np
import pytest

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


# -- события (журнал пластичности) -------------------------------------------------

def test_events_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "m.db", 8)
    store.event_append("write", {"id": 1}, ts=1.5)
    store.event_append("link", {"ids": [1, 2]}, ts=2.0)
    events = list(store.iter_events())
    assert events[0]["type"] == "write" and events[0]["id"] == 1 and events[0]["ts"] == 1.5
    assert events[1]["ids"] == [1, 2]
    assert store.event_count() == 2
    reopened = MemoryStore(tmp_path / "m.db", 8)
    assert list(reopened.iter_events()) == events
    store.close()
    reopened.close()


def test_gate_decisions_and_recall_stats(tmp_path):
    store = MemoryStore(tmp_path / "m.db", 8)
    store.event_append("write", {"action": "create"}, ts=1.0)
    store.event_append("write", {"action": "reinforce"}, ts=2.0)
    store.event_append("write", {}, ts=3.0)  # legacy-импорт без action
    store.event_append("recall", {"latency_ms": 10.0}, ts=4.0)
    store.event_append("recall", {"latency_ms": 20.0}, ts=5.0)
    decisions = store.gate_decisions()
    assert decisions == {"create": 1, "reinforce": 1, "unknown": 1}
    count, avg_ms = store.recall_stats()
    assert count == 2 and avg_ms == 15.0
    store.close()


# -- eligibility --------------------------------------------------------------------

def test_elig_add_reward_drain(tmp_path):
    store = MemoryStore(tmp_path / "m.db", 8)
    src = np.arange(4, dtype=np.int32)
    dst = np.arange(10, 14, dtype=np.int32)
    store.elig_add(src, dst, strength=1.0, created_at=100.0, source_ids=[7, 9])
    store.elig_add(src + 20, dst + 20, strength=2.0, created_at=110.0, source_ids=[9])
    assert store.elig_pending() == 2

    touched = store.elig_reward([9], factor=2.0)  # оба события касаются следа 9
    assert touched == 2
    assert store.elig_reward([12345], factor=2.0) == 0
    assert store.elig_reward([7], factor=1.0) == 0  # нейтральный фактор

    rows = {tuple(sorted(r[4])): r for r in store.elig_drain()}
    assert store.elig_pending() == 0 and store.elig_drain() == []
    e79 = rows[(7, 9)]
    assert list(e79[0]) == src.tolist() and list(e79[1]) == dst.tolist()
    assert e79[2] == 2.0 and e79[3] == 100.0
    e9 = rows[(9,)]
    assert e9[2] == 4.0
    store.close()


# -- рёбра L2 -------------------------------------------------------------------------

def test_edges_apply_decay_prune_accumulate(tmp_path):
    stride = 100
    store = MemoryStore(tmp_path / "m.db", 8)
    src = np.array([1, 2], dtype=np.int64)
    dst = np.array([3, 4], dtype=np.int64)
    w = np.array([1.0, 1.0], dtype=np.float32)

    committed, pruned = store.edges_apply(src, dst, w, now=100.0, tau=10.0,
                                          min_weight=0.05, stride=stride)
    assert (committed, pruned) == (2, 0)
    keys, ws = store.edges_load()
    assert keys.tolist() == [1 * stride + 3, 2 * stride + 4]
    assert np.allclose(ws, [1.0, 1.0])
    rev1 = store.edges_rev()

    # аккумуляция в существующее ребро + распад за dt=tau/2 (фактор ~0.6065)
    committed, pruned = store.edges_apply(
        np.array([1], dtype=np.int64), np.array([3], dtype=np.int64),
        np.array([0.5], dtype=np.float32), now=105.0, tau=10.0,
        min_weight=0.05, stride=stride)
    assert committed == 1 and pruned == 0
    keys, ws = store.edges_load()
    assert dict(zip(keys.tolist(), ws.tolist()))[1 * stride + 3] == pytest.approx(1.1065, rel=1e-3)
    assert store.edges_rev() > rev1

    # полный распад и обрезка слабых
    store.edges_apply(np.empty(0), np.empty(0), np.empty(0), now=160.0, tau=10.0,
                      min_weight=0.05, stride=stride)
    count, total_w = store.edges_stats(now=160.0, tau=10.0)
    assert count == 0 and total_w == 0.0
    store.close()


def test_edges_import_preserves_tick(tmp_path):
    store = MemoryStore(tmp_path / "m.db", 8)
    store.edges_import(np.array([5, 6], dtype=np.int64),
                       np.array([0.4, 0.2], dtype=np.float32), last_tick=777.0)
    count, _ = store.edges_stats(now=800.0, tau=1e9)
    assert count == 2
    assert float(store.get_meta("last_edge_tick")) == 777.0
    store.close()


# -- счётчики db_meta -------------------------------------------------------------------

def test_bump_consume_meta_int(tmp_path):
    store = MemoryStore(tmp_path / "m.db", 8)
    assert store.consume_meta_int("absent") == 0
    assert store.bump_meta_int("pending_reward_touches") == 1
    assert store.bump_meta_int("pending_reward_touches", 4) == 5
    assert store.consume_meta_int("pending_reward_touches") == 5
    assert store.consume_meta_int("pending_reward_touches") == 0
    store.close()


def test_meta_roundtrip_json_config(tmp_path):
    store = MemoryStore(tmp_path / "m.db", 8)
    cfg = {"dim": 256, "n_units": 1024}
    store.set_meta("config", json.dumps(cfg, sort_keys=True))
    assert json.loads(store.get_meta("config")) == cfg
    assert store.get_meta("nope") is None
    store.close()
