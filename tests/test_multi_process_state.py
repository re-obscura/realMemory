"""Мультипроцессная целостность состояния и миграция наследия v0.3.

Ключевой регресс WP1: долгоживущий MCP-сервер и краткоживущие хуки работают
с одной базой одновременно; никакое состояние больше не теряется по принципу
«последний писатель снапшота выиграл».
"""
import json
import sqlite3

import numpy as np
import pytest

from realmemory import Hippocampus


@pytest.fixture
def pair(tmp_path, tiny_cfg, clock):
    """Два независимых экземпляра над одной директорией памяти."""
    a = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    b = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    try:
        yield a, b
    finally:
        a.close()
        b.close()


def test_eligibility_from_a_reaches_b_consolidation(pair):
    a, b = pair
    ia = a.remember("korvex kv001 qixa").memory_id
    # bind пишет процесс A (write-through в общую таблицу)...
    a.remember("wreniq wf002 suvat", related_ids=(ia,))
    assert a.pending_eligibility >= 1
    # ...а консолидирует чужой процесс B
    b.consolidate()
    edges_b = b.stats()["edges"]
    assert edges_b > 0
    # A видит и чужую консолидацию, и исчезнувший pending (состояние общее)
    assert a.stats()["edges"] == edges_b
    assert a.pending_eligibility == 0


def test_decision_counters_survive_process_restart(pair):
    a, b = pair
    a.remember("korvex kv003 qixa")
    a.remember("korvex kv003 qixa")  # REINFORCE
    s = b.stats()  # B даже не писал: счётчики из журнала событий, а не из памяти процесса
    assert s["decisions"]["create"] == 1
    assert s["decisions"]["reinforce"] == 1
    assert s["writes"] == 2


def test_reward_from_a_strengthens_edges_committed_by_b(pair):
    a, b = pair
    i = a.remember("korvex kv004 qixa").memory_id
    j = a.remember("plimso pt005 suvat", related_ids=(i,)).memory_id
    a.feedback([j], 1.0)  # усиливает незакоммиченные bind'ы следа j
    b.consolidate()
    _, ws = b.store.edges_load()
    # базовая сила bind по related_ids = 0.5; reward ×2 должен её удвоить
    assert float(ws.max()) == pytest.approx(1.0, rel=1e-3)


def test_concurrent_consolidations_do_not_double_decay(tmp_path, tiny_cfg, clock):
    a = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    b = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    try:
        i = a.remember("korvex kv006 qixa").memory_id
        j = a.remember("plimso pt007 suvat", related_ids=(i,)).memory_id
        a.link_memories([i, j], strength=4.0)  # одно событие, сильное
        clock.advance(1000.0)
        a.consolidate()
        w_after_a = dict(
            zip(*[arr.tolist() for arr in a.store.edges_load()])
        )
        clock.advance(1000.0)
        b.consolidate()  # пустой сон B: только тик часов, без повторного коммита
        w_after_b = dict(
            zip(*[arr.tolist() for arr in b.store.edges_load()])
        )
        assert set(w_after_a) == set(w_after_b)
        factor = np.exp(-1000.0 / tiny_cfg.tau_edge_stable)
        for key, w_before in w_after_a.items():
            assert w_after_b[key] == pytest.approx(w_before * factor, rel=1e-3), (
                "вес обязан распасться ровно один раз за Δt между снами"
            )
    finally:
        a.close()
        b.close()


def test_legacy_snapshot_and_journal_imported_once(tmp_path, tiny_cfg, clock):
    dirp = tmp_path / "brain"
    h1 = Hippocampus.open(dirp, config=tiny_cfg, clock=clock)
    try:
        x = h1.remember("korvex kv801 qixa").memory_id
        y = h1.remember("plimso pt802 suvat").memory_id
        h1.link_memories([x, y], strength=1.5)
        h1.consolidate()
        keys, ws = h1.store.edges_load()
    finally:
        h1.close()

    # откатываем базу к «наследию v0.3»: таблицы пусты, состояние в файлах
    con = sqlite3.connect(str(dirp / "memory.db"))
    with con:
        con.execute("DELETE FROM edges")
        con.execute("DELETE FROM elig_sources")
        con.execute("DELETE FROM eligibility")
        con.execute("DELETE FROM events")
        con.execute("DELETE FROM db_meta WHERE key IN "
                    "('journal_imported','snapshot_imported','last_edge_tick','edges_rev')")
    con.close()
    snapshot_payload = {
        "version": 1,
        "config": tiny_cfg.snapshot_fields(),
        "network": {
            "keys": keys,
            "weights": ws,
            "last_tick": float(clock.now()),
        },
        "eligibility": {
            "tau": tiny_cfg.tau_eligibility,
            "events": [
                ([1, 2, 3], [10, 11, 12], 0.8, float(clock.now()), [x, y]),
            ],
        },
        "stats": {},
        "decisions": {},
        "pending_rewards": 0,
    }
    import pickle

    with (dirp / "snapshot.pkl").open("wb") as f:
        pickle.dump(snapshot_payload, f)
    (dirp / "journal.jsonl").write_text(
        "\n".join([
            json.dumps({"type": "write", "id": int(x), "kind": "episodic",
                        "chars": 18, "t": 100.0}),
            json.dumps({"type": "recall", "k": 5, "items": 1, "abstained": False,
                        "latency_ms": 3.2, "top_conf": 0.42, "t": 101.0}),
        ]) + "\n",
        encoding="utf-8",
    )

    h2 = Hippocampus.open(dirp, config=tiny_cfg, clock=clock)
    try:
        # рёбра восстановлены из снапшота
        count, total_w = h2.store.edges_stats(float(clock.now()), tiny_cfg.tau_edge_stable)
        assert count == keys.size and total_w == pytest.approx(float(ws.sum()), rel=1e-3)
        # eligibility-событие из снапшота ждёт своего сна
        assert h2.pending_eligibility == 1
        # журнал импортирован, legacy write помечен как create
        decisions = h2.stats()["decisions"]
        assert decisions["create"] >= 1
        recalls = [e for e in h2.store.iter_events() if e.get("type") == "recall"]
        assert len(recalls) == 1 and recalls[0]["latency_ms"] == 3.2
    finally:
        h2.close()

    assert not (dirp / "snapshot.pkl").exists()
    assert (dirp / "snapshot.pkl.imported").exists()
    assert not (dirp / "journal.jsonl").exists()
    assert (dirp / "journal.jsonl.imported").exists()

    # повторное открытие ничего не импортирует второй раз
    h3 = Hippocampus.open(dirp, config=tiny_cfg, clock=clock)
    try:
        assert h3.pending_eligibility == 1  # то же единственное событие
        assert h3.stats()["journal_events"] < 10
    finally:
        h3.close()


def test_db_config_geometry_guard(tmp_path, tiny_cfg, clock):
    from realmemory.config import MemoryConfig

    dirp = tmp_path / "brain"
    h1 = Hippocampus.open(dirp, config=tiny_cfg, clock=clock)
    h1.close()
    other = MemoryConfig(dim=tiny_cfg.dim, n_units=tiny_cfg.n_units * 2,
                         k_sparse=tiny_cfg.k_sparse, sdr_seed=tiny_cfg.sdr_seed)
    with pytest.raises(RuntimeError, match="n_units"):
        Hippocampus.open(dirp, config=other, clock=clock)
