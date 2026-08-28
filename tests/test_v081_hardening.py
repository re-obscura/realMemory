"""Упрочнение v0.8.1: протокол отзыва при revise/GC, fail-closed привязка
демонов, валидация peer-адресов, инкрементальные счётчики наблюдаемости,
keyword-кандидаты гейта записи в точном режиме."""
import sqlite3
import threading

import numpy as np
import pytest

from realmemory import Hippocampus, MemoryConfig
from realmemory.team import registry
from realmemory.team.policy import TeamPolicy, require_bind_token, set_project_shareable
from realmemory.team.recall_team import _safe_peer_address
from realmemory.team.transport import CoordinatorClient, CoordinatorError
from realmemory.timeprov import FakeClock


def _cfg(**over) -> MemoryConfig:
    fields = {
        "dim": 256, "n_units": 512, "k_sparse": 48, "sdr_seed": 5,
        "bucket_cap": 32,
        "tau_episodic": 60 * 86400.0,
        "tau_semantic": 600 * 86400.0,
        "gc_grace_below_floor_s": 5 * 86400.0,
    }
    fields.update(over)
    cfg = MemoryConfig(**fields)
    cfg.validate()
    return cfg


def _policy() -> TeamPolicy:
    policy = TeamPolicy()
    policy.identity = "me"
    set_project_shareable(policy, "proj", True)
    return policy


# -- протокол отзыва: revise и GC обязаны ставить tombstones -------------------------

def test_revise_retracts_published_trace(tmp_path):
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock(),
                         author="me")
    try:
        mid = h.remember("решение о кэше трансляций rk401", scope="proj").memory_id
        created = registry.publish(h.store, [mid], policy=_policy(), now=1.0)
        assert len(created) == 1

        h.clock.advance(10.0)
        h.update_fact(mid, "кэш трансляций переехал в редис rk402")

        assert registry.active_publications(h.store) == []
        tombs = registry.tombstoned_publications(h.store)
        assert len(tombs) == 1 and tombs[0].trace_id == mid
        # tombstone ещё не доставлен — ждёт следующего sync
        awaiting = h.store.publications_unsynced()
        assert [r[0] for r in awaiting] == [tombs[0].publication_id]
    finally:
        h.close()


def test_gc_forget_retracts_published_trace(tmp_path):
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock(),
                         author="me")
    try:
        mid = h.remember("забываемое решение ортун oz501", scope="proj").memory_id
        registry.publish(h.store, [mid], policy=_policy(), now=1.0)
        h.store.publication_mark_synced(
            [p.publication_id for p in registry.active_publications(h.store)], 2.0)
        assert registry.sync_status(h.store)["awaiting_sync"] == 0

        h.clock.advance(200 * 86400.0)  # retention ниже пола, grace прошёл
        report = h.consolidate()
        assert report.forgotten_traces == 1
        assert report.publications_retracted == 1
        assert registry.active_publications(h.store) == []
        assert len(registry.tombstoned_publications(h.store)) == 1
    finally:
        h.close()


# -- fail-closed привязка сетевых демонов ---------------------------------------------

def test_require_bind_token_refuses_nonloopback_without_token():
    with pytest.raises(SystemExit, match="токена"):
        require_bind_token("0.0.0.0", None, what="peer-endpoint")
    with pytest.raises(SystemExit):
        require_bind_token("192.168.1.10", "", what="coordinator")
    # loopback без токена и внешний адрес с токеном разрешены
    require_bind_token("127.0.0.1", None, what="x")
    require_bind_token("localhost", "", what="x")
    require_bind_token("0.0.0.0", "sekret", what="x")


# -- валидация адресов peer'ов (адрес приходит из чужих heartbeat'ов) -----------------

def test_safe_peer_address_rejects_injection():
    assert _safe_peer_address("192.168.1.5:8410") == "192.168.1.5:8410"
    assert _safe_peer_address("andrey-pc.local:8410") == "andrey-pc.local:8410"
    assert _safe_peer_address(" 127.0.0.1:8410 ") == "127.0.0.1:8410"
    # URL-инъекция и мусор не проходят
    assert _safe_peer_address("evil.example.com/steal#x:80") is None
    assert _safe_peer_address("user@host:80") is None
    assert _safe_peer_address("127.0.0.1:0") is None
    assert _safe_peer_address("127.0.0.1:99999") is None
    assert _safe_peer_address("") is None


# -- координатор: без имени эмбеддера поиск бессмыслен --------------------------------

def test_coordinator_requires_embedder_name(tmp_path):
    srv = make_coordinator(tmp_path / "coord")
    try:
        port = srv.server_address[1]
        client = CoordinatorClient(f"http://127.0.0.1:{port}", timeout_s=3.0)
        with pytest.raises(CoordinatorError, match="400"):
            client.search(np.zeros(4, dtype=np.float32), k=3, embedder="")
    finally:
        srv.shutdown()
        srv.server_close()


def make_coordinator(data_dir):
    from realmemory.team.coordinator import make_server

    srv = make_server(data_dir, host="127.0.0.1", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# -- инкрементальные счётчики наблюдаемости -------------------------------------------

def test_obs_counters_match_journal_and_survive_rotation(tmp_path):
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock())
    try:
        h.remember("счётчики наблюдаемости джен ga701")
        h.recall("счётчики наблюдаемости", k=1)
        h.consolidate()

        def journal_truth():
            decisions = h.store.gate_decisions()
            recalls, avg = h.store.recall_stats()
            return sum(decisions.values()), decisions, recalls, avg

        writes, decisions, recalls, _avg = journal_truth()
        stats = h.stats()
        assert stats["writes"] == writes
        assert stats["decisions"]["create"] == decisions.get("create", 0)
        assert stats["recalls"] == recalls

        # посев: стереть счётчики — stats() обязан восстановить их сканом журнала
        con = sqlite3.connect(str(tmp_path / "rm" / "memory.db"), timeout=5)
        try:
            con.execute("DELETE FROM db_meta WHERE key LIKE 'obs_%'")
            con.commit()
        finally:
            con.close()
        stats_reseeded = h.stats()
        assert stats_reseeded["writes"] == writes
        assert stats_reseeded["recalls"] == recalls
        assert stats_reseeded["decisions"]["create"] == stats["decisions"]["create"]

        # ротация журнала счётчики не теряет
        h.store.events_prune(2)
        stats_after_rotation = h.stats()
        assert stats_after_rotation["writes"] == writes
        assert stats_after_rotation["recalls"] == recalls
    finally:
        h.close()


# -- гейт записи в точном режиме видит keyword-кандидатов ------------------------------

def test_probe_exact_mode_sees_fts_candidates_beyond_budget(tmp_path):
    """recall_oversample=1 → бюджет гейта 3: четвёртый близкий след выше
    theta_link в косинусный префикс не попадает и был гейту невидим, пока
    точная ветка не учитывает FTS-кандидатов."""
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(recall_oversample=1),
                         clock=FakeClock())
    try:
        if not h.store.fts_enabled:
            pytest.skip("SQLite без FTS5")
        texts = [
            "альфа ТКН42 бета qw81",
            "альфа ТКН42 гамма qw81",
            "альфа ТКН42 дельта qw81",
            "альфа ТКН42 эпсилон qw81",
        ]
        ids = [h.remember(t, force_new=True).memory_id for t in texts]
        emb, sdr = h._encode(texts[0])
        cosines = {i: h._cosine(emb, h.store.get(i).embedding) for i in ids}
        assert all(c >= h.config.theta_link for c in cosines.values()), cosines
        lowest = min(ids, key=lambda i: cosines[i])

        best_id, _best_cos, near = h._probe(emb, sdr, scope=None, text=texts[0])
        assert best_id == ids[0]  # косинус 1.0 — лучший в любом случае
        assert lowest not in (best_id,)   # он глубже бюджетного префикса
        assert lowest in near             # но виден через keyword-канал
    finally:
        h.close()
