"""Сквозные сценарии фасада Hippocampus.

Проверяются продуктовые обещания фазы 0:
  - точный и paraphrase-recall на уровне косинус-бейзлайна;
  - гейт новизны (CREATE/REINFORCE/LINK);
  - суперсед фактов и воздержание вместо галлюцинации;
  - подкрепление через feedback, забывание по времени;
  - ассоциативный multi-hop recall через link_memories;
  - консолидация (коммит связей, повышение до семантических);
  - save/reopen с идентичным поведением.
"""
import hashlib

import numpy as np
import pytest

from realmemory import Hippocampus
from realmemory.types import DecisionAction

TOPIC_POOLS = [
    ["korvex", "deltun", "miphar", "soltak"],
    ["wreniq", "fablex", "guvnot", "charod"],
    ["plimso", "vendrik", "thaxol", "brumel"],
    ["yandor", "oximet", "klavur", "zenpak"],
]
NOISE = "wwgrond zztilde qqzorf"


def _obj_token(i: int) -> str:
    """Уникальный объектный токен факта из хэш-букв: соседние индексы
    не делят триграмм, novelty-гейт не сливает разные факты."""
    digest = hashlib.blake2b(f"fact-{i}".encode("ascii"), digest_size=9,
                             person=b"rm-test").hexdigest()
    return "".join(chr(ord("a") + int(c, 16)) for c in digest)


def _corpus(n=30, seed=7):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        pool = TOPIC_POOLS[i % len(TOPIC_POOLS)]
        w1 = pool[int(rng.integers(len(pool)))]
        attr = ["qixa", "romed", "suvat"][int(rng.integers(3))]
        items.append(f"{w1} {_obj_token(i)} {attr}")
    return items


def _subset(text: str, fi: int, rng) -> str:
    """Запрос из двух слов факта: объектный токен + одно из остальных."""
    parts = text.split()
    obj = _obj_token(fi)
    rest = [p for p in parts if p != obj]
    keep = rest[int(rng.integers(len(rest)))]
    return f"{keep} {obj}" if rng.random() < 0.5 else f"{obj} {keep}"


def test_recall_matches_exact_cosine_baseline(hippo):
    corpus = _corpus()
    ids = [hippo.remember(t).memory_id for t in corpus]
    emb = np.stack([hippo.embedder.embed(t) for t in corpus])
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb /= norms
    rng = np.random.default_rng(11)
    pipe_hits, base_hits = 0, 0
    trials = 30
    for _ in range(trials):
        fi = int(rng.integers(len(corpus)))
        q = _subset(corpus[fi], fi, rng)
        packet = hippo.recall(q, k=3)
        pipe_hits += int(any(it.memory_id == ids[fi] for it in packet.items))
        qe = hippo.embedder.embed(q)
        sims = emb @ qe
        top = {int(t) for t in np.argpartition(-sims, 2)[:3]}
        base_hits += int(fi in top)
    assert pipe_hits >= 0.95 * trials
    assert pipe_hits >= base_hits - 2  # конвейер не проигрывает точному поиску


def test_gate_decisions_end_to_end(hippo):
    first = hippo.remember("korvex kv001 qixa")
    assert first.decision.action is DecisionAction.CREATE and first.created
    again = hippo.remember("korvex kv001 qixa")
    assert again.decision.action is DecisionAction.REINFORCE
    assert not again.created and again.memory_id == first.memory_id
    shuffled = "qixa korvex kv001"
    para = hippo.remember(shuffled)
    assert para.decision.action in (DecisionAction.REINFORCE, DecisionAction.LINK)
    other = hippo.remember("plimso pt002 suvat")
    assert other.decision.action is DecisionAction.CREATE


def test_update_fact_supersedes_old(hippo):
    old = hippo.remember("korvex kv101 qixa")
    new = hippo.update_fact(old.memory_id, "korvex kv101 zufet")
    assert new.created
    rec = hippo.store.get(old.memory_id)
    assert rec.status == "superseded" and rec.superseded_by == new.memory_id
    packet = hippo.recall("korvex kv101", k=5)
    returned_ids = [it.memory_id for it in packet.items]
    assert new.memory_id in returned_ids
    assert old.memory_id not in returned_ids
    with_history = hippo.recall("korvex kv101", k=10, include_superseded=True)
    assert old.memory_id in [it.memory_id for it in with_history.items]


def test_abstention_on_unknown_topic(hippo):
    hippo.remember("korvex kv201 qixa")
    hippo.remember("plimso pt202 suvat")
    packet = hippo.recall(NOISE, k=5)
    assert packet.abstained and len(packet.items) == 0


def test_feedback_boosts_retention(hippo, clock):
    a = hippo.remember("korvex kv301 qixa").memory_id
    b = hippo.remember("plimso pt302 suvat").memory_id
    assert hippo.feedback([a], 1.0) == 1
    clock.advance(50_000.0)  # половина tau_episodic
    pa = hippo.store.get(a)
    pb = hippo.store.get(b)
    assert pa.base_strength > pb.base_strength
    qa = hippo.recall("korvex kv301", k=1).items[0]
    qb = hippo.recall("plimso pt302", k=1).items[0]
    assert qa.retention > qb.retention


def test_negative_feedback_weakens_trace(hippo, clock):
    a = hippo.remember("korvex kv311 qixa").memory_id
    b = hippo.remember("plimso pt312 suvat").memory_id
    clock.advance(10_000.0)
    assert hippo.feedback([a], -0.5) == 1
    pa, pb = hippo.store.get(a), hippo.store.get(b)
    assert pa.base_strength == pytest.approx(pb.base_strength * 0.5)
    # таймер подкрепления и счётчик продвижения не тронуты
    assert pa.last_reinforced_at == pb.last_reinforced_at
    assert pa.reinforced_count == pb.reinforced_count == 0
    qa = hippo.recall("korvex kv311", k=1).items[0]
    qb = hippo.recall("plimso pt312", k=1).items[0]
    assert qa.retention < qb.retention


def test_zero_feedback_is_neutral(hippo):
    res = hippo.remember("korvex kv313 qixa")
    before = hippo.store.get(res.memory_id)
    assert hippo.feedback([res.memory_id], 0.0) == 0
    after = hippo.store.get(res.memory_id)
    assert after.base_strength == before.base_strength
    assert after.reinforced_count == before.reinforced_count == 0


def test_repeated_negative_feedback_never_promotes(hippo, clock):
    res = hippo.remember("korvex kv315 qixa")
    mid = res.memory_id
    for _ in range(6):
        hippo.feedback([mid], -0.5)
    rec = hippo.store.get(mid)
    assert rec.base_strength < 0.01
    assert rec.kind == "episodic"
    clock.advance(2500.0)  # за порогом promote_min_age_s
    report = hippo.consolidate(save=False)
    assert report.promoted_to_semantic == 0
    assert hippo.recall("korvex kv315", k=3).abstained


def test_recall_encodes_queries_via_embed_query(tmp_path, tiny_cfg, clock):
    """Асимметрия эмбеддера: факты — embed(), поисковые запросы — embed_query()."""
    from realmemory.encoding.embedder import HashingEmbedder

    class SpyEmbedder(HashingEmbedder):
        def __init__(self):
            super().__init__(dim=tiny_cfg.dim)
            self.embed_calls = 0
            self.query_calls = 0

        def embed(self, text):
            self.embed_calls += 1
            return super().embed(text)

        def embed_query(self, text):
            self.query_calls += 1
            return super().embed(text)

    spy = SpyEmbedder()
    h = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock, embedder=spy)
    try:
        h.remember("korvex kv121 qixa")
        assert (spy.embed_calls, spy.query_calls) == (1, 0)
        h.recall("korvex kv121", k=1)
        assert spy.query_calls == 1, "запрос должен кодироваться через embed_query()"
        assert spy.embed_calls == 1
    finally:
        h.close()


def test_forgetting_after_long_idle(hippo, clock):
    res = hippo.remember("korvex kv401 qixa")
    assert not hippo.recall("korvex kv401", k=3).abstained
    clock.advance(300_000.0)  # 3x tau_episodic
    assert hippo.recall("korvex kv401", k=3).abstained
    assert hippo.store.get(res.memory_id) is not None  # архивируется, не удаляется


def test_consolidation_promotes_to_semantic(hippo, clock):
    res = hippo.remember("korvex kv501 qixa")
    for _ in range(5):
        hippo.remember("korvex kv501 qixa")
    clock.advance(2500.0)  # > promote_min_age_s
    report = hippo.consolidate(save=False)
    assert report.promoted_to_semantic >= 1
    assert hippo.store.get(res.memory_id).kind == "semantic"


def test_associative_link_surfaces_partner(hippo):
    a = hippo.remember("korvex kv601 qixa").memory_id
    b = hippo.remember("plimso pt602 suvat").memory_id
    assert hippo.link_memories([a, b], strength=2.0) > 0
    # связи живут в eligibility-логе до первого «сна»; консолидируем и проверяем
    hippo.consolidate(save=False)
    packet = hippo.recall("plimso pt602", k=5)
    texts = {it.memory_id: it for it in packet.items}
    assert b in texts
    assert texts[b].source == "direct"
    assert any(it.memory_id == a for it in packet.items), (
        "связанный след должен всплывать через волну ассоциаций"
    )


def test_related_ids_create_edges_after_consolidation(hippo):
    c = hippo.remember("korvex kv701 qixa", force_new=True).memory_id
    d = hippo.remember("wreniq wf702 suvat", force_new=True, related_ids=(c,)).memory_id
    assert hippo.eligibility.pending_count >= 1
    hippo.consolidate(save=False)
    assert hippo.network.edge_count > 0
    assert c != d


def test_save_reload_identical_behaviour(tmp_path, tiny_cfg, clock):
    dirp = tmp_path / "brain"
    h1 = Hippocampus.open(dirp, config=tiny_cfg, clock=clock)
    a = h1.remember("korvex kv801 qixa").memory_id
    b = h1.remember("plimso pt802 suvat").memory_id
    h1.link_memories([a, b], strength=1.5)
    h1.consolidate()  # коммит связей + снапшот
    edges_before = h1.stats()["edges"]
    q = "plimso pt802"
    before = h1.recall(q, k=5)
    h1.close()

    h2 = Hippocampus.open(dirp, config=tiny_cfg, clock=clock)
    try:
        after = h2.recall(q, k=5)
        assert h2.stats()["edges"] == edges_before
        assert [it.memory_id for it in after.items] == [it.memory_id for it in before.items]
        assert np.allclose(
            [it.confidence for it in after.items],
            [it.confidence for it in before.items],
            atol=1e-6,
        )
        assert any(it.memory_id == a for it in after.items)
    finally:
        h2.close()


def test_input_validation(hippo):
    with pytest.raises(ValueError):
        hippo.remember("   ")
    with pytest.raises(ValueError):
        hippo.remember("korvex kv902 qixa", kind="bogus")
    with pytest.raises(ValueError):
        hippo.recall("")
    with pytest.raises(KeyError):
        hippo.remember("korvex kv901 qixa", related_ids=(9999,))
    with pytest.raises(ValueError):
        hippo.feedback([], reward=2.0)


def test_stats_shape(hippo):
    hippo.remember("korvex kv111 qixa")
    hippo.remember("korvex kv111 qixa")
    hippo.recall("korvex kv111")
    s = hippo.stats()
    assert s["memories_active"] == 1
    assert s["decisions"]["create"] == 1 and s["decisions"]["reinforce"] == 1
    assert s["recalls"] == 1 and s["writes"] == 2
    assert s["avg_recall_ms"] >= 0
