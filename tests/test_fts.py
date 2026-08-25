"""Keyword-канал FTS5: точные токены при низком косинусе + профили порогов."""
import pytest

from realmemory.store.sqlite_store import MemoryStore, build_fts_query, tokenize


def test_tokenize_and_fts_expr():
    assert tokenize("Ошибка ECONNREFUSED при деплое") == [
        "ошибка", "econnrefused", "при", "деплое",
    ]
    expr = build_fts_query("ошибка ECONNREFUSED ошибка")
    assert expr == '"ошибка" OR "econnrefused"'
    assert build_fts_query("!!! ???") is None


def test_store_fts_match_and_sync(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    if not store.fts_enabled:
        pytest.skip("SQLite без FTS5")
    import numpy as np

    from realmemory.types import MemoryRecord

    def rec(text: str) -> MemoryRecord:
        rng = np.random.default_rng(1)
        return MemoryRecord(
            id=None, text=text, kind="episodic", status="active", meta={},
            embedding=rng.standard_normal(8).astype(np.float32),
            sdr=np.array([0, 1, 2, 3], dtype=np.int32),
            created_at=1.0, updated_at=1.0, reinforced_count=0,
            last_reinforced_at=1.0, base_strength=0.5, valid_from=1.0,
        )

    a = store.insert(rec("deploy failed with ECONNREFUSED at gateway"))
    b = store.insert(rec("cooking recipe borsh with beetroot"))
    hits = store.fts_match('"econnrefused" OR "gateway"', limit=10)
    assert [r[0] for r in hits] == [a]
    store.mark_superseded(a, by_id=b, when=2.0)  # UPDATE триггер не ломает индекс
    assert [r[0] for r in store.fts_match('"econnrefused"', limit=10)] == [a]
    store.close()


def test_recall_finds_exact_token_via_keyword_channel(hippo):
    """Запрос по редкому токену находит след; при косинусе ниже cos_min_recall
    срабатывает keyword-канал (source='keyword'), а не воздержание."""
    filler = " ".join(f"zq{chr(ord('a') + i)}{i}" for i in range(12))
    fact = f"{filler} ECONNREFUSED42"
    res = hippo.remember(fact)
    packet = hippo.recall("ECONNREFUSED42", k=5)
    assert not packet.abstained
    assert packet.items[0].memory_id == res.memory_id

    # выдавливаем семантический канал: косинусный фильтр не пускает никого,
    # точное совпадение токенов обязано достать след через FTS
    hippo.config.cos_min_recall = 0.95
    kw_packet = hippo.recall("ECONNREFUSED42", k=5)
    assert not kw_packet.abstained
    top = kw_packet.items[0]
    assert top.memory_id == res.memory_id
    assert top.source == "keyword"
    assert top.confidence > 0


def test_keyword_channel_respects_scope(hippo):
    filler = " ".join(f"wq{chr(ord('a') + i)}{i}" for i in range(12))
    hippo.remember(f"{filler} RARETOKEN99", scope="alpha")
    packet = hippo.recall("RARETOKEN99", k=5, scope="beta")
    assert packet.abstained
    packet_all = hippo.recall("RARETOKEN99", k=5)
    assert not packet_all.abstained


def test_noise_abstention_unchanged(hippo):
    hippo.remember("korvex kv001 qixa")
    packet = hippo.recall("wwgrond zztilde qqzorf", k=3)
    assert packet.abstained  # нет ни семантики, ни точных токенов


def test_probe_merges_keyword_candidates(hippo):
    """Повторная запись с тем же точным токеном должна REINFORCE/LINK, а не плодить копию."""
    filler = " ".join(f"vq{chr(ord('a') + i)}{i}" for i in range(10))
    first = hippo.remember(f"{filler} TOKENXYZ7").memory_id
    again = hippo.remember(f"{filler} TOKENXYZ7")
    assert not again.created or again.decision.action.value in ("link", "reinforce")
    if not again.created:
        assert again.memory_id == first


def test_fastembed_threshold_profile_exists():
    pytest.importorskip("fastembed")
    from realmemory.encoding.embedder_local import FastEmbedProvider

    profile = FastEmbedProvider.recommended_thresholds
    assert 0.30 <= profile["theta_link"] <= 0.40
    assert 0.13 <= profile["cos_min_recall"] <= 0.20
