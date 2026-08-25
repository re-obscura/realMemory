"""Изоляция проектов (namespace) и идентичность эмбеддера базы."""
import sqlite3

import pytest

from realmemory import Hippocampus
from realmemory.encoding.embedder import HashingEmbedder


def _hasher(tiny_cfg):
    return HashingEmbedder(dim=tiny_cfg.dim)


def test_namespaces_are_isolated(tmp_path, tiny_cfg, clock):
    a = Hippocampus.open(tmp_path / "root", config=tiny_cfg, clock=clock, namespace="projA")
    b = Hippocampus.open(tmp_path / "root", config=tiny_cfg, clock=clock, namespace="projB")
    try:
        a.remember("korvex kvA1 qixa")
        b.remember("plimso ptB2 suvat")
        assert a.store.count() == 1 and b.store.count() == 1
        assert not a.recall("korvex kvA1", k=3).abstained
        assert b.recall("korvex kvA1", k=3).abstained, "чужой namespace не виден"
        assert (tmp_path / "root" / "projA" / "memory.db").exists()
        assert (tmp_path / "root" / "projB" / "memory.db").exists()
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("bad", ["../evil", "", ".hidden", "a/b", "x" * 65])
def test_invalid_namespace_rejected(tmp_path, tiny_cfg, clock, bad):
    with pytest.raises(ValueError):
        Hippocampus.open(tmp_path, config=tiny_cfg, clock=clock, namespace=bad)


def test_embedder_identity_enforced(tmp_path, tiny_cfg, clock):
    p = tmp_path / "brain"
    h = Hippocampus.open(p, config=tiny_cfg, clock=clock, embedder=_hasher(tiny_cfg))
    h.remember("korvex kvC1 qixa")
    h.close()

    h2 = Hippocampus.open(p, config=tiny_cfg, clock=clock, embedder=_hasher(tiny_cfg))
    try:
        assert h2.store.count() == 1
    finally:
        h2.close()

    # другой seed -> другие векторы при том же dim; смешивать нельзя
    with pytest.raises(RuntimeError, match="эмбеддер"):
        Hippocampus.open(p, config=tiny_cfg, clock=clock,
                         embedder=HashingEmbedder(dim=tiny_cfg.dim, seed=8))


def test_legacy_unlabeled_db_adopts_current_embedder(tmp_path, tiny_cfg, clock):
    """База до введения маркировки открывается без ошибки и получает метку."""
    p = tmp_path / "legacy"
    h = Hippocampus.open(p, config=tiny_cfg, clock=clock, embedder=_hasher(tiny_cfg))
    h.remember("korvex kvD1 qixa")
    h.close()

    con = sqlite3.connect(str(p / "memory.db"))
    con.execute("DELETE FROM db_meta")
    con.commit()
    con.close()

    h2 = Hippocampus.open(p, config=tiny_cfg, clock=clock, embedder=_hasher(tiny_cfg))
    try:
        assert h2.store.get_meta("embedder") is not None
        assert h2.store.count() == 1
    finally:
        h2.close()


def test_store_meta_roundtrip(hippo):
    assert hippo.store.get_meta("missing") is None
    hippo.store.set_meta("k", "v1")
    hippo.store.set_meta("k", "v2")  # upsert
    assert hippo.store.get_meta("k") == "v2"


def test_verify_embedder_off_for_vectorless_tools(tmp_path, tiny_cfg, clock):
    """Хуки сна/брифа открывают базу с подставным эмбеддером; проверка должна
    отключаться и не перезаписывать записанную идентичность."""
    p = tmp_path / "brain"
    h = Hippocampus.open(p, config=tiny_cfg, clock=clock, embedder=_hasher(tiny_cfg))
    h.remember("korvex kvE1 qixa")
    h.close()

    other = HashingEmbedder(dim=tiny_cfg.dim, seed=8)
    h2 = Hippocampus.open(p, config=tiny_cfg, clock=clock, embedder=other,
                          verify_embedder=False)
    try:
        assert h2.store.count() == 1
        assert h2.store.get_meta("embedder") == _hasher(tiny_cfg).name
    finally:
        h2.close()
