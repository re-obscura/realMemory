"""Локальный эмбеддер (fastembed ONNX). Требует установленный пакет и модель;
при их отсутствии тесты корректно пропускаются."""
import numpy as np
import pytest

pytest.importorskip("fastembed")

from realmemory.encoding.embedder_local import DEFAULT_MODEL, FastEmbedProvider


@pytest.fixture(scope="module")
def provider():
    try:
        return FastEmbedProvider()
    except Exception as exc:  # noqa: BLE001 - сеть/кэш модели могут быть недоступны
        pytest.skip(f"модель эмбеддера недоступна: {exc}")


def test_dim_matches_model(provider):
    assert provider.dim == 384  # paraphrase-multilingual-MiniLM-L12-v2


def test_deterministic_and_normalized(provider):
    a = provider.embed("Проект использует PostgreSQL для хранения данных")
    b = provider.embed("Проект использует PostgreSQL для хранения данных")
    assert np.allclose(a, b, atol=1e-6)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-4


def test_query_and_passage_shapes(provider):
    q = provider.embed_query("какая база данных у проекта?")
    p = provider.embed("факт о проекте")
    assert q.shape == (provider.dim,) and p.shape == (provider.dim,)


def test_blank_is_zero_vector(provider):
    v = provider.embed("   ")
    assert v.shape == (provider.dim,) and not v.any()


def test_related_beats_unrelated(provider):
    def cos(a_text, b_text):
        a, b_ = provider.embed(a_text), provider.embed(b_text)
        return float(a @ b_)

    related = cos(
        "Пользователь предпочитает PostgreSQL для хранения данных",
        "База данных проекта — Postgres, миграции через alembic",
    )
    unrelated = cos(
        "Пользователь предпочитает PostgreSQL для хранения данных",
        "Кот спит на подоконнике и ловит мышей",
    )
    assert related > unrelated + 0.05
    assert "multilingual" in DEFAULT_MODEL
