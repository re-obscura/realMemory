"""Провайдеры эмбеддингов. Контракт: docs/CONTRACTS.md.

Дефолтный HashingEmbedder — детерминированный feature-hashing без внешних
моделей: лексическое сходство текстов даёт высокую косинусную близость,
семантики нет. Боевой эмбеддер подключается реализацией того же протокола
(например, обёрткой над sentence-transformers или API), ядро не меняется.
"""
from __future__ import annotations

import hashlib
import re
import sys
from typing import TYPE_CHECKING, Literal, Protocol, overload

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - рантайму не нужен: fastembed импортируется лениво
    from .embedder_local import FastEmbedProvider

_WORD_RE = re.compile(r"[a-z0-9а-яёäöüß]+", re.IGNORECASE)


class EmbeddingProvider(Protocol):
    """Протокол эмбеддера: name + dim + embed(). Реализации обязаны быть
    детерминированными для одного экземпляра и потокобезопасными на чтение.
    name — стабильная идентичность (модель+версия): база отказывается
    открываться с другим эмбеддером, чтобы не смешивать несравнимые векторы."""

    dim: int

    @property
    def name(self) -> str: ...

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Feature hashing слов и символьных 3-грамм через blake2b.

    Встроенный hash() Python солёный per-process — не годится; blake2b
    стабилен кроссплатформенно и между запусками. Знак вклада берётся из
    старшего бита хэша, индекс — из младших (некоррелированные биты).
    """

    def __init__(self, dim: int = 256, seed: int = 7) -> None:
        if dim <= 0:
            raise ValueError("dim должен быть положительным")
        self.dim = int(dim)
        self.seed = int(seed)
        # blake2b key допускает до 64 байт
        self._key = bytes([(seed * 37 + 11) & 0xFF])

    @property
    def name(self) -> str:
        return f"hashing(dim={self.dim},seed={self.seed})"

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=self._key).digest()
        h = int.from_bytes(digest, "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        tokens: list[str] = []
        for w in words:
            tokens.append("w:" + w)
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tokens.append("g:" + w[i : i + 3])
        return tokens

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text or not text.strip():
            return vec
        for tok in self._tokens(text):
            idx, sign = self._bucket(tok)
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec


@overload
def make_embedder(choice: Literal["local"], *,
                  dim: int | None = None) -> FastEmbedProvider: ...
@overload
def make_embedder(choice: Literal["hashing"], *,
                  dim: int | None = None) -> HashingEmbedder: ...
@overload
def make_embedder(choice: str, *, dim: int | None = None) -> EmbeddingProvider: ...
def make_embedder(choice: str, *, dim: int | None = None) -> EmbeddingProvider:
    """Единая фабрика эмбеддеров: local (fastembed ONNX), hashing
    (детерминированный тестовый) или auto (local при наличии fastembed,
    иначе hashing — диагностика только в stderr: stdout stdio-сервера занят
    JSON-RPC). dim фиксирует размерность (hashing; у local — вместо пробы
    модели), None — дефолты провайдеров.

    Живёт в encoding, а не в api.mcp_server: командный recall
    (team.recall_team) пересобирает эмбеддер базы по тем же правилам."""
    if choice == "hashing":
        return HashingEmbedder(dim=256 if dim is None else dim)
    if choice == "local":
        from .embedder_local import FastEmbedProvider

        return FastEmbedProvider(dim=dim)
    if choice == "auto":
        try:
            from .embedder_local import FastEmbedProvider

            return FastEmbedProvider(dim=dim)
        except ImportError:
            print(
                "[realmemory] fastembed не установлен — используется HashingEmbedder. "
                "Установите: pip install 'realmemory[local]'",
                file=sys.stderr,
            )
            return HashingEmbedder(dim=256 if dim is None else dim)
    raise ValueError(f"неизвестный эмбеддер: {choice}")
