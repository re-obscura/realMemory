"""Локальный эмбеддер на ONNX Runtime CPU через fastembed.

Опциональная зависимость: pip install 'realmemory[local]'.
Модель по умолчанию — sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
(dim=384): мультиязычная (русский+английский), ONNX; скачивается один раз в кэш
HF (~470 МБ) и работает локально, без сети и без внешних API.

Для моделей семейства e5 учитываются асимметричные префиксы
("passage: "/"query: "); для остальных текст кодируется как есть.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def default_cache_dir() -> Path:
    """Постоянный кэш моделей: ~/.cache/realmemory/fastembed.
    Дефолт fastembed — temp-каталог системы, который может быть вычищен."""
    return Path.home() / ".cache" / "realmemory" / "fastembed"


class FastEmbedProvider:
    """Реализация EmbeddingProvider на fastembed (локальный ONNX-инференс).

    Потокобезопасность: ONNX-сессия не гарантирует многопоточность — прикрываем
    замком; нагрузка одиночного агента это полностью покрывает.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        dim: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "fastembed не установлен. Установите: pip install 'realmemory[local]'"
            ) from exc
        self.model_name = model_name
        self._prefixes = "e5" in model_name.lower()
        kwargs = {
            "model_name": model_name,
            "cache_dir": str(Path(cache_dir)) if cache_dir else str(default_cache_dir()),
        }
        self._model = TextEmbedding(**kwargs)
        self._lock = threading.Lock()
        probe = next(self._model.embed(["dim probe"]))
        self.dim = int(dim) if dim is not None else int(probe.shape[0])

    @property
    def name(self) -> str:
        """Идентичность с версией библиотеки: её смена меняет векторы той же модели."""
        try:
            from importlib.metadata import version

            ver = version("fastembed")
        except ImportError:  # PackageNotFoundError; pragma: no cover - метаданных может не быть
            ver = "?"
        return f"fastembed:{self.model_name}@{ver}"

    def _vec(self, text: str, prefix: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        payload = (prefix + text.strip()) if self._prefixes else text.strip()
        with self._lock:
            vec = next(self._model.embed([payload]))
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0.0 else v

    def embed(self, text: str) -> np.ndarray:
        """Эмбеддинг записываемого факта."""
        return self._vec(text, _PASSAGE_PREFIX)

    def embed_query(self, text: str) -> np.ndarray:
        """Эмбеддинг поискового запроса (для e5 — с query-префиксом)."""
        return self._vec(text, _QUERY_PREFIX)

    @staticmethod
    def available_models() -> list[str]:
        from fastembed import TextEmbedding

        return list(TextEmbedding.list_supported_models())
