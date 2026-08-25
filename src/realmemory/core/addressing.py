"""L1 — адресация голосованием по инвертированному индексу SDR-юнитов.

След пишет указатель (id в SQLite) в бакеты всех своих on-битов; запрос
суммирует попадания по своим on-битам. Коррелированные паттерны делят часть
юнитов -> целевой указатель набирает ~rho*k голосов против k^2/N у случайных
(robust-LSH поведение). Нейроморфная интерпретация: юниты — нейроны, запись —
аксональные указатели, голоса — суммация пресинаптических совпадений,
top-k — WTA.

Историческая заметка: первая реализация использовала классический
Kanerva-SDM на плотных биполярных адресах с Хэмминг-радиусом. Бенчмарк
показал её несостоятельность для семантически близких (но не почти
идентичных) запросов: из-за концентрации Хэмминговой метрики и
антикорреляции расстояний шары доступа двух паттернов с cos~0.6 почти не
пересекаются (hits@10 = 0.495 против 1.0 у точного косинуса). Инвертированный
индекс по SDR лишен этого дефекта; детали — docs/ARCHITECTURE.md §3.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QueryResult:
    """Результат запроса к L1.

    candidates — указатели по убыванию голосов; votes[i] — голоса candidates[i];
    active_locations — сколько уникальных юнитов запроса имели непустые бакеты
    (нормализатор голосов).
    """

    candidates: np.ndarray  # int64
    votes: np.ndarray  # int32
    active_locations: int


class SDRVotingIndex:
    def __init__(self, n_units: int, bucket_cap: int = 64) -> None:
        if n_units <= 0 or bucket_cap <= 0:
            raise ValueError("n_units и bucket_cap должны быть положительными")
        self.n_units = int(n_units)
        self.bucket_cap = int(bucket_cap)
        self._buckets: list[deque[int]] = [deque() for _ in range(self.n_units)]

    # -- запись / чтение -------------------------------------------------------

    def write(self, sdr: np.ndarray, pointer: int) -> int:
        """Записать указатель в бакеты on-битов. Возвращает #затронутых бакетов."""
        pointer = int(pointer)
        if pointer < 0:
            raise ValueError("pointer должен быть >= 0")
        touched = 0
        for u in np.unique(np.asarray(sdr, dtype=np.int64)).tolist():
            if not 0 <= u < self.n_units:
                raise ValueError(f"юнит {u} вне диапазона [0, {self.n_units})")
            bucket = self._buckets[u]
            if pointer in bucket:  # повторная запись не дублируется
                continue
            bucket.append(pointer)
            while len(bucket) > self.bucket_cap:
                bucket.popleft()  # вытесняется старейший (palimpsest под давлением)
            touched += 1
        return touched

    def query(self, sdr: np.ndarray, max_candidates: int) -> QueryResult:
        """Кандидаты по убыванию голосов юнитов запроса."""
        if max_candidates < 1:
            raise ValueError("max_candidates должен быть >= 1")
        units = np.unique(np.asarray(sdr, dtype=np.int64))
        empty_c = np.empty(0, dtype=np.int64)
        empty_v = np.empty(0, dtype=np.int32)
        if units.size == 0:
            return QueryResult(empty_c, empty_v, 0)
        ptrs: list[int] = []
        active = 0
        for u in units.tolist():
            if not 0 <= u < self.n_units:
                raise ValueError(f"юнит {u} вне диапазона [0, {self.n_units})")
            bucket = self._buckets[u]
            if bucket:
                active += 1
                ptrs.extend(bucket)
        if not ptrs:
            return QueryResult(empty_c, empty_v, active)
        uniq, counts = np.unique(np.asarray(ptrs, dtype=np.int64), return_counts=True)
        order = np.argsort(-counts, kind="stable")
        sel = order[: min(int(max_candidates), uniq.size)]
        return QueryResult(uniq[sel], counts[sel].astype(np.int32), active)

    # -- служебное ---------------------------------------------------------------

    def load_factor(self) -> float:
        """Средняя длина бакета — мера суперпозиции/давления ёмкости."""
        filled = [len(b) for b in self._buckets if b]
        return float(np.mean(filled)) if filled else 0.0

    def state_dict(self) -> dict:
        return {"buckets": {i: list(b) for i, b in enumerate(self._buckets) if b}}

    def load_state_dict(self, state: dict) -> None:
        for b in self._buckets:
            b.clear()
        for i, items in state.get("buckets", {}).items():
            u = int(i)
            if not 0 <= u < self.n_units:
                raise ValueError(f"юнит {u} вне диапазона")
            self._buckets[u].extend(int(p) for p in items)
