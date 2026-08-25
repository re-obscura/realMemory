"""L2 — сборочная сеть: пластичные ассоциации между юнитами SDR-пространства.

Рёбра направленные, key=(i·n_units+j) -> вес. То, что активировалось вместе,
связывается (bind); spread распространяет активацию по рёбрам с затуханием —
это ассоциативный обход графа «что с чем вспоминалось», вырастающий из
статистики использования. Стабильные веса распадаются экспоненциально между
консолидациями (decay_tick), слабые обрезаются.
"""
from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np


class AssemblyNetwork:
    def __init__(
        self,
        n_units: int,
        edge_min_weight: float = 0.02,
        tau_edge_stable: float = 90 * 86400.0,
        seed: int = 47,
        max_pairs_per_bind: int = 256,
    ) -> None:
        if n_units <= 0 or max_pairs_per_bind <= 0:
            raise ValueError("n_units и max_pairs_per_bind должны быть положительными")
        if edge_min_weight <= 0 or tau_edge_stable <= 0:
            raise ValueError("edge_min_weight и tau_edge_stable должны быть положительными")
        self.n_units = int(n_units)
        self.edge_min_weight = float(edge_min_weight)
        self.tau = float(tau_edge_stable)
        self.max_pairs_per_bind = int(max_pairs_per_bind)
        self._rng = np.random.default_rng(seed)
        self._edges: dict[int, float] = {}
        self._last_tick: float | None = None
        # ленивый CSR-индекс для обхода
        self._csr_dirty = True
        self._keys_sorted: np.ndarray = np.empty(0, dtype=np.int64)
        self._dst_sorted: np.ndarray = np.empty(0, dtype=np.int64)  # dst = key % n_units
        self._ws: np.ndarray = np.empty(0, dtype=np.float32)
        self._indptr: np.ndarray = np.zeros(n_units + 1, dtype=np.int64)

    # -- служебное ---------------------------------------------------------------

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def total_weight(self) -> float:
        return float(sum(self._edges.values()))

    def _add_edge(self, i: int, j: int, w: float) -> int:
        """Добавить вес к ребру. Возвращает 1 только если ребро НОВОЕ
        (слияние дубликатов не считается добавлением)."""
        if i == j:
            return 0
        key = i * self.n_units + j
        existed = key in self._edges
        self._edges[key] = self._edges.get(key, 0.0) + w
        self._csr_dirty = True
        return 0 if existed else 1

    def _check_units(self, units: np.ndarray) -> np.ndarray:
        u = np.unique(np.asarray(units, dtype=np.int64))
        if u.size and (u.min() < 0 or u.max() >= self.n_units):
            raise ValueError(f"юниты должны быть в [0, {self.n_units})")
        return u

    # -- запись --------------------------------------------------------------------

    def bind(self, units_a, units_b, strength: float, now: float) -> int:
        """Связать два паттерна (обе стороны). Возвращает #новых рёбер.

        Перед записью применяет распад существующих рёбер за прошедшее время:
        сетевые часы двигаются любой записью, забывание идёт непрерывно,
        а не только во время консолидации.
        """
        a = self._check_units(units_a)
        b = self._check_units(units_b)
        if a.size == 0 or b.size == 0 or strength <= 0:
            return 0
        self._advance_time(now)
        total = a.size * b.size
        cap = min(self.max_pairs_per_bind, int(total))
        flat = self._rng.choice(total, size=cap, replace=False)
        ii = a[flat // b.size]
        jj = b[flat % b.size]
        added = 0
        for i, j in zip(ii.tolist(), jj.tolist()):
            added += self._add_edge(i, j, strength)
            added += self._add_edge(j, i, strength)
        return added

    def commit_eligibility(self, src, dst, w, now: float) -> None:
        """Влить закоммиченный eligibility-батч (сначала decay_tick)."""
        src = np.asarray(src, dtype=np.int32)
        dst = np.asarray(dst, dtype=np.int32)
        w = np.asarray(w, dtype=np.float32)
        self.decay_tick(now)
        for i, j, wi in zip(src.tolist(), dst.tolist(), w.tolist()):
            self._add_edge(int(i), int(j), float(wi))

    def _advance_time(self, now: float) -> None:
        """Продвинуть сетевые часы с распадом; вызывается любой записью."""
        if self._last_tick is not None and now < self._last_tick:
            raise ValueError("время не может идти назад")
        if self._last_tick is None or now == self._last_tick:
            self._last_tick = now
            return
        factor = math.exp(-(now - self._last_tick) / self.tau)
        dead: list[int] = []
        for key, w in self._edges.items():
            nw = w * factor
            if nw < self.edge_min_weight:
                dead.append(key)
            else:
                self._edges[key] = nw
        for key in dead:
            del self._edges[key]
        self._last_tick = now
        self._csr_dirty = True

    def decay_tick(self, now: float) -> int:
        """Экспоненциальный распад всех стабильных рёбер за прошедшее время,
        обрезка ниже edge_min_weight. Возвращает оставшееся число рёбер."""
        self._advance_time(float(now))
        return len(self._edges)

    # -- чтение ---------------------------------------------------------------------

    def _build_csr(self) -> None:
        if not self._csr_dirty:
            return
        if self._edges:
            keys = np.fromiter(self._edges.keys(), dtype=np.int64, count=len(self._edges))
            ws = np.fromiter(self._edges.values(), dtype=np.float32, count=len(self._edges))
            order = np.argsort(keys, kind="stable")
            keys, ws = keys[order], ws[order]
            srcs = keys // self.n_units
            self._indptr = np.searchsorted(srcs, np.arange(self.n_units + 1))
            self._keys_sorted = keys
            self._dst_sorted = keys % self.n_units
            self._ws = ws
        else:
            self._indptr = np.zeros(self.n_units + 1, dtype=np.int64)
            self._keys_sorted = np.empty(0, dtype=np.int64)
            self._dst_sorted = np.empty(0, dtype=np.int64)
            self._ws = np.empty(0, dtype=np.float32)
        self._csr_dirty = False

    def spread(
        self,
        query_units,
        depth: int = 2,
        alpha: float = 0.5,
        top_m: int = 32,
        eps: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Распространение активации от query-юнитов по рёбрам.

        Возвращает (units, scores): юниты по убыванию суммарной активации,
        query-юниты входят со score=1.0. Юниты с score < eps отбрасываются.
        """
        q = self._check_units(query_units)
        if depth < 0 or alpha <= 0 or top_m < 1 or eps < 0:
            raise ValueError("некорректные параметры spread")
        self._build_csr()
        act: dict[int, float] = {int(u): 1.0 for u in q.tolist()}
        frontier = dict(act)
        for _ in range(int(depth)):
            if not frontier:
                break
            incoming: dict[int, float] = {}
            for u, sc in frontier.items():
                lo, hi = int(self._indptr[u]), int(self._indptr[u + 1])
                if hi <= lo:
                    continue
                nbrs = self._dst_sorted[lo:hi]
                wvs = self._ws[lo:hi]
                for nb, wv in zip(nbrs.tolist(), wvs.tolist()):
                    val = sc * alpha * float(wv)
                    if val >= eps:
                        incoming[nb] = incoming.get(nb, 0.0) + val
            nxt: dict[int, float] = {}
            for nb, val in incoming.items():
                act[nb] = act.get(nb, 0.0) + val
                nxt[nb] = val
            frontier = dict(sorted(nxt.items(), key=lambda kv: -kv[1])[:top_m])
        ranked = sorted(act.items(), key=lambda kv: (-kv[1], kv[0]))[:top_m]
        units = np.fromiter((u for u, _ in ranked), dtype=np.int32, count=len(ranked))
        scores = np.fromiter((s for _, s in ranked), dtype=np.float32, count=len(ranked))
        return units, scores

    def neighbors(self, unit: int) -> Iterator[tuple[int, float]]:
        """Прямые соседи юнита (для отладки/тестов)."""
        if not 0 <= int(unit) < self.n_units:
            raise ValueError("юнит вне диапазона")
        self._build_csr()
        lo, hi = int(self._indptr[unit]), int(self._indptr[unit + 1])
        for k, wv in zip(self._dst_sorted[lo:hi].tolist(), self._ws[lo:hi].tolist()):
            yield int(k), float(wv)

    # -- снапшот ---------------------------------------------------------------------

    def state_dict(self) -> dict:
        self._build_csr()
        return {
            "keys": self._keys_sorted.copy(),
            "weights": self._ws.copy(),
            "last_tick": self._last_tick,
        }

    def load_state_dict(self, state: dict) -> None:
        keys = np.asarray(state["keys"], dtype=np.int64)
        ws = np.asarray(state["weights"], dtype=np.float32)
        if keys.size != ws.size:
            raise ValueError("keys и weights должны совпадать по длине")
        self._edges = (
            {int(k): float(wv) for k, wv in zip(keys.tolist(), ws.tolist())} if keys.size else {}
        )
        self._last_tick = state.get("last_tick")
        self._csr_dirty = True
