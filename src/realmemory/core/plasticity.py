"""Пластичность: eligibility-лог (третий фактор) и агрегация направленных пар.

Свежие bind'ы накапливаются в логе с меткой времени и источником; к моменту
коммита каждый вклад затухает exp(-dt/tau) — это «след готовности» синапса.
Reward от агента умножает силу ещё не закоммиченных событий (three-factor
learning: pre/post активность × нейромодуляция).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _units(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("ожидался одномерный массив юнитов")
    return arr


@dataclass
class EligibilityEvent:
    src_units: np.ndarray  # int32
    dst_units: np.ndarray  # int32
    strength: float
    created_at: float
    source_ids: frozenset[int]  # id следов-источников (для reward)


class EligibilityLog:
    def __init__(self, tau: float) -> None:
        if tau <= 0:
            raise ValueError("tau должен быть положительным")
        self.tau = float(tau)
        self._events: list[EligibilityEvent] = []

    @property
    def pending_count(self) -> int:
        return len(self._events)

    def add(
        self,
        src_units: np.ndarray,
        dst_units: np.ndarray,
        strength: float,
        now: float,
        source_ids,
    ) -> None:
        src = _units(src_units)
        dst = _units(dst_units)
        if src.size != dst.size or src.size == 0:
            raise ValueError("src/dst должны быть непустыми и одинаковой длины")
        if strength <= 0:
            raise ValueError("strength должен быть > 0")
        self._events.append(
            EligibilityEvent(src, dst, float(strength), float(now),
                             frozenset(int(i) for i in source_ids))
        )

    def reward(self, source_ids, reward: float) -> int:
        """Усилить/ослабить события, затрагивающие данные следы. Возвращает #событий."""
        ids = {int(i) for i in source_ids}
        factor = max(0.0, 1.0 + float(reward))
        touched = 0
        for e in self._events:
            if e.source_ids & ids and factor != 1.0:
                e.strength *= factor
                touched += 1
        return touched

    def commit(self, now: float):
        """Слить все события в агрегированные пары с затуханием; очистить лог.

        Возвращает (src, dst, weights): int32/int32/float32, слитые по ключам.
        """
        srcs, dsts, ws = [], [], []
        for e in self._events:
            eff = e.strength * math.exp(-(now - e.created_at) / self.tau)
            if eff > 1e-12:  # затухший вклад и обнулённый reward не дают ребра
                srcs.append(e.src_units)
                dsts.append(e.dst_units)
                ws.append(np.full(e.src_units.size, eff, dtype=np.float32))
        self._events.clear()
        if not srcs:
            return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
        return merge_pairs(np.concatenate(srcs), np.concatenate(dsts), np.concatenate(ws))

    # -- снапшот -----------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "tau": self.tau,
            "events": [
                (e.src_units.tolist(), e.dst_units.tolist(), e.strength, e.created_at,
                 sorted(e.source_ids))
                for e in self._events
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        tau = float(state["tau"])
        if tau <= 0:
            raise ValueError("tau должен быть положительным")
        self.tau = tau
        self._events = []
        for src, dst, strength, created_at, source_ids in state["events"]:
            if float(strength) <= 0.0:
                # негативный reward мог обнулить ещё незакоммиченное событие —
                # оно отменено и ребра не даст; пропускаем вместо ошибки
                continue
            self.add(np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32),
                     float(strength), float(created_at), [int(i) for i in source_ids])


def merge_pairs(src, dst, w):
    """Агрегация дубликатов направленных пар суммой весов.

    Ключ пары — (src << 32) | dst; юниты предполагаются >= 0 и < 2^31.
    """
    src = _units(src).astype(np.int64)
    dst = _units(dst).astype(np.int64)
    w = np.asarray(w, dtype=np.float64)
    if not (src.size == dst.size == w.size):
        raise ValueError("src, dst, w должны быть одной длины")
    keys = (src << 32) | dst
    uniq, inverse = np.unique(keys, return_inverse=True)
    agg = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(agg, inverse, w)
    u_src = (uniq >> 32).astype(np.int32)
    u_dst = (uniq & 0xFFFFFFFF).astype(np.int32)
    return u_src, u_dst, agg.astype(np.float32)
