"""Затухание следов и повышение эпизодов до семантических.

Чистые функции над примитивами записи — без состояния, легко тестируются на
FakeClock. Retention = base_strength * exp(-(now - last_reinforced)/tau(kind)):
подкрепление поднимает base (с потолком strength_cap) и сбрасывает таймер,
семантические следы затухают медленнее эпизодических.
"""
from __future__ import annotations

import math

from ..config import MemoryConfig
from ..types import KIND_EPISODIC, KIND_SEMANTIC


def retention(
    base_strength: float,
    last_reinforced_at: float,
    now: float,
    kind: str,
    cfg: MemoryConfig,
) -> float:
    tau = cfg.tau_semantic if kind == KIND_SEMANTIC else cfg.tau_episodic
    dt = max(0.0, now - last_reinforced_at)
    r = base_strength * math.exp(-dt / tau)
    if r < 1e-9:  # численный ноль: след недоступен
        return 0.0
    return min(1.0, max(0.0, r))


def reinforce_values(
    base_strength: float,
    reinforced_count: int,
    cfg: MemoryConfig,
) -> tuple[float, int]:
    return min(cfg.strength_cap, base_strength * cfg.reinforce_bump), int(reinforced_count) + 1


def weaken_value(base_strength: float, reward: float) -> float:
    """Отрицательный reward ослабляет след пропорционально: -0.5 → ×0.5,
    -1 → 0. Таймер подкрепления и счётчик продвижения не трогаются."""
    if not -1.0 <= reward < 0.0:
        raise ValueError("weaken_value ожидает reward в [-1, 0)")
    return max(0.0, base_strength * (1.0 + reward))


def should_promote(
    kind: str,
    reinforced_count: int,
    created_at: float,
    now: float,
    cfg: MemoryConfig,
) -> bool:
    """Эпизод становится семантическим после достаточного числа подкреплений
    и минимального возраста (механическая семантизация «эпизод → факт»)."""
    if kind != KIND_EPISODIC:
        return False
    return (
        reinforced_count >= cfg.promote_after_reinforcements
        and (now - created_at) >= cfg.promote_min_age_s
    )
