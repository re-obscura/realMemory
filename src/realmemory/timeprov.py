"""Провайдеры времени. Все тайминги realMemory идут через TimeProvider,
чтобы тесты могли двигать часы (FakeClock) без ожидания реального времени."""
from __future__ import annotations

import time
from typing import Protocol


class TimeProvider(Protocol):
    def now(self) -> float: ...


class SystemClock:
    """Реальное системное время."""

    def now(self) -> float:
        return time.time()


class FakeClock:
    """Управляемое время для тестов: старт с фиксированной эпохи, ручной ход."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("нельзя двигать часы назад")
        self._now += float(seconds)
        return self._now
