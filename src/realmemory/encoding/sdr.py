"""Кодирование векторов: плотные биполярные адреса (L1) и разреженные SDR (L2).

Почему адресация L1 сделана на плотных биполярных векторах, а не на SDR:
расстояние Хэмминга двух независимых биполярных векторов распределено
Binomial(n_bits, 1/2) с mu=n_bits/2 и сигмой sqrt(n_bits)/2, а близкие входы
(косинус rho) дают d ~= n_bits*(1-rho)/2 — режимы «случайно/похоже/идентично»
разделены десятками сигм. У разреженных множеств перекрытие концентрируется
около k^2/N, и фиксированная случайная «локация» не может быть близкой к
широкому классу паттернов — селективности нет (см. docs/ARCHITECTURE.md §3).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


class BipolarProjector:
    """Вектор -> биполярный адрес ±1 длины n_bits: sign(W·v̂).

    Нулевой вход проецируется во все +1 (конвенция sign(0)=+1) — такие адреса
    считаются вырожденными и отбраковываются вызывающей стороной.
    """

    def __init__(self, dim: int, n_bits: int, seed: int = 13) -> None:
        if dim <= 0 or n_bits <= 0:
            raise ValueError("dim и n_bits должны быть положительными")
        rng = np.random.default_rng(seed)
        self.dim = int(dim)
        self.n_bits = int(n_bits)
        self._w = rng.standard_normal((n_bits, dim)).astype(np.float32)
        self._w /= np.sqrt(np.float32(dim))

    def project(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"ожидался вектор формы ({self.dim},), получен {v.shape}")
        n = float(np.linalg.norm(v))
        s = self._w @ v if n == 0.0 else self._w @ (v / n)
        return np.where(s >= 0.0, np.int8(1), np.int8(-1))


class SDREncoder:
    """Вектор -> SDR: k on-битов (top-k случайной проекции), отсортированных.

    Перекрытие SDR монотонно связано с косинусной близостью входов
    (проверяется тестом); используется как пространство юнитов L2.
    """

    def __init__(self, dim: int, n_units: int, k: int, seed: int = 29) -> None:
        if dim <= 0 or n_units <= 0 or k <= 0:
            raise ValueError("dim, n_units, k должны быть положительными")
        rng = np.random.default_rng(seed)
        self.dim = int(dim)
        self.n_units = int(n_units)
        self.k = min(int(k), int(n_units))
        self._w = rng.standard_normal((n_units, dim)).astype(np.float32)
        self._w /= np.sqrt(np.float32(dim))

    def encode(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        if v.shape != (self.dim,):
            raise ValueError(f"ожидался вектор формы ({self.dim},), получен {v.shape}")
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return np.empty(0, dtype=np.int32)
        p = self._w @ (v / n)
        part = np.argpartition(-p, self.k - 1)[: self.k]
        return np.sort(part).astype(np.int32)


def overlap(a: np.ndarray, b: np.ndarray) -> int:
    """Размер пересечения двух множеств on-битов."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0
    return int(np.intersect1d(a, b, assume_unique=True).size)


def overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    """|A∩B| / min(|A|,|B|); пустые множества -> 0.0."""
    denom = min(int(np.asarray(a).size), int(np.asarray(b).size))
    if denom == 0:
        return 0.0
    return overlap(a, b) / denom


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    inter = overlap(a, b)
    union = int(a.size) + int(b.size) - inter
    return inter / union if union else 0.0


@dataclass(frozen=True)
class CalibrationStats:
    mu_overlap: float
    sigma_overlap: float
    n_pairs: int
    expected_random_fraction: float  # mu / k — ожидаемая доля шума в перекрытии


def calibrate_sparse(encoder: SDREncoder, n_samples: int = 120, seed: int = 99) -> CalibrationStats:
    """Статистика перекрытий случайных SDR — базовая линия для порогов новизны."""
    rng = np.random.default_rng(seed)
    sdrs = []
    for _ in range(int(n_samples)):
        v = rng.standard_normal(encoder.dim).astype(np.float32)
        v /= np.linalg.norm(v)
        sdrs.append(encoder.encode(v))
    ovs = np.asarray(
        [overlap(sdrs[i], sdrs[j]) for i, j in combinations(range(len(sdrs)), 2)],
        dtype=np.float64,
    )
    mu = float(ovs.mean())
    return CalibrationStats(
        mu_overlap=mu,
        sigma_overlap=float(ovs.std()),
        n_pairs=int(ovs.size),
        expected_random_fraction=mu / max(1, encoder.k),
    )
