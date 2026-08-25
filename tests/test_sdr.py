"""Контракты кодирования: детерминизм, монотонность перекрытий, метрики."""
import math

import numpy as np
import pytest

from realmemory.encoding.sdr import (
    BipolarProjector,
    SDREncoder,
    calibrate_sparse,
    jaccard,
    overlap,
    overlap_fraction,
)

DIM = 64


def _unit(rng, dim=DIM):
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _mix(base: np.ndarray, cos: float, rng) -> np.ndarray:
    """Вектор с точно заданной косинусной близостью cos к base."""
    w = rng.standard_normal(base.size).astype(np.float32)
    w -= base * float(w @ base)
    w /= np.linalg.norm(w)
    v = cos * base + math.sqrt(max(0.0, 1.0 - cos * cos)) * w
    return v / np.linalg.norm(v)


def test_encoder_deterministic_sorted_unique():
    e1 = SDREncoder(DIM, 512, 48, seed=11)
    e2 = SDREncoder(DIM, 512, 48, seed=11)
    v = _unit(np.random.default_rng(0))
    a, b = e1.encode(v), e2.encode(v)
    assert a.shape == (48,) and a.dtype == np.int32
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)
    e3 = SDREncoder(DIM, 512, 48, seed=12)
    assert not np.array_equal(a, e3.encode(v))


def test_wrong_shape_raises():
    enc = SDREncoder(DIM, 256, 16, seed=1)
    with pytest.raises(ValueError):
        enc.encode(np.zeros(DIM + 1, dtype=np.float32))


def test_zero_vector_conventions():
    enc = SDREncoder(DIM, 256, 16, seed=1)
    assert enc.encode(np.zeros(DIM, dtype=np.float32)).size == 0
    proj = BipolarProjector(DIM, 128, seed=1)
    out = proj.project(np.zeros(DIM, dtype=np.float32))
    assert out.dtype == np.int8 and np.all(out == 1)


def test_overlap_monotone_in_cosine():
    enc = SDREncoder(DIM, 1024, 96, seed=7)
    rng = np.random.default_rng(42)
    base = _unit(rng)
    levels = [-0.5, 0.0, 0.25, 0.5, 0.75, 0.95]
    means = []
    for cos in levels:
        ovs = [
            overlap(enc.encode(base), enc.encode(_mix(base, cos, rng))) for _ in range(40)
        ]
        means.append(float(np.mean(ovs)))
    assert means[-1] > means[0] + 10
    for lo, hi in zip(means, means[1:]):  # noqa: RUF007
        assert hi >= lo - 1.5  # статистическая монотонность
    assert overlap(enc.encode(base), enc.encode(base)) == 96


def test_metrics_basics():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([2, 3, 4], dtype=np.int32)
    assert overlap(a, b) == 2
    assert overlap_fraction(a, b) == pytest.approx(2 / 3)
    assert jaccard(a, b) == pytest.approx(2 / 4)
    assert overlap(np.empty(0, np.int32), a) == 0


def test_calibration_stats():
    enc = SDREncoder(DIM, 1024, 64, seed=5)
    st = calibrate_sparse(enc, n_samples=60, seed=1)
    assert st.n_pairs == 60 * 59 // 2
    assert st.mu_overlap < enc.k * 0.25
    assert st.sigma_overlap >= 0.0
