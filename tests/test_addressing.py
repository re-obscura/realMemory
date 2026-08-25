"""Контракты L1: голосование по инвертированному индексу SDR-юнитов.

Селективность следует из геометрии разреженных множеств: идентичный паттерн
даёт k голосов, коррелированный с долей rho общих юнитов — ~rho*k,
случайный — ~k^2/N.
"""
import numpy as np
import pytest

from realmemory.core.addressing import QueryResult, SDRVotingIndex

N_UNITS = 512
K = 48


def _rand_sdr(rng):
    return np.sort(rng.choice(N_UNITS, size=K, replace=False)).astype(np.int32)


def _noisy_copy(sdr, rng, replace_frac=0.3):
    """Копия с заменой части юнитов на случайные (сохраняет размер)."""
    n_replace = int(K * replace_frac)
    out = sdr.copy()
    drop = rng.choice(K, size=n_replace, replace=False)
    pool = np.setdiff1d(np.arange(N_UNITS), sdr, assume_unique=True)
    add = rng.choice(pool, size=n_replace, replace=False)
    out[drop] = add
    return np.sort(out).astype(np.int32)


def test_exact_recall_top1_always():
    idx = SDRVotingIndex(N_UNITS, bucket_cap=64)
    rng = np.random.default_rng(0)
    pairs = [(_rand_sdr(rng), i) for i in range(200)]
    for sdr, p in pairs:
        idx.write(sdr, p)
    hits = sum(int(idx.query(sdr, 3).candidates[0] == p) for sdr, p in pairs)
    assert hits == len(pairs)


def test_correlated_query_found():
    idx = SDRVotingIndex(N_UNITS, bucket_cap=64)
    rng = np.random.default_rng(1)
    ok, trials = 0, 200
    for t in range(trials):
        sdr = _rand_sdr(rng)
        idx.write(sdr, 1000 + t)
        qr = idx.query(_noisy_copy(sdr, rng, replace_frac=0.3), max_candidates=3)
        ok += int((1000 + t) in qr.candidates.tolist())
    assert ok >= 0.98 * trials


def test_random_query_low_false_positive():
    idx = SDRVotingIndex(N_UNITS, bucket_cap=64)
    rng = np.random.default_rng(2)
    # на заполненном индексе: случайный запрос делит с каждым ~K^2/N ≈ 4.5
    # юнитов одинаково, цель конкурирует с сотнями равных — в топ-5 редко
    idx.write(_rand_sdr(rng), 777)
    for p in range(200):
        idx.write(_rand_sdr(rng), p)
    fp, trials = 0, 300
    for _ in range(trials):
        qr = idx.query(_rand_sdr(rng), max_candidates=5)
        fp += int(777 in qr.candidates.tolist())
    assert fp / trials < 0.10


def test_votes_scale_with_overlap():
    idx = SDRVotingIndex(N_UNITS, bucket_cap=1024)
    rng = np.random.default_rng(3)
    target = _rand_sdr(rng)
    idx.write(target, 42)
    v_self = int(idx.query(target, 5).votes[0])
    v_half = int(idx.query(_noisy_copy(target, rng, replace_frac=0.5), 5).votes[0])
    v_rand = int(idx.query(_rand_sdr(rng), 5).votes[0])
    assert v_self == K
    assert 0.25 * K <= v_half <= 0.85 * K
    assert v_rand <= 0.25 * K


def test_write_dedupe_keeps_votes_stable():
    idx = SDRVotingIndex(N_UNITS, bucket_cap=64)
    rng = np.random.default_rng(4)
    sdr = _rand_sdr(rng)
    idx.write(sdr, 1)
    v1 = idx.query(sdr, 5).votes[0]
    idx.write(sdr, 1)
    idx.write(sdr, 1)
    v2 = idx.query(sdr, 5).votes[0]
    assert int(v1) == int(v2)


def test_bucket_cap_evicts_oldest():
    idx = SDRVotingIndex(N_UNITS, bucket_cap=4)
    rng = np.random.default_rng(5)
    sdr = _rand_sdr(rng)
    for p in range(10):
        idx.write(sdr, p)
    got = set(idx.query(sdr, 20).candidates.tolist())
    assert got == {6, 7, 8, 9}


def test_load_factor_and_empty_query():
    idx = SDRVotingIndex(N_UNITS, bucket_cap=16)
    qr = idx.query(_rand_sdr(np.random.default_rng(6)), 5)
    assert isinstance(qr, QueryResult) and qr.candidates.size == 0
    assert idx.load_factor() == 0.0


def test_out_of_range_unit_raises():
    idx = SDRVotingIndex(N_UNITS, bucket_cap=4)
    with pytest.raises(ValueError):
        idx.write(np.array([N_UNITS + 1]), 1)
    with pytest.raises(ValueError):
        idx.query(np.array([-1]), 5)


def test_state_roundtrip():
    idx = SDRVotingIndex(N_UNITS, bucket_cap=8)
    rng = np.random.default_rng(7)
    sdrs = [_rand_sdr(rng) for _ in range(20)]
    for p, sdr in enumerate(sdrs):
        idx.write(sdr, p)
    fresh = SDRVotingIndex(N_UNITS, bucket_cap=8)
    fresh.load_state_dict(idx.state_dict())
    for p, sdr in enumerate(sdrs):
        qr = fresh.query(sdr, 1)
        assert qr.candidates.size >= 1 and int(qr.candidates[0]) == p
