"""Контракты пластичности: агрегация пар, затухание eligibility, reward."""
import numpy as np
import pytest

from realmemory.core.plasticity import EligibilityLog, merge_pairs


def test_merge_pairs_aggregates_duplicates():
    src = np.array([1, 1, 2], np.int32)
    dst = np.array([2, 2, 3], np.int32)
    w = np.array([1.0, 1.0, 1.0])
    s, d, w2 = merge_pairs(src, dst, w)
    m = dict(zip(zip(s.tolist(), d.tolist()), w2.tolist()))
    assert m[(1, 2)] == pytest.approx(2.0)
    assert m[(2, 3)] == pytest.approx(1.0)


def test_commit_applies_decay_and_clears():
    log = EligibilityLog(tau=10.0)
    log.add(np.array([1]), np.array([2]), strength=1.0, now=0.0, source_ids=[5])
    log.add(np.array([1]), np.array([2]), strength=1.0, now=5.0, source_ids=[5])
    src, dst, w = log.commit(now=10.0)
    assert log.pending_count == 0
    expected = float(np.exp(-1.0) + np.exp(-0.5))
    assert len(w) == 1 and w[0] == pytest.approx(expected, rel=1e-6)
    assert src[0] == 1 and dst[0] == 2


def test_reward_amplifies_pending_only():
    log = EligibilityLog(tau=100.0)
    log.add(np.array([1]), np.array([2]), 1.0, 0.0, [7])
    assert log.reward([7], 1.0) == 1
    _, _, w = log.commit(now=0.0)
    assert w[0] == pytest.approx(2.0)


def test_negative_reward_zeroes_contribution():
    log = EligibilityLog(tau=100.0)
    log.add(np.array([1]), np.array([2]), 1.0, 0.0, [7])
    log.reward([7], -1.0)
    src, dst, w = log.commit(now=0.0)
    # обнулённое событие не даёт ребра вовсе
    assert src.size == dst.size == w.size == 0


def test_fully_decayed_commit_is_empty():
    log = EligibilityLog(tau=1.0)
    log.add(np.array([1]), np.array([2]), 1.0, 0.0, [7])
    src, dst, w = log.commit(now=50.0)
    assert src.size == dst.size == w.size == 0


def test_reward_scopes_to_source_ids():
    log = EligibilityLog(tau=100.0)
    log.add(np.array([1]), np.array([2]), 1.0, 0.0, [7])
    log.add(np.array([3]), np.array([4]), 1.0, 0.0, [8])
    assert log.reward([7], 1.0) == 1
    _, _, w = log.commit(now=0.0)
    assert sorted(w.tolist()) == [1.0, 2.0]


def test_state_roundtrip():
    log = EligibilityLog(tau=10.0)
    log.add(np.array([1, 2]), np.array([3, 4]), 0.7, 1.0, {9})
    state = log.state_dict()
    fresh = EligibilityLog(tau=1.0)
    fresh.load_state_dict(state)
    assert fresh.pending_count == 1 and fresh.tau == 10.0
    s, d, w = fresh.commit(now=1.0)
    assert w.size == 2  # две различные пары из одного события
    assert dict(zip(zip(s.tolist(), d.tolist()), w.tolist()))[(1, 3)] == pytest.approx(0.7)
    assert dict(zip(zip(s.tolist(), d.tolist()), w.tolist()))[(2, 4)] == pytest.approx(0.7)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        EligibilityLog(tau=0.0)
    log = EligibilityLog(10.0)
    with pytest.raises(ValueError):
        log.add(np.array([1]), np.array([]), 1.0, 0.0, [1])
    with pytest.raises(ValueError):
        log.add(np.array([1]), np.array([2]), 0.0, 0.0, [1])
