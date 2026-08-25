"""Контракты L2-сборочной сети: связывание, spread, multi-hop, распад, снапшот."""
import numpy as np
import pytest

from realmemory.core.assembly import AssemblyNetwork

UNITS = 1024


def _pat(rng, k=24):
    return np.sort(rng.choice(UNITS, size=k, replace=False)).astype(np.int32)


def test_bind_then_spread_reaches_partner_not_stranger():
    rng = np.random.default_rng(0)
    net = AssemblyNetwork(UNITS, edge_min_weight=0.01, tau_edge_stable=1e9,
                          seed=1, max_pairs_per_bind=256)
    a, b, c = _pat(rng), _pat(rng), _pat(rng)
    net.bind(a, b, strength=1.0, now=0.0)
    units, scores = net.spread(a, depth=1, alpha=0.5, top_m=64, eps=0.001)
    u = set(units.tolist())
    assert len(u & set(b.tolist())) >= 8
    assert len(u & set(c.tolist())) <= 2
    # query-юниты входят с 1.0; хабовые юниты партнёра могут набирать сумму > 1
    assert 1.0 in np.asarray(scores, dtype=np.float32).tolist()
    assert float(scores.max()) >= 1.0


def test_pair_cap_both_directions():
    rng = np.random.default_rng(1)
    net = AssemblyNetwork(UNITS, seed=2, max_pairs_per_bind=32)
    added = net.bind(_pat(rng, 64), _pat(rng, 64), 1.0, 0.0)
    assert added == 64  # 32 пары x 2 направления


def test_self_bind_creates_intra_pattern_edges():
    """bind паттерна с самим собой — автоассоциация внутри сборки (легитимно).
    Все k(k-1) направленных рёбер между юнитами паттерна новы."""
    rng = np.random.default_rng(9)
    net = AssemblyNetwork(UNITS, seed=3)
    p = _pat(rng, 8)
    added = net.bind(p, p, 1.0, 0.0)
    assert added == 8 * 7
    assert net.edge_count == added
    units, _scores = net.spread(p, depth=1, eps=1e-3)
    assert set(p.tolist()) <= set(units.tolist())


def test_decay_reduces_then_prunes_all():
    rng = np.random.default_rng(3)
    net = AssemblyNetwork(UNITS, edge_min_weight=0.05, tau_edge_stable=100.0, seed=4)
    a, b = _pat(rng), _pat(rng)
    net.bind(a, b, 1.0, now=0.0)
    w0 = net.total_weight
    remaining = net.decay_tick(now=100.0)
    assert remaining > 0
    assert net.total_weight <= w0 * np.exp(-1.0) + 1e-6
    net.decay_tick(now=10_000.0)
    assert net.edge_count == 0


def test_spread_multihop_two_steps():
    rng = np.random.default_rng(5)
    net = AssemblyNetwork(UNITS, edge_min_weight=0.001, tau_edge_stable=1e9,
                          seed=6, max_pairs_per_bind=256)
    a, b, c = _pat(rng), _pat(rng), _pat(rng)
    net.bind(a, b, 1.0, 0.0)
    net.bind(b, c, 1.0, 0.0)
    units, _ = net.spread(a, depth=2, alpha=0.6, top_m=200, eps=1e-4)
    assert len(set(units.tolist()) & set(c.tolist())) >= 3


def test_spread_validation_and_empty():
    net = AssemblyNetwork(UNITS, seed=7)
    units, scores = net.spread([], depth=2)
    assert units.size == 0 and scores.size == 0
    with pytest.raises(ValueError):
        net.spread([UNITS + 5])


def test_neighbors_and_weights():
    net = AssemblyNetwork(UNITS, seed=8)
    a = np.array([10, 20, 30], dtype=np.int32)
    b = np.array([40, 50], dtype=np.int32)
    net.bind(a, b, 0.5, 0.0)
    nb = dict(net.neighbors(10))
    assert set(nb) == {40, 50}
    assert all(v == pytest.approx(0.5) for v in nb.values())


def test_state_roundtrip_preserves_spread():
    rng = np.random.default_rng(10)
    net = AssemblyNetwork(UNITS, edge_min_weight=0.001, tau_edge_stable=1e9, seed=11)
    a, b = _pat(rng), _pat(rng)
    net.bind(a, b, 1.5, 0.0)
    st = net.state_dict()
    fresh = AssemblyNetwork(UNITS, seed=99)
    fresh.load_state_dict(st)
    u1, s1 = net.spread(a, depth=1, eps=1e-3)
    u2, s2 = fresh.spread(a, depth=1, eps=1e-3)
    assert np.array_equal(u1, u2)
    assert np.allclose(s1, s2)
