"""Контракты политик: гейт новизны, затухание, повышение эпизодов."""
import pytest

from realmemory.config import MemoryConfig
from realmemory.policies.decay import reinforce_values, retention, should_promote
from realmemory.policies.novelty import gate
from realmemory.types import DecisionAction


def test_gate_threshold_mapping():
    cfg = MemoryConfig()
    assert gate(0.9, cfg) is DecisionAction.REINFORCE
    assert gate(cfg.theta_reinforce, cfg) is DecisionAction.REINFORCE
    mid = (cfg.theta_link + cfg.theta_reinforce) / 2
    assert gate(mid, cfg) is DecisionAction.LINK
    assert gate(cfg.theta_link, cfg) is DecisionAction.LINK
    assert gate(0.0, cfg) is DecisionAction.CREATE


def test_retention_monotone_and_semantic_slower():
    cfg = MemoryConfig(tau_episodic=100.0, tau_semantic=1000.0)
    assert retention(0.5, 10.0, 10.0, "episodic", cfg) == pytest.approx(0.5)
    episodic = retention(0.5, 10.0, 110.0, "episodic", cfg)
    semantic = retention(0.5, 10.0, 110.0, "semantic", cfg)
    assert 0.0 < episodic < 0.5
    assert semantic > episodic
    assert retention(0.5, 10.0, 10_000.0, "episodic", cfg) == 0.0
    # время назад не увеличивает retention сверх base
    assert retention(0.5, 110.0, 10.0, "episodic", cfg) == 0.5


def test_reinforce_values_bump_and_cap():
    cfg = MemoryConfig(initial_strength=0.5, reinforce_bump=1.25, strength_cap=1.0)
    base, count = reinforce_values(0.5, 0, cfg)
    assert (base, count) == (pytest.approx(0.625), 1)
    base, count = reinforce_values(0.99, 4, cfg)
    assert base == 1.0 and count == 5


def test_should_promote_boundaries():
    cfg = MemoryConfig(promote_after_reinforcements=3, promote_min_age_s=100.0)
    assert not should_promote("episodic", 2, 0.0, 500.0, cfg)   # мало подкреплений
    assert not should_promote("episodic", 3, 0.0, 50.0, cfg)    # мал возраст
    assert should_promote("episodic", 3, 0.0, 150.0, cfg)
    assert not should_promote("semantic", 10, 0.0, 150.0, cfg)  # уже семантический


def test_config_validation():
    good = MemoryConfig()
    good.validate()
    bad = MemoryConfig(theta_link=0.5, theta_reinforce=0.45)
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        MemoryConfig(initial_strength=1.2).validate()
