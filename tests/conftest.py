"""Общие фикстуры тестов realMemory."""
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realmemory import Hippocampus, MemoryConfig
from realmemory.timeprov import FakeClock


@pytest.fixture
def tiny_cfg() -> MemoryConfig:
    cfg = MemoryConfig(
        dim=256,
        n_units=512,
        k_sparse=48,
        sdr_seed=5,
        bucket_cap=32,
        tau_episodic=100_000.0,
        tau_semantic=1_000_000.0,
        initial_strength=0.5,
        promote_after_reinforcements=5,
        promote_min_age_s=2000.0,
        tau_eligibility=50_000.0,
        tau_edge_stable=300_000.0,
    )
    cfg.validate()
    return cfg


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def hippo(tmp_path, tiny_cfg, clock):
    h = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    yield h
    h.close()
