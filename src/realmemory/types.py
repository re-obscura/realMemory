"""Публичные типы данных realMemory. Контракт: docs/CONTRACTS.md."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

KIND_EPISODIC = "episodic"
KIND_SEMANTIC = "semantic"
STATUS_ACTIVE = "active"
STATUS_SUPERSEDED = "superseded"
SCOPE_GLOBAL = "global"

SOURCE_DIRECT = "direct"
SOURCE_ASSOCIATED = "associated"
SOURCE_KEYWORD = "keyword"


class DecisionAction(str, Enum):
    """Решение гейта новизны при записи."""

    CREATE = "create"
    REINFORCE = "reinforce"
    LINK = "link"


@dataclass(frozen=True)
class WriteDecision:
    action: DecisionAction
    target_id: int | None = None
    related_ids: tuple[int, ...] = ()
    novelty: float = 1.0
    best_cosine: float = 0.0


@dataclass(frozen=True)
class WriteResult:
    memory_id: int
    decision: WriteDecision
    created: bool


@dataclass(frozen=True)
class RecalledMemory:
    memory_id: int
    text: str
    kind: str
    cosine: float
    confidence: float
    retention: float
    source: str
    created_at: float
    updated_at: float
    meta: dict[str, Any]
    scope: str = SCOPE_GLOBAL


@dataclass(frozen=True)
class RecallPacket:
    query: str
    items: tuple[RecalledMemory, ...]
    abstained: bool
    latency_ms: float


@dataclass(frozen=True)
class ConsolidationReport:
    edges_committed: int = 0
    edges_pruned: int = 0
    promoted_to_semantic: int = 0
    rewards_applied: int = 0
    journal_pruned: int = 0
    elapsed_ms: float = 0.0


@dataclass
class MemoryRecord:
    """Внутренняя единица хранения. Живёт в SQLite, кэшируется фасадом."""

    id: int | None
    text: str
    kind: str
    status: str
    meta: dict[str, Any]
    embedding: np.ndarray  # float32 (dim,)
    sdr: np.ndarray  # int32 sorted on-bits (k,)
    created_at: float
    updated_at: float
    reinforced_count: int
    last_reinforced_at: float
    base_strength: float
    valid_from: float
    valid_to: float | None = None
    superseded_by: int | None = None
    scope: str = SCOPE_GLOBAL  # 'global' или имя проекта; recall видит свой проект + global
