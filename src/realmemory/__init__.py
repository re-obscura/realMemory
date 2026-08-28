"""realMemory — персистентный слой памяти для LLM-агентов."""
from .config import MemoryConfig
from .hippocampus import Hippocampus
from .types import (
    ConsolidationReport,
    DecisionAction,
    RecalledMemory,
    RecallPacket,
    WriteDecision,
    WriteResult,
)

__version__ = "0.8.2"
__all__ = [
    "ConsolidationReport",
    "DecisionAction",
    "Hippocampus",
    "MemoryConfig",
    "RecallPacket",
    "RecalledMemory",
    "WriteDecision",
    "WriteResult",
]
