"""Гейт новизны: классификация записи по косинусной близости к известным следам."""
from __future__ import annotations

from ..config import MemoryConfig
from ..types import DecisionAction


def gate(best_cosine: float, cfg: MemoryConfig) -> DecisionAction:
    """best_cosine >= theta_reinforce -> REINFORCE;
    >= theta_link -> LINK; иначе CREATE."""
    if best_cosine >= cfg.theta_reinforce:
        return DecisionAction.REINFORCE
    if best_cosine >= cfg.theta_link:
        return DecisionAction.LINK
    return DecisionAction.CREATE
