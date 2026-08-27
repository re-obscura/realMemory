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


def gate_for_author(best_cosine: float, cfg: MemoryConfig,
                    *, same_author: bool | None) -> DecisionAction:
    """Авторозависимый вариант гейта для командного слоя.

    Близкая запись ОТ ДРУГОГО автора никогда не усиливает чужой след:
    REINFORCE деградирует до LINK (две сохраняющие авторство точки зрения
    на одну тему остаются различимы и связываются), а ниже порога связи —
    до CREATE. same_author=None означает «сравнивать не с чем» (у incoming
    записи нет identity либо целевой след старше командного слоя) — правило
    не применяется, поведение историческое. Это же сохраняет совместимость
    личных баз, где author пуст у всех записей.
    """
    action = gate(best_cosine, cfg)
    if action is DecisionAction.REINFORCE and same_author is False:
        return (DecisionAction.LINK
                if best_cosine >= cfg.theta_link else DecisionAction.CREATE)
    return action
