"""Политики: гейт новизны и затухание/повышение следов."""
from .decay import reinforce_values, retention, should_promote
from .novelty import gate

__all__ = ["gate", "reinforce_values", "retention", "should_promote"]
