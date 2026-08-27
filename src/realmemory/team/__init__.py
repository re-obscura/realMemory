"""Командный слой realMemory: явный шеринг с локальным registry публикаций.

Модули:
  identity  — кто я (identity-файл → git config → пусто);
  policy    — правила «что МОЖНО публиковать» из ~/.realmemory/team.yaml
              (+ never-rules, работающие fail-closed);
  registry  — журнал публикаций и отзывов (tombstones), источник истины
              будущей сетевой синхронизации; сама сеть — этап позже.

Приватность по умолчанию: ни одна функция здесь не отправляет байты наружу;
`publish` только фиксирует решение локально. Обычные memorize/recall ведут
себя ровно как раньше.
"""
from .identity import resolve_identity
from .policy import (
    DEFAULT_POLICY_PATH,
    ShareDecision,
    TeamPolicy,
    classify,
    load_policy,
    save_policy,
    set_project_shareable,
)
from .registry import (
    PublicationRow,
    RegistryError,
    active_publications,
    publish,
    retract,
)

__all__ = [
    "DEFAULT_POLICY_PATH",
    "PublicationRow",
    "RegistryError",
    "ShareDecision",
    "TeamPolicy",
    "active_publications",
    "classify",
    "load_policy",
    "publish",
    "resolve_identity",
    "retract",
    "save_policy",
    "set_project_shareable",
]
