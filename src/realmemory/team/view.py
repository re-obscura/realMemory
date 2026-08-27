"""Сборка данных для представления командного слоя (CLI/TUI).

Чистые функции над store+policy без сетевого I/O: TUI остаётся тонким
рендерером, а все решения принимает policy/registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..types import MemoryRecord
from .policy import ELIGIBLE, TeamPolicy, classify


@dataclass
class ProjectView:
    name: str
    total: int = 0                 # активных следов в скоупе
    eligible: int = 0              # проходят все правила сейчас
    blocked: int = 0               # never-правила
    published: int = 0             # активных публикаций из registry
    records: list[tuple[MemoryRecord, str, str]] = field(default_factory=list)
    # records: (след, статус classify, человекочитаемая причина)


def _iter_scope_records(store, scope: str) -> list[MemoryRecord]:
    return [rec for rec in store.iter_active() if rec.scope == scope]


def projects_view(store, policy: TeamPolicy) -> list[ProjectView]:
    """Карточки всех известных проектов: живые скоупы ∪ включённые в правила."""
    names: dict[str, ProjectView] = {}
    for scope, n in sorted(store.scope_counts().items()):
        names[scope] = ProjectView(name=scope, total=int(n))
    for rule in policy.projects:
        if rule.name not in names:
            names[rule.name] = ProjectView(name=rule.name)

    pub_by_project: dict[str, int] = {}
    for pub in store.publications_active():
        key = str(pub[2])
        pub_by_project[key] = pub_by_project.get(key, 0) + 1

    views: list[ProjectView] = []
    for name in sorted(names):
        view = names[name]
        view.published = pub_by_project.get(name, 0)
        for rec in _iter_scope_records(store, name):
            decision = classify(rec, policy)
            view.records.append((rec, decision.status, decision.reason))
            if decision.status == ELIGIBLE:
                view.eligible += 1
        statuses = {s for _, s, _ in view.records}
        view.blocked = sum(1 for _, s, _ in view.records
                           if s.startswith("blocked"))
        view.total = max(view.total, len(view.records))   # count() может отставать от iter
        del statuses
        views.append(view)
    return views


def find_project(views: list[ProjectView], name: str) -> ProjectView | None:
    for v in views:
        if v.name == name:
            return v
    return None
