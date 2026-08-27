"""Политика шеринга: что МОЖНО покинуть машину, а что — никогда.

Двухуровневая модель:
  shareable-правила (projects/kinds/min_reinforcements) задают класс
    публикуемого; переключатель проекта в TUI/CLI правит именно их;
  never-правила (meta-теги + regex по тексту) работают fail-closed и сильнее
    всего остального: публикация кандидата с попаданием отклоняется всегда.

Хранение: ~/.realmemory/team.yaml. Неизвестные пользователю ключи YAML
сохраняются при перезаписи без потерь (raw-словарь).
"""
from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_POLICY_PATH = Path.home() / ".realmemory" / "team.yaml"

# статусы решения classify(); наружу отдаём как есть (стабильный контракт)
ELIGIBLE = "eligible"
BLOCKED_NEVER = "blocked-never"
WRONG_KIND = "not-shareable-kind"
LOW_REINFORCEMENTS = "low-reinforcements"
PROJECT_OFF = "project-not-shareable"


@dataclass
class ProjectRule:
    name: str
    kinds: list[str] | None = None           # None → унаследовать глобальные
    min_reinforcements: int | None = None    # None → унаследовать глобальные


@dataclass
class TeamPolicy:
    identity: str = ""
    coordinator: str | None = None            # этап сети; сейчас только бейдж
    token_env: str = "REALMEMORY_TEAM_TOKEN"
    default_kinds: list[str] = field(default_factory=lambda: ["semantic"])
    default_min_reinforcements: int = 2
    projects: list[ProjectRule] = field(default_factory=list)
    never_meta_tags: list[str] = field(
        default_factory=lambda: ["private", "secret", "credentials"])
    never_text_patterns: list[str] = field(default_factory=lambda: [
        r"(?i)\bapi[_-]?key\b\s*=", r"(?i)\bpassword\s*[:=]", r"(?i)секретн",
        r"(?i)\bпароль\s*[:=]",
    ])
    raw: dict[str, Any] = field(default_factory=dict, repr=False,
                                compare=False)

    def project_rule(self, name: str) -> ProjectRule | None:
        for p in self.projects:
            if p.name == name:
                return p
        return None

    def is_shareable_project(self, name: str) -> bool:
        return self.project_rule(name) is not None

    def kinds_for(self, rule: ProjectRule | None) -> list[str]:
        return (rule.kinds if rule and rule.kinds else self.default_kinds)

    def min_reinforcements_for(self, rule: ProjectRule | None) -> int:
        return (rule.min_reinforcements if rule and rule.min_reinforcements
                else self.default_min_reinforcements)


@dataclass(frozen=True)
class ShareDecision:
    trace_id: int
    status: str            # ELIGIBLE / BLOCKED_NEVER / WRONG_KIND / ...
    reason: str = ""


def load_policy(path: Path | None = None) -> TeamPolicy:
    import yaml  # зависимость extra [team]; ядро её не тянет

    p = Path(path) if path else DEFAULT_POLICY_PATH
    policy = TeamPolicy()
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except OSError:
        return policy
    if not isinstance(raw, dict):
        return policy
    policy.raw = copy.deepcopy(raw)
    policy.identity = str(raw.get("identity", "") or "")
    policy.coordinator = raw.get("coordinator") or None
    policy.token_env = str(raw.get("token_env") or policy.token_env)
    policy.default_kinds = list(raw.get("default_kinds") or ["semantic"])
    policy.default_min_reinforcements = int(raw.get("min_reinforcements", 2))
    for item in raw.get("projects") or []:
        if isinstance(item, str):                      # короткая форма "- name"
            policy.projects.append(ProjectRule(name=item))
        elif isinstance(item, dict) and item.get("name"):
            policy.projects.append(ProjectRule(
                name=str(item["name"]),
                kinds=list(item["kinds"]) if item.get("kinds") else None,
                min_reinforcements=(int(item["min_reinforcements"])
                                    if item.get("min_reinforcements") is not None
                                    else None),
            ))
    ne = raw.get("never") or {}
    if isinstance(ne, dict):
        if ne.get("meta_tags"):
            policy.never_meta_tags = [str(t) for t in ne["meta_tags"]]
        if ne.get("text_patterns"):
            policy.never_text_patterns = [str(x) for x in ne["text_patterns"]]
    return policy


def save_policy(policy: TeamPolicy, path: Path | None = None) -> Path:
    """Записать политику, сохранив неизвестные ключи исходного файла."""
    import yaml

    p = Path(path) if path else DEFAULT_POLICY_PATH
    raw = copy.deepcopy(policy.raw or {})
    raw["identity"] = policy.identity
    if policy.coordinator:
        raw["coordinator"] = policy.coordinator
    else:
        raw.pop("coordinator", None)
    raw["token_env"] = policy.token_env
    raw.setdefault("default_kinds", policy.default_kinds)
    raw["default_kinds"] = policy.default_kinds
    raw["min_reinforcements"] = policy.default_min_reinforcements
    raw["projects"] = [
        {"name": pr.name, **({"kinds": pr.kinds} if pr.kinds else {}),
         **({"min_reinforcements": pr.min_reinforcements}
            if pr.min_reinforcements is not None else {})}
        for pr in policy.projects
    ]
    raw["never"] = {"meta_tags": policy.never_meta_tags,
                    "text_patterns": policy.never_text_patterns}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return p


def set_project_shareable(policy: TeamPolicy, name: str, on: bool,
                          kinds: list[str] | None = None,
                          min_reinforcements: int | None = None) -> bool:
    """Включить/выключить проект в shareable; возвращает True при изменении."""
    before = policy.is_shareable_project(name)
    if on == before:
        return False
    if on:
        policy.projects.append(ProjectRule(name=name, kinds=kinds,
                                           min_reinforcements=min_reinforcements))
    else:
        policy.projects = [p for p in policy.projects if p.name != name]
    return True


def _trace_tags(rec_meta: dict[str, Any]) -> set[str]:
    tags = rec_meta.get("tags")
    if isinstance(tags, (list, tuple, set)):
        return {str(t).lower() for t in tags}
    return set()


def classify(rec, policy: TeamPolicy) -> ShareDecision:
    """Решение о допустимости публикации конкретного следа.

    Порядок проверок принципиален: never-правила применяются первыми и не
    переопределяются ничем (fail-closed), затем eligibility-требования.
    """
    tid = int(rec.id or 0)
    text_re = [re.compile(pat) for pat in policy.never_text_patterns]
    lowered_tags = {t.lower() for t in policy.never_meta_tags}
    hit_tags = _trace_tags(rec.meta) & lowered_tags
    if hit_tags:
        return ShareDecision(tid, BLOCKED_NEVER,
                             f"meta-тег(и): {', '.join(sorted(hit_tags))}")
    for rex in text_re:
        if rex.search(rec.text or ""):
            return ShareDecision(tid, BLOCKED_NEVER,
                                 f"текст под never-паттерн {rex.pattern!r}")

    rule = policy.project_rule(rec.scope)
    if rule is None:
        return ShareDecision(tid, PROJECT_OFF,
                             f"проект «{rec.scope}» не включён в shareable")

    if rec.status != "active":
        return ShareDecision(tid, WRONG_KIND, "след не активен")

    if rec.kind not in policy.kinds_for(rule):
        return ShareDecision(tid, WRONG_KIND,
                             f"kind={rec.kind} не входит в {policy.kinds_for(rule)}")

    need = policy.min_reinforcements_for(rule)
    if int(rec.reinforced_count) < need:
        return ShareDecision(tid, LOW_REINFORCEMENTS,
                             f"подкреплений {rec.reinforced_count} < {need}")
    return ShareDecision(tid, ELIGIBLE)


def compile_never_patterns(policy: TeamPolicy) -> list[re.Pattern]:
    return [re.compile(x) for x in policy.never_text_patterns]


def describe() -> str:  # pragma: no cover - справочная строка для CLI
    fields = ", ".join(f.name for f in dataclasses.fields(TeamPolicy))
    return f"TeamPolicy({fields})"
