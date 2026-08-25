"""Разрешение имени проекта для скоупов памяти.

Иерархия источников (первый непустой выигрывает):
  1. явный параметр тула/CLI (--project);
  2. переменная окружения REALMEMORY_PROJECT (готовый слаг);
  3. переменные ZCODE_PROJECT_DIR / CLAUDE_PROJECT_DIR (путь к проекту,
     инжектируются хуками ZCode автоматически);
  4. текущая рабочая директория, если в ней есть маркер репозитория
     (.git или .zcode) — типичный случай запуска MCP-сервера в корне проекта.

None означает «контекст не определён»: recall ищет без фильтра по скоупам,
memorize пишет в global.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")

_REPO_MARKERS = (".git", ".zcode")


def normalize_slug(name: str | None) -> str | None:
    """Валидировать слаг проекта; None/невалидное имя -> None."""
    if not name:
        return None
    name = name.strip()
    return name if SLUG_RE.fullmatch(name) else None


def _slug_from_dir(dirpath: str | None) -> str | None:
    if not dirpath:
        return None
    return normalize_slug(Path(dirpath).name)


def resolve_project(explicit: str | None = None) -> str | None:
    """Текущий проект по иерархии источников; см. докстринг модуля.

    Невалидный explicit — ошибка: вызывающий явно указал имя и опечатка
    должна быть видна сразу, а не молча уходить в global.
    """
    if explicit is not None:
        slug = normalize_slug(explicit)
        if slug is None:
            raise ValueError(
                "project: 1–64 символа из [A-Za-z0-9_.-], первый — буква или цифра"
            )
        return slug
    slug = normalize_slug(os.environ.get("REALMEMORY_PROJECT"))
    if slug:
        return slug
    slug = _slug_from_dir(
        os.environ.get("ZCODE_PROJECT_DIR") or os.environ.get("CLAUDE_PROJECT_DIR")
    )
    if slug:
        return slug
    cwd = Path.cwd()
    if any((cwd / marker).exists() for marker in _REPO_MARKERS):
        return normalize_slug(cwd.name)
    return None
