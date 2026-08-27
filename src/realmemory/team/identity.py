"""Идентичность автора для атрибуции записей командного слоя.

Порядок разрешения: явное значение → переменная окружения REALMEMORY_IDENTITY
→ файл ~/.realmemory/identity (первая непустая строка) → git config user.name.
Настройка одноразовая, дальше значение подставляется прозрачно в каждую запись.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

IDENTITY_FILE = Path.home() / ".realmemory" / "identity"


def _read_identity_file(path: Path | None = None) -> str:
    p = Path(path) if path else IDENTITY_FILE
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:64]
    except OSError:
        pass
    return ""


@lru_cache(maxsize=1)
def _git_user_name() -> str:
    try:
        out = subprocess.run(
            ["git", "config", "user.name"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()[:64] if out.returncode == 0 else ""


def resolve_identity(explicit: str | None = None) -> str:
    """Разрешить имя автора; пустая строка означает «безлично» (тесты/хуки)."""
    for candidate in (
        explicit,
        os.environ.get("REALMEMORY_IDENTITY", "").strip(),
        _read_identity_file(),
        _git_user_name(),
    ):
        if candidate:
            return str(candidate).strip()[:64]
    return ""
