"""Принудительный UTF-8 для stdout/stderr: урок cp1252-пайпов Windows.

Хуки и CLI говорят с клиентом по пайпам: на Windows кодировка пайпа — ANSI
(cp1252 и т.п.), и кириллица в выводе падает UnicodeEncodeError. Контракт
вывода — UTF-8, выставляем его принудительно. Живёт здесь, а не в hook_cli,
потому что нужен и отчёту, и team-CLI (граница hook_cli не для этого).
"""
from __future__ import annotations

import sys


def force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # не-TextIO обёртки (capture, редкие пайпы) — как есть
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
