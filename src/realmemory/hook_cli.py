"""Хуки автоматизации realMemory: брифинг на старте сессии, «сон» в конце.

python -m realmemory.hook_cli brief  --path ./rm_data [--top N] [--plain]
    Печатает JSON SessionStart-хука (additionalContext) с кратким состоянием
    памяти: объёмы, ключевые семантические факты. Модель эмбеддера НЕ грузится.
python -m realmemory.hook_cli sleep --path ./rm_data
    Консолидация («сон»): коммит связей, распад, метрики в журнал. Троттлинг
    по состоянию базы: пропускает, если с прошлого сна не было ни записей,
    ни feedback, ни незакоммиченных bind'ов. Вывод пустой, код возврата 0.

Обе команды безопасно сосуществуют с работающим MCP-сервером: всё состояние
в SQLite, консолидации сериализуются BEGIN IMMEDIATE.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path


def _infer_dim(path: Path) -> int:
    """Размерность эмбеддингов из существующей базы (без загрузки модели)."""
    db = path / "memory.db"
    if not db.exists():
        return 256
    con = sqlite3.connect(str(db))
    try:
        row = con.execute("SELECT length(embedding) FROM memories LIMIT 1").fetchone()
    finally:
        con.close()
    return int(row[0]) // 4 if row else 256


def _load_cfg(path: Path):
    """Конфиг из базы (записывается при первом открытии); None если нет/битый."""
    db = path / "memory.db"
    if not db.exists():
        return None
    con = sqlite3.connect(str(db))
    try:
        row = con.execute("SELECT value FROM db_meta WHERE key='config'").fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        con.close()
    if not row:
        return None
    from .config import MemoryConfig

    try:
        return MemoryConfig.from_snapshot(json.loads(row[0]))
    except Exception:  # noqa: BLE001 - битый конфиг не должен ломать хук
        return None


def _open(path: str, namespace: str | None = None):
    from . import Hippocampus, MemoryConfig
    from .encoding.embedder import HashingEmbedder
    from .timeprov import SystemClock

    p = _resolve_hook_root(Path(path), namespace)
    cfg = _load_cfg(p) or MemoryConfig(dim=_infer_dim(p))
    return Hippocampus.open(
        p, config=cfg, embedder=HashingEmbedder(dim=cfg.dim), clock=SystemClock(),
        verify_embedder=False,  # хук не порождает векторов; боевой эмбеддер базы не трогаем
    )


def _resolve_hook_root(root: Path, namespace: str | None) -> Path:
    """Корень хранилища с учётом namespace; хуки открывают его напрямую."""
    return root / namespace if namespace else root


_BRIEF_BUDGET_CHARS = 600  # бюджет фактов в additionalContext, кроме шапки


def cmd_brief(args) -> int:
    import math

    from .policies.decay import retention
    from .projects import resolve_project

    hippo = _open(args.path, getattr(args, "namespace", None))
    try:
        now = float(hippo.clock.now())
        project = resolve_project(getattr(args, "project", None))

        def in_scope(rec_scope: str) -> bool:
            return project is None or rec_scope == project or rec_scope == "global"

        semantic: list[tuple[float, int, str]] = []
        episodic: list[tuple[float, float, str]] = []
        for rec in hippo.store.iter_active():
            if not in_scope(rec.scope):
                continue
            ret = retention(rec.base_strength, rec.last_reinforced_at, now, rec.kind, hippo.config)
            if rec.kind == "semantic":
                semantic.append((ret, rec.reinforced_count, rec.text))
            elif rec.kind == "episodic":
                # прочность = живучесть × подкрепления: устойчивые решения
                # проекта показываем раньше одноразовых заметок
                score = ret * (1.0 + math.log1p(rec.reinforced_count))
                episodic.append((score, ret, rec.text))
        semantic.sort(key=lambda t: (-t[0], -t[1]))
        episodic.sort(key=lambda t: (-t[0], -t[1]))

        header = (
            f"[realMemory] persistent memory online: "
            f"{hippo.store.count(status='active')} traces ({len(semantic)} semantic), "
            f"{hippo.network.edge_count} associations"
        )
        if project:
            header += f"; project scope: {project}"
        lines = [
            header +
            ". Retrieve context with `recall` before asserting facts; record "
            "decisions and preferences with `memorize`; grade usefulness with "
            "`reflect`."
        ]
        used = 0
        shown: set[str] = set()
        for _, _, text in semantic[: args.top]:
            line = f"• {text}"
            if used + len(line) > _BRIEF_BUDGET_CHARS:
                break
            lines.append(line)
            shown.add(text)
            used += len(line)
        added_epi = 0
        for _, _, text in episodic:
            if added_epi >= args.episodic_top or used >= _BRIEF_BUDGET_CHARS:
                break
            if text in shown:
                continue
            line = f"· {text}"
            if used + len(line) > _BRIEF_BUDGET_CHARS:
                break
            lines.append(line)
            shown.add(text)
            used += len(line)
            added_epi += 1
        ctx = "\n".join(lines)
        if args.plain:
            print(ctx)
            return 0
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": ctx,
            }
        }, ensure_ascii=False))
        return 0
    finally:
        hippo.close()


def _sleep_needed(db: Path) -> bool:
    """Сон нужен, если с прошлой консолидации что-то изменилось: записи/feedback
    (updated_at), новые eligibility-события. Нет прошлого сна — нужен."""
    if not db.exists():
        return True
    con = sqlite3.connect(str(db))
    try:
        row = con.execute(
            "SELECT value FROM db_meta WHERE key='last_consolidate_at'"
        ).fetchone()
        maxu = con.execute("SELECT MAX(updated_at) FROM memories").fetchone()[0]
        pend = con.execute("SELECT COUNT(*) FROM eligibility").fetchone()[0]
    except sqlite3.DatabaseError:
        return True
    finally:
        con.close()
    if row is None:
        return True
    last = float(row[0])
    newest = float(maxu) if maxu is not None else 0.0
    return newest > last or pend > 0


def cmd_sleep(args) -> int:
    root = _resolve_hook_root(Path(args.path), getattr(args, "namespace", None))
    if not _sleep_needed(root / "memory.db"):
        return 0
    hippo = _open(args.path, getattr(args, "namespace", None))
    try:
        report = hippo.consolidate()
        if args.verbose:
            print(json.dumps({**vars(report)}, ensure_ascii=False))
        _maybe_team_auto_sync(hippo)
    finally:
        hippo.close()
    return 0


def _maybe_team_auto_sync(hippo) -> None:
    """Опциональный авто-sync командного слоя после сна. Никогда не ломает
    сон: любая ошибка — строка в stderr, статус синхронизации доедет позже."""
    try:
        from pathlib import Path as _Path

        from .team.policy import load_policy
        from .team.sync import push

        env_path = os.environ.get("REALMEMORY_POLICY_PATH")
        policy = load_policy(_Path(env_path) if env_path else None)
        if not (policy.coordinator and policy.auto_sync):
            return
        summary = push(hippo.store, policy)
        if summary.published or summary.retracted or summary.auto_retracted:
            # Stop-хук: stdout может попадать в контекст сессии — только stderr
            print(f"[realmemory] team auto-sync: +{summary.published} "
                  f"отзывов {summary.retracted + summary.auto_retracted}",
                  file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - сон важнее командного слоя
        print(f"[realmemory] team auto-sync skipped: {exc}", file=sys.stderr)


def _report_hook_error(args, exc: BaseException) -> None:
    """Оставить след об упавшем хуке в журнале событий (best effort)."""
    try:
        root = _resolve_hook_root(Path(args.path), getattr(args, "namespace", None))
        con = sqlite3.connect(str(root / "memory.db"), timeout=2)
        try:
            con.execute(
                "INSERT INTO events(ts, type, data) VALUES(?,?,?)",
                (
                    time.time(),
                    "hook_error",
                    json.dumps({"cmd": args.cmd, "error": str(exc)},
                               ensure_ascii=False),
                ),
            )
            con.commit()
        finally:
            con.close()
    except Exception:  # noqa: BLE001, S110 - сообщать об ошибке сообщения об ошибке поздно
        pass


def _force_utf8_streams() -> None:
    """Хуки говорят с клиентом по пайпам: на Windows кодировка пайпа — ANSI
    (cp1252 и т.п.), и кириллица в брифе/ошибке падает UnicodeEncodeError.
    Контракт вывода — UTF-8, так что выставляем его принудительно."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass  # не-TextIO обёртки (capture, редкие пайпы) — оставляем как есть


def main(argv=None) -> None:
    _force_utf8_streams()
    parser = argparse.ArgumentParser(prog="realmemory-hooks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("brief", help="SessionStart: JSON additionalContext")
    b.add_argument("--path", required=True)
    b.add_argument("--namespace", default=None, help="подкаталог внутри --path")
    b.add_argument("--project", default=None,
                   help="скоуп проекта; по умолчанию определяется автоматически")
    b.add_argument("--top", type=int, default=7, help="максимум семантических фактов")
    b.add_argument("--episodic-top", type=int, default=5,
                   help="максимум прочных эпизодических фактов после семантики")
    b.add_argument("--plain", action="store_true", help="человекочитаемый текст вместо JSON")
    b.set_defaults(fn=cmd_brief)

    s = sub.add_parser("sleep", help="Stop: консолидация с троттлингом по состоянию базы")
    s.add_argument("--path", required=True)
    s.add_argument("--namespace", default=None, help="подкаталог внутри --path")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_sleep)

    args = parser.parse_args(argv)
    try:
        code = args.fn(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - хук не должен ломать сессию агента
        # тишина при отказе хука недопустима для эксплуатации: ошибка видна в
        # stderr сессии и остаётся в журнале событий для отчёта
        _report_hook_error(args, exc)
        print(f"[realmemory] hook {args.cmd} failed: {exc}", file=sys.stderr)
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
