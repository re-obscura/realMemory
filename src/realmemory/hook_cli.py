"""Хуки автоматизации realMemory: брифинг на старте сессии, «сон» в конце.

python -m realmemory.hook_cli brief  --path ./rm_data [--top N] [--plain]
    Печатает JSON SessionStart-хука (additionalContext) с кратким состоянием
    памяти: объёмы, ключевые семантические факты. Модель эмбеддера НЕ грузится.
python -m realmemory.hook_cli sleep --path ./rm_data [--min-interval-s S]
    Консолидация («сон»): коммит связей, распад, метрики в журнал, снапшот.
    Троттлинг: пропускает, если с прошлого сна ничего не менялось и не прошло
    min-interval-s (по умолчанию 30 минут). Вывод пустой, код возврата 0.
"""
from __future__ import annotations

import argparse
import json
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


def _open(path: str, namespace: str | None = None):
    from . import Hippocampus, MemoryConfig
    from .encoding.embedder import HashingEmbedder
    from .timeprov import SystemClock

    p = _resolve_hook_root(Path(path), namespace)
    cfg = None
    snap = p / "snapshot.pkl"
    if snap.exists():
        try:
            with snap.open("rb") as f:
                import pickle

                payload = pickle.load(f)
            cfg = MemoryConfig.from_snapshot(payload["config"])
        except Exception:  # noqa: BLE001 - битый снапшот не должен ломать хук
            cfg = None
    if cfg is None:
        cfg = MemoryConfig(dim=_infer_dim(p))
    return Hippocampus.open(
        p, config=cfg, embedder=HashingEmbedder(dim=cfg.dim), clock=SystemClock(),
        verify_embedder=False,  # хук не порождает векторов; боевой эмбеддер базы не трогаем
    )


def _resolve_hook_root(root: Path, namespace: str | None) -> Path:
    """Корень хранилища с учётом namespace; хуки открывают его напрямую."""
    return root / namespace if namespace else root


def cmd_brief(args) -> int:
    hippo = _open(args.path, getattr(args, "namespace", None))
    try:
        now = float(hippo.clock.now())
        from .policies.decay import retention

        semantic = []
        for rec in hippo.store.iter_active():
            if rec.kind != "semantic":
                continue
            ret = retention(rec.base_strength, rec.last_reinforced_at, now, rec.kind, hippo.config)
            semantic.append((ret, rec.reinforced_count, rec.text))
        semantic.sort(key=lambda t: (-t[0], -t[1]))
        lines = [
            (
                f"[realMemory] persistent memory online: "
                f"{hippo.store.count(status='active')} traces ({len(semantic)} semantic), "
                f"{hippo.network.edge_count} associations. Retrieve context with "
                "`recall` before asserting facts; record decisions and preferences "
                "with `memorize`; grade usefulness with `reflect`."
            )
        ]
        for _, _, text in semantic[: args.top]:
            lines.append(f"• {text}")
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


def cmd_sleep(args) -> int:
    root = _resolve_hook_root(Path(args.path), getattr(args, "namespace", None))
    snap = root / "snapshot.pkl"
    db = root / "memory.db"
    # троттлинг: недавно спали и с тех пор ничего не писали
    recently_slept = snap.exists() and (time.time() - snap.stat().st_mtime) < args.min_interval_s
    nothing_new = snap.exists() and db.exists() and snap.stat().st_mtime >= db.stat().st_mtime
    if recently_slept and nothing_new:
        return 0
    hippo = _open(args.path, getattr(args, "namespace", None))
    try:
        report = hippo.consolidate(save=True)
        if args.verbose:
            print(json.dumps({**vars(report)}, ensure_ascii=False))
    finally:
        hippo.close()
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="realmemory-hooks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("brief", help="SessionStart: JSON additionalContext")
    b.add_argument("--path", required=True)
    b.add_argument("--namespace", default=None, help="подкаталог внутри --path")
    b.add_argument("--top", type=int, default=7)
    b.add_argument("--plain", action="store_true", help="человекочитаемый текст вместо JSON")
    b.set_defaults(fn=cmd_brief)

    s = sub.add_parser("sleep", help="Stop: консолидация с троттлингом")
    s.add_argument("--path", required=True)
    s.add_argument("--namespace", default=None, help="подкаталог внутри --path")
    s.add_argument("--min-interval-s", type=float, default=1800.0)
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=cmd_sleep)

    args = parser.parse_args(argv)
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
