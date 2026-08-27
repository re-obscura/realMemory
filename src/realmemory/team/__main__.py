"""CLI командного слоя: python -m realmemory.team {ui,status,policy}"""
from __future__ import annotations

import argparse
import json
import sys


def _open_store(path: str):
    from pathlib import Path

    from .tui import _store_for

    return _store_for(Path(path))


def cmd_status(args) -> int:
    from . import registry as reg
    from .identity import resolve_identity
    from .policy import load_policy
    from .view import projects_view

    policy = load_policy(args.policy_path)
    if not policy.identity:
        policy.identity = resolve_identity()
    store = _open_store(args.path)
    try:
        views = projects_view(store, policy)
        payload = {
            "identity": policy.identity,
            "coordinator": policy.coordinator,
            "projects": [
                {"name": v.name, "shareable": policy.is_shareable_project(v.name),
                 "active": v.total, "eligible": v.eligible,
                 "blocked_never": v.blocked, "published": v.published}
                for v in views
            ],
            "registry": reg.sync_status(store),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        who = policy.identity or "(безлично)"
        coord = policy.coordinator or "(не настроен — оффлайн-режим)"
        print(f"identity: {who} · coordinator: {coord}")
        for v in views:
            mark = "✓" if policy.is_shareable_project(v.name) else "·"
            print(f" {mark} {v.name:<24} активных {v.total:>4}  кандидатов "
                  f"{v.eligible:>3}  never-block {v.blocked:>2}  в команде {v.published}")
        print("registry:", reg.stats_line(store))
        return 0
    finally:
        store.close()


def cmd_policy(args) -> int:
    from .identity import resolve_identity
    from .policy import DEFAULT_POLICY_PATH, load_policy, save_policy

    path = args.policy_path or DEFAULT_POLICY_PATH
    if args.action == "path":
        print(str(path))
        return 0
    policy = load_policy(path)
    print(f"# {path}")
    print(f"identity: {policy.identity or resolve_identity()}")
    print(f"coordinator: {policy.coordinator or '(нет)'}")
    print("projects:")
    if not policy.projects:
        print("  (ни один проект не включён)")
    for pr in policy.projects:
        extra = []
        if pr.kinds:
            extra.append(f"kinds={pr.kinds}")
        if pr.min_reinforcements is not None:
            extra.append(f"min_rep={pr.min_reinforcements}")
        print(f"  - {pr.name}" + (" · " + ", ".join(extra) if extra else ""))
    print(f"never meta_tags={policy.never_meta_tags}")
    print(f"never patterns={len(policy.never_text_patterns)} шт.")
    if args.action == "init":
        save_policy(policy, path)
        print("(файл записан)")
    return 0


def cmd_ui(args) -> int:  # pragma: no cover - интерактивный режим
    if not sys.stdin.isatty():
        print("[realmemory] ui требует терминал; используй status", file=sys.stderr)
        return 1
    from .tui import run_app

    run_app(args.path, policy_path=args.policy_path)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="realmemory-team",
                                     description="командный слой realMemory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p, need_path=True):
        p.add_argument("--path", required=need_path, help="каталог базы памяти")
        p.add_argument("--policy-path", default=None,
                       help="путь к team.yaml (по умолчанию ~/.realmemory/team.yaml)")

    u = sub.add_parser("ui", help="интерактивный выбор публикаций (Textual)")
    add_common(u)
    u.set_defaults(fn=cmd_ui)

    s = sub.add_parser("status", help="сводка по проектам и registry")
    add_common(s)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    pl = sub.add_parser("policy", help="показать путь/содержимое политики")
    pl.add_argument("action", nargs="?", choices=["show", "path", "init"],
                    default="show")
    pl.add_argument("--policy-path", default=None)
    pl.set_defaults(fn=cmd_policy)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
