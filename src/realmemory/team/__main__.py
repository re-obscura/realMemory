"""CLI командного слоя: python -m realmemory.team {ui,status,policy}"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _open_store(path: str):
    from pathlib import Path

    from .tui import _store_for

    return _store_for(Path(path))


def _policy_or_die(args):
    from .identity import resolve_identity
    from .policy import load_policy

    policy = load_policy(args.policy_path)
    if not policy.identity:
        policy.identity = resolve_identity()
    return policy


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


def cmd_sync(args) -> int:
    from .sync import push

    policy = _policy_or_die(args)
    store = _open_store(args.path)
    try:
        summary = push(store, policy, timeout_s=args.timeout)
    except Exception as exc:  # noqa: BLE001 - человекочитаемая причина
        print(f"[realmemory] sync failed: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    line = (f"опубликовано {summary.published}, "
            f"отзывов доставлено {summary.retracted}")
    if summary.marked != summary.published + summary.retracted:
        line += f" (помечено {summary.marked})"
    if summary.auto_retracted:
        line += f"; авто-отзыв забытых локально: {summary.auto_retracted}"
    print(line)
    return 0


def cmd_recall_team(args) -> int:
    from .recall_team import recall_team

    try:
        answer = recall_team(args.path, args.query, k=args.k,
                             author=args.author or None,
                             project=args.project or None,
                             policy_path=args.policy_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[realmemory] recall_team failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({
            "abstained": answer.abstained,
            "max_age_s": answer.max_age_s,
            "coordinator": answer.coordinator,
            "coordinator_error": answer.coordinator_error,
            "online": answer.presence_online,
            "hits": [vars(h) for h in answer.hits],
        }, ensure_ascii=False, indent=2))
        return 0
    who = ", ".join(answer.presence_online) or "(никого online)"
    age = (f"возраст выдачи до {answer.max_age_s / 3600:.1f} ч"
           if answer.max_age_s is not None else "живые данные")
    live = ", ".join(answer.peers_live) or "нет"
    failed = "; ".join(answer.peers_failed)
    print(f"coordinator {answer.coordinator} · online: {who} · "
          f"live: {live} · {age}")
    if answer.coordinator_error:
        print(f"координатор недоступен: {answer.coordinator_error}")
    if failed:
        print(f"peer недоступен: {failed}")
    if answer.abstained:
        print("в командной памяти по теме ничего нет")
        return 0
    for h in answer.hits:
        mark = "LIVE" if h.source == "live" else "кэш"
        print(f"[{h.score:.3f}] ({h.author}/{h.project}, {mark}) {h.text}")
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

    sy = sub.add_parser("sync",
                        help="довести публикации/отзывы до координатора")
    add_common(sy)
    sy.add_argument("--timeout", type=float, default=4.0)
    sy.set_defaults(fn=cmd_sync)

    rt = sub.add_parser("recall-team", help="поиск по кэшу команды")
    add_common(rt)
    rt.add_argument("query")
    rt.add_argument("--k", type=int, default=5)
    rt.add_argument("--author", default=None)
    rt.add_argument("--project", default=None)
    rt.add_argument("--json", action="store_true")
    rt.set_defaults(fn=cmd_recall_team)

    sv = sub.add_parser("serve",
                        help="живой peer-endpoint + presence-хартбиты")
    sv.add_argument("--path", required=True)
    sv.add_argument("--host", default="127.0.0.1",
                    help="адрес привязки (для LAN: 0.0.0.0)")
    sv.add_argument("--port", type=int, default=8410)
    sv.add_argument("--advertise", default=None,
                    help="адрес для presence вместо автоопределённого LAN IP")
    sv.add_argument("--policy-path", default=None)
    sv.set_defaults(fn=cmd_serve)

    args = parser.parse_args(argv)
    return args.fn(args)


def cmd_serve(args) -> int:  # pragma: no cover - долгоживущий демон
    """Живой peer-endpoint + периодический presence-хартбит."""
    import threading

    from .identity import resolve_identity
    from .peer import lan_ip, make_peer_server, start_heartbeat
    from .policy import load_policy

    policy = load_policy(args.policy_path)
    if not policy.identity:
        policy.identity = resolve_identity()
    token = os.environ.get(policy.token_env or "", "").strip() or None
    # binding на 0.0.0.0 слушает все интерфейсы, но в presence рекламировать
    # его нельзя: коллега уйдёт connect'иться на свой же localhost
    advertised = getattr(args, "advertise", None) or (
        lan_ip() if args.host in ("0.0.0.0", "::") else args.host)
    address = f"{advertised}:{args.port}"
    srv = make_peer_server(args.path, host=args.host, port=args.port,
                           token=token)
    stop = threading.Event()
    if policy.coordinator:
        start_heartbeat(policy, address, stop)
        print(f"[realmemory] peer {address} · presence → {policy.coordinator}"
              f" (identity: {policy.identity})", flush=True)
    else:
        print("[realmemory] peer без координатора: presence отключена",
              flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
