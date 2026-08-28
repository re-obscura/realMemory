"""MCP-сервер realMemory — постоянная память («гиппокамп») для агента.

Тулы названы как естественные когнитивные действия: recall, memorize,
reflect, revise, introspect, dream_log. Опциональная зависимость:
pip install 'realmemory[mcp]'. Модуль импортируется без установленного mcp;
build_server() даёт понятный ImportError.
Запуск: python -m realmemory.api.mcp_server --path ./rm_data [--namespace ns]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _packet_to_dict(packet) -> dict[str, Any]:
        return {
            "query": packet.query,
            "abstained": packet.abstained,
            "latency_ms": round(packet.latency_ms, 2),
            "items": [
                {
                    "id": it.memory_id,
                    "text": it.text,
                    "kind": it.kind,
                    "cosine": it.cosine,
                    "confidence": it.confidence,
                    "retention": it.retention,
                    "source": it.source,
                    "scope": it.scope,
                    "author": it.author,
                    "created_at": it.created_at,
                    "meta": dict(it.meta),
                }
                for it in packet.items
            ],
        }


def build_server(hippo, default_project: str | None = None,
                 team_root=None, team_policy_path=None):
    """Собрать FastMCP-сервер над фасадом. Требует пакет fastmcp (extra [mcp]).

    default_project — скоуп по умолчанию (обычно определён по рабочей директории);
    явный аргумент project тула имеет приоритет. team_root/team_policy_path:
    если политика командного слоя настроена (coordinator задан), добавляется
    тула recall_team; без настройки тул не существует — kill-switch."""
    team_enabled = False
    if team_root is not None:
        try:
            from ..team.policy import load_policy

            _policy = load_policy(team_policy_path)
            team_enabled = bool(_policy.coordinator)
        except Exception:  # noqa: BLE001 - отсутствие команды ≠ сломанный сервер
            team_enabled = False
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "MCP extra не установлен. Установите: pip install 'realmemory[mcp]'"
        ) from exc

    mcp = FastMCP("realmemory")

    def _effective(project: str | None) -> str | None:
        return project or default_project

    @mcp.tool()
    def recall(query: str, k: int = 5, include_superseded: bool = False,
               project: str | None = None) -> str:
        """Recollect long-term memories relevant to the query.

        Use before making claims about the user, their projects or previous
        sessions. Returns JSON: items [{id, text, kind, cosine, confidence,
        retention, source, scope, created_at, meta}] ranked by confidence, plus
        an abstained flag. abstained=true means nothing trustworthy was
        remembered — say so honestly instead of guessing. Items with
        source=associated arrived via associative links and may be less precise;
        present confidence < 0.2 as "possibly", not as fact.

        Scope: searches the current project plus global memory; pass an explicit
        project name to search another project instead.
        """
        packet = hippo.recall(query, k=k, include_superseded=include_superseded,
                              scope=_effective(project))
        return json.dumps(_packet_to_dict(packet), ensure_ascii=False)

    @mcp.tool()
    def memorize(text: str, kind: str = "episodic", related_ids: list[int] | None = None,
                 project: str | None = None) -> str:
        """Commit one durable fact, decision or preference to long-term memory.

        Write conclusions rather than conversation snippets ("we chose SQLite WAL",
        "user prefers concise answers"). The novelty gate decides automatically:
        an already-known fact gets reinforced, a related one links to it, a fresh
        one allocates a new trace — reformulations don't pile up. Pass related_ids
        when this builds on specific earlier memories.

        Scope rules: facts about THIS workspace (decisions, gotchas, stack
        choices) — pass project=<workspace name> (default is detected
        automatically); user-global preferences and identity ("отвечай кратко",
        preferred languages) — omit project. Returns {memory_id, action, created}.
        """
        scope = _effective(project) or "global"
        res = hippo.remember(
            text, kind=kind, related_ids=tuple(related_ids or ()), scope=scope,
        )
        return json.dumps(
            {
                "memory_id": res.memory_id,
                "created": res.created,
                "action": res.decision.action.value,
                "novelty": round(res.decision.novelty, 4),
                "scope": scope,
            },
            ensure_ascii=False,
        )

    @mcp.tool()
    def reflect(memory_ids: list[int], reward: float) -> str:
        """Grade recently recalled memories by how useful they proved.

        Call shortly after acting on recalled items. reward ∈ [-1, 1]:
        positive (+0.3..+1.0) strengthens traces that helped and their fresh
        associations; negative (-0.3..-1.0) weakens misleading, harmful or
        outdated ones toward forgetting. This is the only channel through which
        memory learns to be useful. Returns {touched: n}.
        """
        n = hippo.feedback(ids=memory_ids, reward=reward)
        return json.dumps({"touched": n}, ensure_ascii=False)

    @mcp.tool()
    def revise(old_id: int, new_text: str) -> str:
        """Correct a memory whose reality has changed ("we now use X instead of Y").

        The outdated trace is kept as linked history and excluded from future
        recalls; nothing is silently lost. The replacement inherits the old
        trace's project scope. Returns {old_id, new_id}.
        """
        res = hippo.update_fact(int(old_id), new_text)
        return json.dumps({"old_id": int(old_id), "new_id": res.memory_id},
                          ensure_ascii=False)

    if team_enabled:
        @mcp.tool()
        def recall_team(query: str, k: int = 5, author: str | None = None,
                        project: str | None = None) -> str:
            """Search the TEAM memory cache (published by colleagues).

            Use when the user explicitly asks what the team knows ("как это
            решал Андрей", "что известно команде про X"). Data may be stale:
            every hit carries author/project/published_at - always mention
            the freshness ("опубликовано N часов/дней назад"). abstained=true
            means nothing published on the topic - say so honestly.
            """
            import json as _json
            from pathlib import Path as _Path

            from ..team.recall_team import recall_team as _recall

            answer = _recall(_Path(team_root), query, k=k, author=author,
                             project=project or default_project,
                             policy_path=team_policy_path)
            return _json.dumps({
                "abstained": answer.abstained,
                "max_age_s": answer.max_age_s,
                "online": answer.presence_online,
                "coordinator_error": answer.coordinator_error,
                "hits": [vars(h) for h in answer.hits],
            }, ensure_ascii=False)

    @mcp.tool()
    def introspect() -> str:
        """Quick look inward: current project, trace counts by kind/status/scope,
        association counts, gate decision tallies, activity statistics."""
        stats = {"project": default_project, **hippo.stats(),
                 "scopes": hippo.store.scope_counts()}
        return json.dumps(stats, ensure_ascii=False)

    @mcp.tool()
    def dream_log() -> str:
        """Full state slice accumulated across consolidations ("sleep"):
        retention mean, fading traces, index load, activity counters. Use to
        reason about how memory behaves and evolves over time."""
        return json.dumps(hippo.metrics_snapshot(), ensure_ascii=False)

    return mcp


def make_embedder(choice: str = "auto"):
    """Эмбеддер по выбору: local (fastembed ONNX), hashing (детерминированный
    тестовый) или auto (local при наличии fastembed, иначе hashing)."""
    if choice == "hashing":
        from ..encoding.embedder import HashingEmbedder

        return HashingEmbedder(dim=256)
    if choice == "local":
        from ..encoding.embedder_local import FastEmbedProvider

        return FastEmbedProvider()
    if choice == "auto":
        try:
            from ..encoding.embedder_local import FastEmbedProvider

            return FastEmbedProvider()
        except ImportError:
            # stdout у stdio-сервера — канал JSON-RPC, диагностика только в stderr
            print(
                "[realmemory] fastembed не установлен — используется HashingEmbedder. "
                "Установите: pip install 'realmemory[local]'",
                file=sys.stderr,
            )
            from ..encoding.embedder import HashingEmbedder

            return HashingEmbedder(dim=256)
    raise ValueError(f"неизвестный эмбеддер: {choice}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="realmemory-mcp")
    parser.add_argument("--path", required=True, help="каталог базы памяти")
    parser.add_argument(
        "--namespace",
        default=None,
        help="подкаталог внутри --path: изоляция проектов/контекстов",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="скоуп проекта по умолчанию; по умолчанию определяется автоматически "
             "(REALMEMORY_PROJECT / ZCODE_PROJECT_DIR / рабочая директория с .git)",
    )
    parser.add_argument(
        "--embedder",
        choices=["auto", "local", "hashing"],
        default="local",
        help="эмбеддер: local = fastembed ONNX локально; hashing = детерминированный без модели",
    )
    parser.add_argument("--units", type=int, default=None,
                        help="n_units SDR-пространства; фиксируются базой при первом открытии")
    parser.add_argument("--k-sparse", type=int, default=None,
                        help="on-битов на SDR-паттерн")
    parser.add_argument("--bucket-cap", type=int, default=None,
                        help="горизонт вытеснения L1-бакета (palimpsest)")
    args = parser.parse_args(argv)
    from ..config import MemoryConfig
    from ..hippocampus import Hippocampus
    from ..projects import resolve_project

    embedder = make_embedder(args.embedder)
    cfg = MemoryConfig(dim=embedder.dim)
    profile = getattr(embedder, "recommended_thresholds", None)
    if profile:
        # пороги калиброваны под конкретный эмбеддер (анизотропия моделей);
        # попадут в db_meta.config при первом открытии базы
        for key, val in profile.items():
            setattr(cfg, key, float(val))
    if args.units is not None:
        cfg.n_units = int(args.units)
    if args.k_sparse is not None:
        cfg.k_sparse = int(args.k_sparse)
    if args.bucket_cap is not None:
        cfg.bucket_cap = int(args.bucket_cap)
    cfg.validate()
    identity = ""
    try:
        from ..team.identity import resolve_identity

        identity = resolve_identity()
    except Exception:  # noqa: BLE001 - идентичность не должна ломать запуск
        identity = ""
    hippo = Hippocampus.open(args.path, config=cfg, embedder=embedder,
                             namespace=args.namespace, author=identity)
    default_project = resolve_project(args.project)
    if default_project:
        # stdout занят JSON-RPC-транспортом — служебные строки только в stderr
        print(f"[realmemory] project scope: {default_project}", file=sys.stderr,
              flush=True)
    build_server(hippo, default_project=default_project,
                 team_root=args.path,
                 team_policy_path=_team_policy_path()).run()


def _team_policy_path():
    """Путь политики командного слоя; None → тулы команды не регистрируются."""
    try:
        from ..team.policy import DEFAULT_POLICY_PATH

        return DEFAULT_POLICY_PATH
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":  # pragma: no cover
    main()
