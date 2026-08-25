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
                "created_at": it.created_at,
                "meta": dict(it.meta),
            }
            for it in packet.items
        ],
    }


def build_server(hippo):
    """Собрать FastMCP-сервер над фасадом. Требует пакет fastmcp (extra [mcp])."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise ImportError(
            "MCP extra не установлен. Установите: pip install 'realmemory[mcp]'"
        ) from exc

    mcp = FastMCP("realmemory")

    @mcp.tool()
    def recall(query: str, k: int = 5, include_superseded: bool = False) -> str:
        """Recollect long-term memories relevant to the query.

        Use before making claims about the user, their projects or previous
        sessions. Returns JSON: items [{id, text, kind, cosine, confidence,
        retention, source, created_at, meta}] ranked by confidence, plus an
        abstained flag. abstained=true means nothing trustworthy was remembered —
        say so honestly instead of guessing. Items with source=associated arrived
        via associative links and may be less precise; present confidence < 0.2
        as "possibly", not as fact.
        """
        packet = hippo.recall(query, k=k, include_superseded=include_superseded)
        return json.dumps(_packet_to_dict(packet), ensure_ascii=False)

    @mcp.tool()
    def memorize(text: str, kind: str = "episodic", related_ids: list[int] | None = None) -> str:
        """Commit one durable fact, decision or preference to long-term memory.

        Write conclusions rather than conversation snippets ("we chose SQLite WAL",
        "user prefers concise answers"). The novelty gate decides automatically:
        an already-known fact gets reinforced, a related one links to it, a fresh
        one allocates a new trace — reformulations don't pile up. Pass related_ids
        when this builds on specific earlier memories. Returns {memory_id, action,
        created}. kind: episodic (default) or semantic.
        """
        res = hippo.remember(text, kind=kind, related_ids=tuple(related_ids or ()))
        return json.dumps(
            {
                "memory_id": res.memory_id,
                "created": res.created,
                "action": res.decision.action.value,
                "novelty": round(res.decision.novelty, 4),
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
        recalls; nothing is silently lost. Returns {old_id, new_id}.
        """
        res = hippo.update_fact(int(old_id), new_text)
        return json.dumps({"old_id": int(old_id), "new_id": res.memory_id},
                          ensure_ascii=False)

    @mcp.tool()
    def introspect() -> str:
        """Quick look inward: trace counts by kind/status, association counts,
        gate decision tallies, activity statistics."""
        return json.dumps(hippo.stats(), ensure_ascii=False)

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
            print(
                "[realmemory] fastembed не установлен — используется HashingEmbedder. "
                "Установите: pip install 'realmemory[local]'"
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
        "--embedder",
        choices=["auto", "local", "hashing"],
        default="local",
        help="эмбеддер: local = fastembed ONNX локально; hashing = детерминированный без модели",
    )
    args = parser.parse_args(argv)
    from ..config import MemoryConfig
    from ..hippocampus import Hippocampus

    embedder = make_embedder(args.embedder)
    cfg = MemoryConfig(dim=embedder.dim)
    hippo = Hippocampus.open(args.path, config=cfg, embedder=embedder,
                             namespace=args.namespace)
    build_server(hippo).run()


if __name__ == "__main__":  # pragma: no cover
    main()
