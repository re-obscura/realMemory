"""Полный отчёт о состоянии памяти — инструмент анализа накопленных данных.

Запуск: python -m realmemory.report --path ./rm_data [--json report.json]

Источник один: SQLite-база (следы, рёбра L2, журнал событий). Отчёт отвечает
на вопросы: как растёт память, как ведёт себя гейт новизны, каковы доля
воздержания и латентность, что подкрепляется, что угасает, как менялись
метрики от консолидации к консолидации.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _pctl(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    i = min(len(sorted_xs) - 1, max(0, round(q * (len(sorted_xs) - 1))))
    return float(sorted_xs[i])


def _connect(root: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(root / "memory.db"))


def _index_stats(root: Path) -> dict[str, Any]:
    """Загрузка бакетов L1 по активным следам (прямой подсчёт постингов)."""
    con = _connect(root)
    try:
        counts: Counter[int] = Counter()
        traces = 0
        for (blob,) in con.execute("SELECT sdr FROM memories WHERE status='active'"):
            arr = np.frombuffer(blob, dtype=np.int32)
            counts.update(arr.tolist())
            traces += 1
    finally:
        con.close()
    loads = list(counts.values())
    return {
        "traces_indexed": traces,
        "units_touched": len(counts),
        "mean_bucket_load": round(float(np.mean(loads)), 2) if loads else 0.0,
        "max_bucket_load": int(max(loads)) if loads else 0,
    }


def _db_stats(root: Path) -> dict[str, Any]:
    con = _connect(root)
    try:
        rows = con.execute(
            "SELECT kind, status, COUNT(*) FROM memories GROUP BY kind, status"
        ).fetchall()
        top = con.execute(
            "SELECT id, reinforced_count, kind, substr(text,1,90) FROM memories "
            "WHERE status='active' ORDER BY reinforced_count DESC, last_reinforced_at DESC LIMIT 10"
        ).fetchall()
        stale = con.execute(
            "SELECT id, kind, substr(text,1,90) FROM memories WHERE status='active' "
            "AND kind='episodic' ORDER BY last_reinforced_at ASC LIMIT 10"
        ).fetchall()
    finally:
        con.close()
    return {
        "memories": {f"{kind}/{status}": n for kind, status, n in rows},
        "memories_total": sum(n for _, _, n in rows),
        "top_reinforced": [
            {"id": i, "reinforced": n, "kind": k, "text": t} for i, n, k, t in top
        ],
        "stale_episodic_candidates_for_forgetting": [
            {"id": i, "kind": k, "text": t} for i, k, t in stale
        ],
    }


def _graph_stats(root: Path) -> dict[str, Any]:
    """Рёбра L2 и решения гейта новизны — теперь из таблиц базы."""
    con = _connect(root)
    try:
        (n_edges, total_w) = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(w), 0.0) FROM edges"
        ).fetchone()
        decisions = {
            str(a): int(n)
            for a, n in con.execute(
                "SELECT COALESCE(json_extract(data,'$.action'),'unknown'), COUNT(*) "
                "FROM events WHERE type='write' GROUP BY 1"
            ).fetchall()
        }
    finally:
        con.close()
    return {
        "gate_decisions_all_time": decisions,
        "edges_committed_current": int(n_edges),
        "total_edge_weight": round(float(total_w), 4),
    }


def _event_stats(root: Path) -> dict[str, Any]:
    events = []
    con = _connect(root)
    try:
        for ts, etype, data in con.execute("SELECT ts, type, data FROM events ORDER BY seq"):
            event = {"ts": float(ts), "type": etype}
            try:
                event.update(json.loads(data))
            except ValueError:
                pass
            events.append(event)
    finally:
        con.close()
    by_type = Counter(e.get("type") for e in events)

    recalls = [e for e in events if e.get("type") == "recall"]
    lats = sorted(float(e.get("latency_ms", 0.0)) for e in recalls)
    abstained = sum(1 for e in recalls if e.get("abstained"))
    confs = [float(e["top_conf"]) for e in recalls if e.get("top_conf") is not None]

    feedbacks = [e for e in events if e.get("type") == "feedback"]
    rewards = [float(e.get("reward", 0.0)) for e in feedbacks]

    consolidates = [e for e in events if e.get("type") == "consolidate"]

    series = [
        {
            "ts": time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("ts", 0))),
            "active": e.get("memories_active"),
            "edges": e.get("edges"),
            "retention_mean": e.get("retention_mean"),
            "fading<0.2": e.get("retention_below_02"),
        }
        for e in events
        if e.get("type") == "metrics"
    ]

    return {
        "events_total": len(events),
        "by_type": dict(by_type),
        "recalls": {
            "count": len(recalls),
            "abstain_rate": round(abstained / len(recalls), 4) if recalls else None,
            "latency_p50_ms": round(_pctl(lats, 0.5), 3),
            "latency_p95_ms": round(_pctl(lats, 0.95), 3),
            "mean_top_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        },
        "feedback": {
            "calls": len(feedbacks),
            "positive": sum(1 for r in rewards if r > 0),
            "negative": sum(1 for r in rewards if r < 0),
        },
        "consolidation": {
            "runs": len(consolidates),
            "promoted_to_semantic_total": sum(
                int(e.get("promoted_to_semantic", 0)) for e in consolidates
            ),
            "edges_committed_total": sum(int(e.get("edges_committed", 0)) for e in consolidates),
        },
        "metrics_series_len": len(series),
        "metrics_series_last50": series[-50:],
    }


def build_report(path: str | Path, namespace: str | None = None) -> dict[str, Any]:
    root = Path(path)
    if namespace is not None:
        root = root / namespace
    db_path = root / "memory.db"

    files = {f.name: f.stat().st_size for f in sorted(root.glob("*")) if f.is_file()}
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "path": str(root),
        "files": files,
    }
    report["total_size_mb"] = round(sum(files.values()) / 1e6, 3)
    if not db_path.exists():
        report["empty"] = True
        return report

    report.update(_db_stats(root))
    report["index"] = _index_stats(root)
    report["graph"] = _graph_stats(root)
    report["journal"] = _event_stats(root)
    return report


def render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    add("=" * 64)
    add(f"realMemory report — {report.get('path')} ({report['generated_at']})")
    add("=" * 64)
    add(f"storage: {report.get('total_size_mb', 0)} MB  {report.get('files', {})}")
    if report.get("empty"):
        add("память пуста")
        return "\n".join(lines)

    add("")
    add("-- следы -------------------------------------------------------------")
    for key, val in sorted(report.get("memories", {}).items()):
        add(f"  {key:<28} {val}")
    add(f"  {'итого':<28} {report.get('memories_total')}")

    js = report.get("journal", {})
    add("")
    add("-- журнал ------------------------------------------------------------")
    add(f"  событий всего: {js.get('events_total')}")
    for t, c in sorted(js.get("by_type", {}).items(), key=lambda kv: -kv[1]):
        add(f"    {t!s:<14} {c}")

    rec = js.get("recalls", {})
    add("")
    add("-- recall ------------------------------------------------------------")
    add(f"  запросов: {rec.get('count')}; воздержание: {rec.get('abstain_rate')}")
    add(f"  латентность p50/p95: {rec.get('latency_p50_ms')} / {rec.get('latency_p95_ms')} мс")
    add(f"  средняя уверенность топа: {rec.get('mean_top_confidence')}")
    fb = js.get("feedback", {})
    add(f"  feedback: вызовов {fb.get('calls')} (+{fb.get('positive', 0)} / -{fb.get('negative', 0)})")

    cons = js.get("consolidation", {})
    add("")
    add("-- сон ----------------------------------------------------------------")
    add(f"  консолидаций: {cons.get('runs')}; повышено до семантических: {cons.get('promoted_to_semantic_total')}")
    add(f"  связей закоммичено всего: {cons.get('edges_committed_total')}")

    idx = report.get("index", {})
    add("")
    add("-- L1 индекс ----------------------------------------------------------")
    add(f"  следов в индексе: {idx.get('traces_indexed')}; юнитов задействовано: {idx.get('units_touched')}")
    add(f"  средняя/макс нагрузка бакета: {idx.get('mean_bucket_load')} / {idx.get('max_bucket_load')}")

    graph = report.get("graph", {})
    if graph:
        add("")
        add("-- гейт новизны (все решения) ------------------------------------------")
        for act, c in graph.get("gate_decisions_all_time", {}).items():
            add(f"    {act:<12} {c}")
        if "edges_committed_current" in graph:
            add(f"    активных рёбер L2: {graph['edges_committed_current']}")

    add("")
    add("-- самые подкреплённые ------------------------------------------------")
    for item in report.get("top_reinforced", []):
        add(f"  [{item['id']:>6}] x{item['reinforced']} ({item['kind']}) {item['text']}")

    add("")
    add("-- угасающие эпизоды (дольше всех без подкрепления) -------------------")
    for item in report.get("stale_episodic_candidates_for_forgetting", []):
        add(f"  [{item['id']:>6}] ({item['kind']}) {item['text']}")

    series = js.get("metrics_series_last50", [])
    if series:
        add("")
        add("-- динамика (последние консолидации) -----------------------------------")
        add("  время               active  edges  retention  fading<0.2")
        for m in series[-15:]:
            add(
                f"  {m['ts']:<18} {m['active']!s:>6} {m['edges']!s:>6}"
                f" {m['retention_mean']!s:>9} {m['fading<0.2']!s:>10}"
            )
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="realmemory-report")
    parser.add_argument("--path", required=True, help="каталог базы памяти")
    parser.add_argument("--namespace", default=None, help="подкаталог внутри --path")
    parser.add_argument("--json", default=None, help="сохранить полный отчёт в JSON")
    args = parser.parse_args(argv)
    report = build_report(args.path, namespace=args.namespace)
    print(render(report))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON сохранён: {args.json}")


if __name__ == "__main__":
    main()
