"""Бенчмарк realMemory на реальном тексте (RU/EN, fastembed).

Запуск: python -m realmemory.eval.bench_real [--k 10] [--verbose] [--json out.json]

В отличие от синтетического bench_recall (hashing-эмбеддер, непересекающиеся
словари), здесь проверяется боевая конфигурация: локальная семантическая
модель, естественные переформулировки, поиск по точным токенам и шумовые
запросы на воздержание.

Бенчмарк также печатает калибровку порогов гейта под эмбеддер:
- распределение косинуса «факт — его ближайший сосед» по несвязным фактам
  (нулевой уровень для theta_reinforce/theta_link; дубликат-пары исключены);
- сигнал дубликатов: косинус переформулировки к оригиналу (что должно быть
  выше порога подкрепления);
- max-cos и top1-cos запросов по типам (нулевой уровень для cos_min_recall).

Все ссылки в фикстуре — по строковым id фактов: позиционные индексы в
авторских руках ошибочны.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "bench_real.json"


def _pctl(sorted_xs: Sequence[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    i = min(len(sorted_xs) - 1, max(0, round(q * (len(sorted_xs) - 1))))
    return float(sorted_xs[i])


def _dist(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    return {
        "min": round(s[0], 4),
        "p50": round(_pctl(s, 0.5), 4),
        "p95": round(_pctl(s, 0.95), 4),
        "max": round(s[-1], 4),
    }


def _load_profile(cfg, embedder) -> None:
    profile = getattr(embedder, "recommended_thresholds", None)
    if profile:
        for key, val in profile.items():
            setattr(cfg, key, float(val))
    cfg.validate()


def _run(
    hippo,
    *,
    embedder,
    cfg,
    facts: list[str],
    dup_of: list[int | None],
    queries: list[dict],
    index_of: dict[str, int],
    k: int,
) -> dict[str, Any]:
    # -- запись ------------------------------------------------------------------
    t0 = time.perf_counter()
    ids: list[int] = []
    gate_by_kind: dict[str, dict[str, int]] = {"base": {}, "duplicate": {}}
    for i, text in enumerate(facts):
        res = hippo.remember(text)
        kind = "duplicate" if dup_of[i] is not None else "base"
        action = res.decision.action.value
        gate_by_kind[kind][action] = gate_by_kind[kind].get(action, 0) + 1
        ids.append(res.memory_id)
    write_seconds = time.perf_counter() - t0

    # матрица эмбеддингов фактов для калибровки
    emb = np.stack([embedder.embed(t) for t in facts])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    # сигнал дубликатов: косинус переформулировки к оригиналу
    dup_signal = sorted(
        float(emb[i] @ emb[orig]) for i, orig in enumerate(dup_of) if orig is not None
    )

    # -- запросы -----------------------------------------------------------------
    lats: list[float] = []
    sources: dict[str, int] = {}
    by_type: dict[str, dict[str, float]] = {}
    naive_by_type: dict[str, dict[str, float]] = {}
    rows_by_type: dict[str, list[dict]] = {}

    for q in queries:
        packet = hippo.recall(q["q"], k=k)
        lats.append(round(packet.latency_ms, 3))
        qtype = q["type"]

        qv = np.asarray(embedder.embed_query(q["q"]), dtype=np.float32)
        # без builtin max: на части mypy/numpy-комбинаций его вывод — union,
        # не приводимый к float; EPS-пол уже заодно и с делением на ноль
        norm = float(np.linalg.norm(qv))
        qv /= norm if norm > 1e-9 else 1e-9
        row_sims = emb @ qv
        expect_idx = index_of[q["expect"]] if qtype != "noise" else None
        target_cos = (
            round(float(row_sims[expect_idx]), 6) if expect_idx is not None else None
        )
        item_cos = sorted(
            (it.cosine for it in packet.items if it.source == "direct"), reverse=True,
        )

        # наивный baseline: чистый косинус по всем фактам без гейта/decay/L1,
        # воздержание по фиксированному порогу (тот же cos_min_strong_recall)
        if expect_idx is not None:
            order = np.argsort(-row_sims)[:k]
            naive_rank = (
                int(np.where(order == expect_idx)[0][0]) + 1
                if expect_idx in order else None
            )
        else:
            naive_rank = None

        bucket = by_type.setdefault(qtype, {"total": 0.0, "hits": 0.0, "mrr": 0.0})
        naive_bucket = naive_by_type.setdefault(
            qtype, {"total": 0.0, "hits": 0.0, "mrr": 0.0},
        )
        bucket["total"] += 1
        naive_bucket["total"] += 1
        rank = None
        if qtype == "noise":
            naive_hit = float(row_sims.max()) < cfg.cos_min_strong_recall
            naive_bucket["hits"] += int(naive_hit)
            bucket["hits"] += int(packet.abstained)
            rows_by_type.setdefault(qtype, []).append({"q": q["q"], "item_cos": item_cos})
        else:
            assert expect_idx is not None  # noise отсмотрен выше
            target_id = ids[expect_idx]
            if naive_rank is not None and naive_rank <= k:
                naive_bucket["hits"] += 1
                naive_bucket["mrr"] += 1.0 / naive_rank
            rank = next(
                (i + 1 for i, it in enumerate(packet.items) if it.memory_id == target_id),
                None,
            )
            if rank is not None and rank <= k:
                bucket["hits"] += 1
                bucket["mrr"] += 1.0 / rank
            rows_by_type.setdefault(qtype, []).append({
                "q": q["q"], "rank": rank, "cos_target": target_cos,
                "fact": facts[expect_idx], "item_cos": item_cos,
                "naive_rank": naive_rank,
            })
        for it in packet.items:
            sources[it.source] = sources.get(it.source, 0) + 1

    # -- калибровка ----------------------------------------------------------------
    sims = emb @ emb.T
    # нуль «факт—ближайший сосед» считается без дубликатов-переформулировок,
    # иначе истинные пары портят нулевое распределение
    null_mask = np.ones_like(sims, dtype=bool)
    for i, orig in enumerate(dup_of):
        if orig is not None:
            null_mask[i, orig] = False
            null_mask[orig, i] = False
    sims_null = np.where(null_mask, sims, -np.inf)
    np.fill_diagonal(sims_null, -np.inf)
    nn_sims = sorted(float(x) for x in sims_null.max(axis=1))

    def top1(t: str) -> dict:
        return _dist([r["item_cos"][0] for r in rows_by_type.get(t, []) if r["item_cos"]])

    report: dict[str, Any] = {
        "facts": len(facts),
        "queries": len(queries),
        "writes_per_sec": round(len(facts) / write_seconds, 1),
        "latency_p50_ms": round(_pctl(sorted(lats), 0.5), 2),
        "latency_p95_ms": round(_pctl(sorted(lats), 0.95), 2),
        "sources": sources,
        "gate_base": gate_by_kind["base"],
        "gate_duplicate": gate_by_kind["duplicate"],
        "dup_signal": _dist(dup_signal),
        "thresholds": {
            "theta_reinforce": cfg.theta_reinforce,
            "theta_link": cfg.theta_link,
            "cos_min_recall": cfg.cos_min_recall,
        },
        "by_type": {},
        "naive_baseline": {},
        "calibration": {
            "nn_fact_sim": _dist(nn_sims),
            "query_max_cos": {},
            "target_cos": {
                t: _dist([r["cos_target"] for r in rs if r.get("cos_target") is not None])
                for t, rs in rows_by_type.items()
            },
            "top1_direct_cos": {t: top1(t) for t in ("paraphrase", "token", "noise")},
        },
    }
    for qt in ("paraphrase", "token", "noise"):
        qs = [q["q"] for q in queries if q["type"] == qt]
        if qs:
            report["calibration"]["query_max_cos"][qt] = _dist(
                [float((emb @ _qvec(embedder, qq)).max()) for qq in qs]
            )
    for qtype, b in by_type.items():
        n = max(1.0, b["total"])
        report["by_type"][qtype] = {
            "total": int(b["total"]),
            "hits": int(b["hits"]),
            "hit_rate": round(b["hits"] / n, 4),
            "mrr": round(b["mrr"] / n, 4),
        }
        nb = naive_by_type.get(qtype)
        if nb:
            nn = max(1.0, nb["total"])
            report["naive_baseline"][qtype] = {
                "total": int(nb["total"]),
                "hits": int(nb["hits"]),
                "hit_rate": round(nb["hits"] / nn, 4),
                "mrr": round(nb["mrr"] / nn, 4),
            }
    report["_rows"] = rows_by_type
    return report


def _qvec(embedder, text: str) -> np.ndarray:
    v = np.asarray(embedder.embed_query(text), dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="realmemory-bench-real")
    parser.add_argument("--fixture", default=str(FIXTURE_PATH))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true",
                        help="печатать проваленные запросы")
    parser.add_argument("--json", default=None, help="сохранить полный отчёт в JSON")
    args = parser.parse_args(argv)

    try:
        from ..encoding.embedder_local import FastEmbedProvider

        embedder = FastEmbedProvider()
    except ImportError as exc:
        print(f"[skip] нужен боевой эмбеддер: {exc}")
        return 0

    from ..config import MemoryConfig
    from ..hippocampus import Hippocampus

    cfg = MemoryConfig(dim=embedder.dim)
    _load_profile(cfg, embedder)

    fixture = Path(args.fixture)
    data = json.loads(fixture.read_text(encoding="utf-8"))
    facts = [f["text"] for f in data["facts"]]
    index_of = {f["id"]: i for i, f in enumerate(data["facts"])}
    dup_of: list[int | None] = [
        None if "dup_of" not in f else index_of[f["dup_of"]] for f in data["facts"]
    ]
    queries = data["queries"]

    with tempfile.TemporaryDirectory(prefix="rm-bench-real-", ignore_cleanup_errors=True) as td:
        hippo = Hippocampus.open(td, config=cfg, embedder=embedder)
        try:
            report = _run(
                hippo, embedder=embedder, cfg=cfg, facts=facts, dup_of=dup_of,
                queries=queries, index_of=index_of, k=args.k,
            )
        finally:
            hippo.close()

    rows = report.pop("_rows")

    print("=" * 68)
    print(f"realMemory bench_real — {report['facts']} фактов, "
          f"{report['queries']} запросов, dim={embedder.dim}, k={args.k}")
    print("=" * 68)
    print(f"запись: {report['writes_per_sec']} следов/сек; recall p50/p95: "
          f"{report['latency_p50_ms']} / {report['latency_p95_ms']} мс")
    labels = {"paraphrase": "переформулировки", "token": "точные токены",
              "noise": "шум (воздержание)"}
    print("pipeline vs наивный baseline (чистый косинус, порог воздержания):")
    for qtype in ("paraphrase", "token", "noise"):
        b = report["by_type"].get(qtype)
        if not b:
            continue
        nb = report["naive_baseline"].get(qtype)
        if qtype == "noise":
            print(f"{labels[qtype]:<22} pipeline {b['hit_rate']:.3f} "
                  f"({b['hits']}/{b['total']})   naive {nb['hit_rate']:.3f}"
                  if nb else
                  f"{labels[qtype]:<22} pipeline {b['hit_rate']:.3f} "
                  f"({b['hits']}/{b['total']})")
        else:
            naive_part = (f"   naive hits@{args.k}: {nb['hit_rate']:.3f}, MRR {nb['mrr']:.3f}"
                          if nb else "")
            print(f"{labels[qtype]:<22} pipeline hits@{args.k}: {b['hit_rate']:.3f}, "
                  f"MRR {b['mrr']:.3f}{naive_part}")
    print(f"источники выдачи: {report['sources']}")
    base_total = sum(report["gate_base"].values())
    print(f"гейт на записи: база {report['gate_base']} из {base_total}; "
          f"дубликаты {report['gate_duplicate']}")
    cal = report["calibration"]
    print("калибровка порогов:")
    print(f"  нуль «факт-сосед» (без дупл.): {cal['nn_fact_sim']}")
    print(f"  сигнал дубликатов:             {report['dup_signal']}  "
          f"(reinforce={cfg.theta_reinforce}, link={cfg.theta_link})")
    for t in ("paraphrase", "token", "noise"):
        d = cal["query_max_cos"].get(t)
        if d:
            print(f"  max-cos запроса [{t:<10}]: {d}  (cos_min_recall={cfg.cos_min_recall})")
    for t, d in cal["target_cos"].items():
        print(f"  cos(запрос—цель) [{t:<10}]: {d}")

    if args.verbose:
        print("\n-- проваленные запросы --")
        for qtype in ("paraphrase", "token"):
            for r in rows.get(qtype, []):
                if r["rank"] is None or r["rank"] > args.k:
                    print(f"[{qtype}] rank={r['rank']} cos_target={r['cos_target']} "
                          f"q={r['q']!r}\n     цель: {r['fact'][:90]!r}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"\nJSON сохранён: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
