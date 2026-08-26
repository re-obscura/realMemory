"""Бенчмарк recall: полный конвейер realMemory против точного косинус-бейзлайна.

Фаза 0 измеряет корректность механики извлечения, а не семантическую сложность:
синтетический корпус из непересекающихся словарей, запросы — подмножества слов
факта. Честность сравнения: бейзлайн — тот же эмбеддер, точный поиск по всем
эмбеддингам (верхняя граница качества); конвейер добавляет L1-адресацию,
голоса локаций и retention в ранжирование.

Успех фазы 0 (критерии из docs/CONTRACTS.md):
  pipeline_hits@k >= baseline_hits@k - 0.02 и abstention >= 0.9 на noise-запросах.
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..config import MemoryConfig
from ..hippocampus import Hippocampus

_TOPICS = [
    {"name": "A", "prefix": "kv", "words": ["korvex", "deltun", "miphar", "soltak"]},
    {"name": "B", "prefix": "wf", "words": ["wreniq", "fablex", "guvnot", "charod"]},
    {"name": "C", "prefix": "pt", "words": ["plimso", "vendrik", "thaxol", "brumel"]},
    {"name": "D", "prefix": "yz", "words": ["yandor", "oximet", "klavur", "zenpak"]},
    {"name": "E", "prefix": "hn", "words": ["hupsil", "argemo", "novula", "cetrix"]},
]
_ATTRS = ["qixa", "romed", "suvat", "lanek"]
_NOISE_POOL = ["qqzorf", "xxumple", "vvasken", "zztilde", "wwgrond", "jjamplex", "kkrevno", "ffubdex"]


def _obj_token(i: int) -> str:
    """Уникальный объектный токен факта: хэш-случайные буквы.

    Соседние индексы не делят триграммы — важно: если токены разных фактов
    лексически близки, novelty-гейт корректно сливает их при записи, и
    отдельного следа у факта нет (это поведение системы, а не ошибка).
    """
    digest = hashlib.blake2b(f"fact-{i}".encode("ascii"), digest_size=9,
                             person=b"rm-bench").hexdigest()
    return "".join(chr(ord("a") + int(c, 16)) for c in digest)


def _make_corpus(n_facts: int, seed: int) -> list[tuple[str, int]]:
    """(текст факта, индекс) с уникальным объектным токеном на факт."""
    rng = np.random.default_rng(seed)
    corpus: list[tuple[str, int]] = []
    for i in range(int(n_facts)):
        topic = _TOPICS[int(rng.integers(len(_TOPICS)))]
        w1 = topic["words"][int(rng.integers(len(topic["words"])))]
        attr = _ATTRS[int(rng.integers(len(_ATTRS)))]
        corpus.append((f"{w1} {_obj_token(i)} {attr}", i))
    return corpus


def _subset_query(fact_text: str, fi: int, rng) -> str:
    """Запрос-подмножество: два слова из трёх, всегда включая объектный токен."""
    parts = fact_text.split()
    obj = _obj_token(fi)
    rest = [p for p in parts if p != obj]
    keep = rest[int(rng.integers(len(rest)))]
    return f"{keep} {obj}" if rng.random() < 0.5 else f"{obj} {keep}"


def _bench_config(n_facts: int) -> MemoryConfig:
    """Конфиг под размер корпуса.

    Ёмкость L1: нагрузка на юнит = corpus*k/n_units; горизонт palimpsest —
    bucket_cap относительно этой нагрузки. Для 1500 фактов: 1500*96/2048 ≈ 70
    на бакет при cap 512 — вытеснения нет; dim=1024+ держит коллизионный пол
    hashing-эмбеддера ниже cos_min_recall (условие честного abstention).
    Для крупных корпусов ёмкостные параметры обязаны расти вместе с базой
    (иначе начинается осмысленное вытеснение-забывание): при ≥10k фактов
    n_units=16384 держит нагрузку ≤300 на бакет на 50k.
    """
    n_units = 16384 if int(n_facts) > 5000 else 2048
    return MemoryConfig(
        dim=2048,
        n_units=n_units,
        k_sparse=96,
        bucket_cap=512,
        # порог воздержания — в середине зазора между полом коллизий эмбеддера
        # на несвязных текстах (~0.12 при dim=2048) и связными запросами (>0.4)
        cos_min_recall=0.18,
        # точный скан видит хвост псевдосовпадений, который узкое окно голосования
        # физически не достигало; относительное воздержание возвращает его:
        # цель держит объектный токен (top-1 >= 0.88), шум не поднимается выше
        # ~0.21..0.30 на 50k, медиана разреженного корпуса около нуля
        exact_abstain_rel_margin=0.30,
    )


def run(
    n_facts: int = 1500,
    n_queries: int = 200,
    k: int = 10,
    seed: int = 0,
    config: MemoryConfig | None = None,
    verbose: bool = True,
) -> dict:
    rng = np.random.default_rng(seed + 1)
    cfg = config or _bench_config(n_facts)
    tmp = Path(tempfile.mkdtemp(prefix="rm_bench_"))
    hippo = Hippocampus.open(tmp / "rm", config=cfg)
    try:
        corpus = _make_corpus(n_facts, seed)
        t0 = time.perf_counter()
        ids = [hippo.remember(text).memory_id for text, _ in corpus]
        write_s = time.perf_counter() - t0
        emb_matrix = np.stack([hippo.embedder.embed(text) for text, _ in corpus])
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix /= norms

        query_idx = rng.choice(len(corpus), size=min(n_queries, len(corpus)), replace=False)
        # факты, слитые гейтом новизны в чужой след (лексические близнецы
        # объектных токенов — поведение системы, их следа не существует);
        # полноту извлечения считаем только по собственным следам,
        # доля слияний публикуется отдельной метрикой
        id_counts = Counter(ids)
        singleton_target = {i: id_counts[ident] == 1 for i, ident in enumerate(ids)}
        pipe_hits = 0
        base_hits = 0
        evaluated = 0
        latencies: list[float] = []
        confidences: list[float] = []
        for qi in query_idx:
            text, fi = corpus[int(qi)]
            q = _subset_query(text, fi, rng)
            packet = hippo.recall(q, k=k)
            latencies.append(packet.latency_ms)
            if singleton_target[fi]:
                evaluated += 1
                pipe_hits += int(any(item.memory_id == ids[fi] for item in packet.items))
                q_emb = hippo.embedder.embed(q)
                qn = float(np.linalg.norm(q_emb))
                sims = emb_matrix @ (q_emb / qn if qn else q_emb)
                top = np.argpartition(-sims, min(k, sims.size) - 1)[:k]
                base_hits += int(fi in {int(t) for t in top})
            if packet.items:
                confidences.append(packet.items[0].confidence)

        abstained = 0
        base_noise_top1: list[float] = []
        for _ in range(max(20, n_queries // 5)):
            noise = " ".join(str(x) for x in rng.choice(_NOISE_POOL, size=3, replace=False))
            packet = hippo.recall(noise, k=k)
            abstained += int(packet.abstained)
            q_emb = hippo.embedder.embed(noise)
            qn = float(np.linalg.norm(q_emb))
            if qn:
                sims = emb_matrix @ (q_emb / qn)
                base_noise_top1.append(float(sims.max()))
        n_noise = max(20, n_queries // 5)

        lat = np.asarray(latencies)

        # восстановление состояния: время пересборки производных структур
        # (L1-бакеты, юнит-индекс, CSR рёбер) при открытии базы
        hippo.close()
        t_open = time.perf_counter()
        hippo = Hippocampus.open(tmp / "rm", config=cfg)
        reopen_ms = (time.perf_counter() - t_open) * 1000.0

        hits_pipe = round(pipe_hits / max(1, evaluated), 4)
        hits_base = round(base_hits / max(1, evaluated), 4)
        abstain_rate = round(abstained / n_noise, 4)
        merged_share = round(1.0 - len(set(ids)) / max(1, len(ids)), 4)
        metrics: dict[str, float | str] = {
            "n_facts": n_facts,
            "gate_merged_share": merged_share,
            "evaluated_queries": evaluated,
            "pipeline_hits@k": hits_pipe,
            "baseline_hits@k": hits_base,
            "abstention_on_noise": abstain_rate,
            "baseline_noise_top1_cos_mean": round(float(np.mean(base_noise_top1)), 4),
            "recall_p50_ms": round(float(np.percentile(lat, 50)), 3),
            "recall_p95_ms": round(float(np.percentile(lat, 95)), 3),
            "writes_per_sec": round(len(ids) / max(write_s, 1e-9), 1),
            "mean_top1_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 4),
            "reopen_rebuild_ms": round(reopen_ms, 1),
        }
        ok = hits_pipe >= hits_base - 0.02 and abstain_rate >= 0.9
        metrics["phase0_gate"] = "PASS" if ok else "FAIL"
        if verbose:
            _print_table(metrics, k)
        return metrics
    finally:
        hippo.close()


def _print_table(m: dict, k: int) -> None:
    width = 34
    print("=" * width)
    print(f"realMemory bench: {m['n_facts']} facts, hits@{k}")
    print("=" * width)
    rows = [
        ("pipeline hits@k (singleton targets)", m["pipeline_hits@k"]),
        ("baseline hits@k (exact cos)", m["baseline_hits@k"]),
        ("evaluated queries", m["evaluated_queries"]),
        ("write-gate merged share", m["gate_merged_share"]),
        ("abstention on noise", m["abstention_on_noise"]),
        ("baseline noise top-1 cos", m["baseline_noise_top1_cos_mean"]),
        ("recall latency p50/p95 ms", f"{m['recall_p50_ms']} / {m['recall_p95_ms']}"),
        ("writes/sec", m["writes_per_sec"]),
        ("reopen rebuild ms", m["reopen_rebuild_ms"]),
        ("mean top-1 confidence", m["mean_top1_confidence"]),
    ]
    for name, val in rows:
        print(f"{name:<32} {val}")
    print("-" * width)
    print(f"phase-0 gate: {m['phase0_gate']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="realmemory-bench")
    parser.add_argument("--facts", type=int, default=1500)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    run(n_facts=args.facts, n_queries=args.queries, k=args.k, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
