"""Golden-тест скоринга recall: пакеты обязаны совпадать байт-в-байт
(кроме latency_ms — wall-clock) с зафиксированным снимком.

Снимок покрывает оба движка (exact/votes), оба правила воздержания,
волну ассоциаций, superseded, scope-фильтр и all_scopes. Регенерация
эталона: python tests/test_recall_snapshot.py (перезаписывает fixtures).
"""
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realmemory import Hippocampus, MemoryConfig
from realmemory.timeprov import FakeClock

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "recall_snapshot.json"

QUERIES = [
    ("deploy pipeline failed at gateway", {"k": 3}),                     # direct exact
    ("borsh cooking recipe", {"k": 2}),
    ("postgres миграция схемы", {"k": 3}),                               # unicode
    ("ECONNREFUSED", {"k": 5}),                                          # exact token
    ("quarterly roadmap review friday notes", {"k": 4}),                 # full keyword
    ("asyncio exception", {"k": 2}),
    ("память агента следы", {"k": 2}),
    ("completely unrelated noise about quantum bots", {"k": 3}),         # noise
    ("deploy pipeline failed at gateway", {"k": 5, "all_scopes": True}),
    ("borsh cooking recipe", {"k": 2, "include_superseded": True}),
    ("meeting notes review", {"k": 10}),                                 # волна ассоциаций
    ("nothing matches this zzzqqq", {"k": 4}),
    ("beetroot", {"k": 3}),
    ("taskgroup", {"k": 3}),
    ("миграция", {"k": 3}),
]


def build_scenario(root: Path, *, exact: bool) -> Hippocampus:
    cfg = MemoryConfig(
        dim=256, n_units=512, k_sparse=48, sdr_seed=5, bucket_cap=64,
        tau_episodic=100_000.0, tau_semantic=1_000_000.0, initial_strength=0.5,
        promote_after_reinforcements=5, promote_min_age_s=2000.0,
        tau_eligibility=50_000.0, tau_edge_stable=300_000.0,
        exact_scan_recall=exact,
        # votes-движок принудительно: потолок 1 всегда превышается
        exact_scan_max_traces=40 if exact else 1,
        # оба правила воздержания включены, чтобы покрыть их ветки
        cos_min_strong_recall=0.42,
        abstain_spread_cos=0.20,
        exact_abstain_rel_margin=0.05,
    )
    cfg.validate()
    clock = FakeClock()
    h = Hippocampus.open(root, config=cfg, clock=clock)
    facts = [
        "deploy pipeline failed with ECONNREFUSED at gateway",
        "cooking recipe borsh with beetroot and beef",
        "meeting notes: quarterly roadmap review on friday",
        "python asyncio taskgroup exception handling",
        "база данных postgres миграция схемы v3",
        "память агента хранит следы и забывает лишнее",
    ]
    for text in facts:
        h.remember(text, scope="proj")
    for _ in range(5):
        h.remember("meeting notes: quarterly roadmap review on friday", scope="proj")
    clock.advance(3000)
    h.consolidate()  # промоция + рёбра L2
    old = h.store.get(1)
    h.update_fact(old.id, "deploy pipeline failed with ECONNRESET at gateway")
    clock.advance(500)
    h.link_memories([2, 3], strength=1.0)
    h.consolidate()
    return h


def capture() -> dict:
    results = {}
    for exact in (True, False):
        with tempfile.TemporaryDirectory() as td:
            h = build_scenario(Path(td) / "rm", exact=exact)
            try:
                packets = []
                for q, kw in QUERIES:
                    packets.append(asdict(h.recall(q, **kw)))
                packets.append(asdict(h.recall("deploy pipeline failed at gateway",
                                               k=3, scope="proj")))
                results[f"exact_{exact}"] = packets
            finally:
                h.close()
    for packets in results.values():
        for p in packets:
            p.pop("latency_ms")  # wall-clock, в эталон не входит
    return quantize(results)


def quantize(obj):
    """Округление float до 5 знаков: последний бит gemv гуляет между сборками
    BLAS (урок v0.8.2, сравнение с abs=1e-5) — recall округляет до 6, этого
    мало для кросс-платформенной стабильности, сравниваем с точностью 1e-5.
    Кортежи (RecallPacket.items после asdict) разворачиваются в списки —
    сравнение всё равно идёт через JSON."""
    if isinstance(obj, float):
        return round(obj, 5)
    if isinstance(obj, tuple):
        return [quantize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: quantize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [quantize(v) for v in obj]
    return obj


def test_recall_packets_stable_against_fixture():
    assert FIXTURE.exists(), "эталон отсутствует: python tests/test_recall_snapshot.py"
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # JSON round-trip: кортежи пакетов сравниваются с массивами эталона;
    # обе стороны квантизированы в capture() — эталон платформонезависим
    actual = json.loads(json.dumps(capture()))
    assert actual == expected, (
        "пакеты recall изменились: если изменение сознательное — "
        "перегенерируйте эталон (python tests/test_recall_snapshot.py)"
    )


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(capture(), ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print("эталон обновлён:", FIXTURE)
