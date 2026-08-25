"""CLI отчёта (python -m realmemory.report): сборка и рендер на живом сценарии."""
import hashlib

from realmemory.hippocampus import Hippocampus
from realmemory.report import build_report, render


def _tok(i: int) -> str:
    digest = hashlib.blake2b(f"fact-{i}".encode("ascii"), digest_size=9,
                             person=b"rm-rep").hexdigest()
    return "".join(chr(ord("a") + int(c, 16)) for c in digest)


def test_report_on_populated_memory(tmp_path, tiny_cfg, clock):
    h = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    try:
        for i in range(5):
            h.remember(f"korvex {_tok(i)} qixa")
        h.recall(f"korvex {_tok(1)}", k=3)
        h.recall("wwgrond zztilde qqzorf", k=3)  # воздержание попадёт в журнал
        h.feedback([1], 0.8)
        h.consolidate()
        path = h.path
    finally:
        h.close()

    report = build_report(path)
    assert not report.get("empty")
    assert report["memories_total"] >= 5
    assert report["journal"]["recalls"]["count"] == 2
    assert report["journal"]["recalls"]["abstain_rate"] == 0.5
    assert report["journal"]["consolidation"]["runs"] == 1
    assert report["index"]["traces_indexed"] >= 5
    assert len(report["top_reinforced"]) >= 1
    assert report["journal"]["metrics_series_len"] == 1

    text = render(report)
    assert "realMemory report" in text
    assert "воздержание" in text
    assert "динамика" in text


def test_report_on_empty_dir(tmp_path):
    report = build_report(tmp_path / "nothing")
    assert report.get("empty")
    assert "пуста" in render(report)
