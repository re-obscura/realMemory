"""Хуки hook_cli: SessionStart-брифинг и «сон» с троттлингом по состоянию базы."""
import json
import sqlite3

import pytest

from realmemory.hook_cli import main as hook_main


@pytest.fixture
def populated(tmp_path, tiny_cfg, clock):
    from realmemory import Hippocampus

    h = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    for i in range(4):
        h.remember(f"korvex kv{i} qixa")
    sem = h.remember("plimso pt9 suvat")
    for _ in range(5):
        h.remember("plimso pt9 suvat")  # подкрепления -> повышение в semantic
    clock.advance(2500)
    h.consolidate()
    path = str(h.path)
    yield path, sem.memory_id
    h.close()


def _meta(db: str, key: str) -> str | None:
    con = sqlite3.connect(db)
    try:
        row = con.execute("SELECT value FROM db_meta WHERE key=?", (key,)).fetchone()
    finally:
        con.close()
    return None if row is None else row[0]


def test_brief_emits_valid_session_start_json(populated, capsys):
    path, _ = populated
    with pytest.raises(SystemExit) as e:
        hook_main(["brief", "--path", path])
    assert e.value.code == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(out)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "traces" in ctx


def test_brief_plain_lists_semantic(populated, capsys):
    path, _ = populated
    with pytest.raises(SystemExit) as e:
        hook_main(["brief", "--path", path, "--plain", "--top", "5"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "semantic" in out
    assert "plimso pt9 suvat" in out


def test_sleep_skips_when_nothing_new(populated, capsys):
    path, _ = populated
    # только что консолидировали и ничего не писали -> тихий пропуск
    with pytest.raises(SystemExit) as e:
        hook_main(["sleep", "--path", path])
    assert e.value.code == 0
    assert capsys.readouterr().out == ""


def test_sleep_runs_when_dirty(populated, capsys):
    path, _ = populated
    db = f"{path}/memory.db"
    # появилась новая запись после прошлого сна -> сон обязан выполниться
    from realmemory import Hippocampus
    from realmemory.config import MemoryConfig
    from realmemory.encoding.embedder import HashingEmbedder

    cfg = MemoryConfig.from_snapshot(json.loads(_meta(db, "config")))
    h2 = Hippocampus.open(path, config=cfg,
                          embedder=HashingEmbedder(dim=cfg.dim), verify_embedder=False)
    try:
        h2.remember("korvex kv8 fresh")
    finally:
        h2.close()
    before = float(_meta(db, "last_consolidate_at"))
    with pytest.raises(SystemExit) as e:
        hook_main(["sleep", "--path", path])
    assert e.value.code == 0
    assert capsys.readouterr().out == ""
    assert float(_meta(db, "last_consolidate_at")) > before


def test_sleep_on_missing_dir_is_safe(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(SystemExit) as e:
        hook_main(["sleep", "--path", str(missing)])
    assert e.value.code == 0
