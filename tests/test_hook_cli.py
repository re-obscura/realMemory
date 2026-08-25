"""Хуки hook_cli: SessionStart-брифинг и «сон» с троттлингом."""
import json
import os
import time

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


def test_sleep_skips_when_recently_idle(populated, capsys):
    path, _ = populated
    # только что консолидировали и ничего не писали -> тихий пропуск
    with pytest.raises(SystemExit) as e:
        hook_main(["sleep", "--path", path])
    assert e.value.code == 0
    assert capsys.readouterr().out == ""


def test_sleep_runs_when_stale_or_dirty(populated, capsys):
    path, _ = populated
    snap = os.path.join(path, "snapshot.pkl")
    old = time.time() - 4000  # старше min-interval -> сон обязан выполниться
    os.utime(snap, (old, old))
    with pytest.raises(SystemExit) as e:
        hook_main(["sleep", "--path", path])
    assert e.value.code == 0
    assert capsys.readouterr().out == ""
    assert os.path.getmtime(snap) > old


def test_sleep_on_missing_dir_is_safe(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(SystemExit) as e:
        hook_main(["sleep", "--path", str(missing)])
    assert e.value.code == 0
