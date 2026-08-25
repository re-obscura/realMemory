"""Скоупы памяти: проектная изоляция при видимости global, resolve_project."""
import pytest

from realmemory import Hippocampus
from realmemory.projects import normalize_slug, resolve_project
from realmemory.types import SCOPE_GLOBAL


def test_project_facts_isolated_global_visible(hippo):
    # лексический эмбеддер тестов: факты различаются уникальными токенами
    g = hippo.remember("briefmode user prefers short answers", scope=SCOPE_GLOBAL)
    p1 = hippo.remember("alpha stack uses kvpg16 engine", scope="alpha")
    hippo.remember("beta stack uses kvmg7 store", scope="beta")

    # свой проект + global видны...
    got = {it.memory_id for it in hippo.recall("kvpg16 stack", k=5, scope="alpha").items}
    assert p1.memory_id in got or g.memory_id in got
    assert all(
        it.scope in ("alpha", SCOPE_GLOBAL)
        for it in hippo.recall("stack", k=10, scope="alpha").items
    )
    # ...чужой проект — нет
    alpha_texts = " ".join(
        it.text for it in hippo.recall("kvmg7 store", k=5, scope="alpha").items
    )
    assert "kvmg7" not in alpha_texts
    # без скоупа — вся память
    everything = {it.memory_id for it in hippo.recall("kvpg16 kvmg7", k=10).items}
    assert len(everything) >= 2


def test_gate_does_not_merge_across_scopes(hippo):
    a = hippo.remember("korvex kv001 qixa", scope="alpha")
    b = hippo.remember("korvex kv001 qixa", scope="beta")
    assert b.created and b.memory_id != a.memory_id
    again_beta = hippo.remember("korvex kv001 qixa", scope="beta")
    assert not again_beta.created and again_beta.memory_id == b.memory_id
    assert hippo.store.get(a.memory_id).scope == "alpha"
    assert hippo.store.get(b.memory_id).scope == "beta"


def test_update_fact_inherits_scope(hippo):
    old = hippo.remember("Проект гамма на Python 3.12", scope="gamma")
    new = hippo.update_fact(old.memory_id, "Проект гамма на Python 3.14")
    assert hippo.store.get(new.memory_id).scope == "gamma"


def test_invalid_scope_rejected(hippo):
    with pytest.raises(ValueError):
        hippo.remember("x y z", scope="не-валидный!")
    with pytest.raises(ValueError):
        hippo.remember("x y z", scope="")


def test_scope_counts(hippo):
    hippo.remember("fact one alpha", scope="alpha")
    hippo.remember("fact two beta", scope="beta")
    counts = hippo.store.scope_counts()
    assert counts["alpha"] == 1 and counts["beta"] == 1
    assert counts.get(SCOPE_GLOBAL, 0) == 0


def test_legacy_db_without_scope_column_migrates(tmp_path):
    """Регресс: база до WP2 (без колонки scope) должна открываться,
    индекс по scope создаётся строго после ALTER."""
    import sqlite3

    db = tmp_path / "legacy.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE memories ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, kind TEXT NOT NULL,"
        " status TEXT NOT NULL, meta TEXT NOT NULL, embedding BLOB NOT NULL,"
        " sdr BLOB NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,"
        " reinforced_count INTEGER NOT NULL DEFAULT 0, last_reinforced_at REAL NOT NULL,"
        " base_strength REAL NOT NULL DEFAULT 1.0, valid_from REAL NOT NULL,"
        " valid_to REAL, superseded_by INTEGER)"
    )
    con.execute(
        "INSERT INTO memories(text,kind,status,meta,embedding,sdr,created_at,updated_at,"
        "reinforced_count,last_reinforced_at,base_strength,valid_from) "
        "VALUES ('old fact','episodic','active','{}',"
        "x'0000000000000000000000000000000000000000000000000000000000000000',"
        "x'00000000',1.0,1.0,0,1.0,0.5,1.0)"
    )
    con.commit()
    con.close()

    from realmemory.store.sqlite_store import MemoryStore

    store = MemoryStore(db, dim=8)
    try:
        assert store.get(1).scope == "global"
        assert store.count(scope="global") == 1
    finally:
        store.close()


# -- resolve_project ---------------------------------------------------------------

def test_resolve_explicit_wins(monkeypatch):
    monkeypatch.setenv("REALMEMORY_PROJECT", "envslug")
    monkeypatch.setenv("ZCODE_PROJECT_DIR", "D:/projs/envdir")
    assert resolve_project("explicit-name") == "explicit-name"


def test_resolve_from_env(monkeypatch):
    monkeypatch.setenv("REALMEMORY_PROJECT", "envslug")
    monkeypatch.setenv("ZCODE_PROJECT_DIR", "D:/projs/envdir")
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    assert resolve_project() == "envslug"


def test_resolve_from_zcode_project_dir(monkeypatch):
    monkeypatch.delenv("REALMEMORY_PROJECT", raising=False)
    monkeypatch.setenv("ZCODE_PROJECT_DIR", "D:/projs/my_repo")
    assert resolve_project() == "my_repo"


def test_resolve_none_outside_repo(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # без .git/.zcode
    for var in ("REALMEMORY_PROJECT", "ZCODE_PROJECT_DIR", "CLAUDE_PROJECT_DIR"):
        monkeypatch.delenv(var, raising=False)
    assert resolve_project() is None


def test_resolve_cwd_with_git_marker(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    for var in ("REALMEMORY_PROJECT", "ZCODE_PROJECT_DIR", "CLAUDE_PROJECT_DIR"):
        monkeypatch.delenv(var, raising=False)
    assert resolve_project() == tmp_path.name


def test_normalize_slug():
    assert normalize_slug("my.repo-2") == "my.repo-2"
    assert normalize_slug("") is None
    assert normalize_slug(None) is None
    assert normalize_slug("bad/slash") is None
    assert normalize_slug("has space") is None  # пробел вне алфавита слагов
    assert normalize_slug("x" * 65) is None


def test_mcp_tools_pass_scope(tmp_path, tiny_cfg, clock):
    """MCP-тулы уважают default_project сервера и явный project."""
    pytest.importorskip("fastmcp")
    import asyncio
    import json

    from fastmcp import Client

    from realmemory.api.mcp_server import build_server

    h = Hippocampus.open(tmp_path / "rm", config=tiny_cfg, clock=clock)
    try:
        server = build_server(h, default_project="demo")

        async def scenario():
            async with Client(server) as client:
                mem = await client.call_tool("memorize", {"text": "demo stack fact"})
                out = json.loads(mem.content[0].text)
                assert out["scope"] == "demo"
                rec = await client.call_tool("recall", {"query": "demo stack"})
                packet = json.loads(rec.content[0].text)
                assert all(item["scope"] == "demo" for item in packet["items"])

        asyncio.run(scenario())
    finally:
        h.close()
