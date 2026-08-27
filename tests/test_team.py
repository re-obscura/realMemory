"""Командный слой: миграция v2 (author/publications), политика шеринга,
registry публикаций с tombstones, headless-смок TUI."""
import asyncio
import sqlite3

import pytest

from realmemory import Hippocampus, MemoryConfig
from realmemory.store.sqlite_store import SCHEMA_VERSION
from realmemory.team import registry
from realmemory.team.identity import resolve_identity
from realmemory.team.policy import (
    BLOCKED_NEVER,
    ELIGIBLE,
    LOW_REINFORCEMENTS,
    PROJECT_OFF,
    WRONG_KIND,
    TeamPolicy,
    classify,
    load_policy,
    save_policy,
    set_project_shareable,
)
from realmemory.timeprov import FakeClock


def _cfg(**over) -> MemoryConfig:
    fields = {
        "dim": 256, "n_units": 512, "k_sparse": 48, "sdr_seed": 5,
        "bucket_cap": 32,
    }
    fields.update(over)
    cfg = MemoryConfig(**fields)
    cfg.validate()
    return cfg


# -- схема v2 и авторство -------------------------------------------------------------

def test_fresh_db_is_v2_with_author_and_publications(tmp_path):
    store = _store = None
    from realmemory.store.sqlite_store import MemoryStore

    store = MemoryStore(tmp_path / "m.db", dim=8)
    try:
        assert store.schema_version == str(SCHEMA_VERSION) == "2"
        con = sqlite3.connect(str(tmp_path / "m.db"))
        cols = {r[1] for r in con.execute("PRAGMA table_info(memories)")}
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "author" in cols and "publications" in tables
        con.close()
    finally:
        store.close()


def test_legacy_db_upgrades_to_v2(tmp_path):
    """База v0.4 (schema_version='1', без author) поднимается цепочкой до v2."""
    db = tmp_path / "legacy.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " text TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,"
        " meta TEXT NOT NULL, embedding BLOB NOT NULL, sdr BLOB NOT NULL,"
        " created_at REAL NOT NULL, updated_at REAL NOT NULL,"
        " reinforced_count INTEGER NOT NULL DEFAULT 0,"
        " last_reinforced_at REAL NOT NULL, base_strength REAL NOT NULL DEFAULT 1.0,"
        " valid_from REAL NOT NULL, valid_to REAL, superseded_by INTEGER,"
        " scope TEXT NOT NULL DEFAULT 'global')")
    con.execute("CREATE TABLE db_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    con.execute("INSERT INTO db_meta VALUES('schema_version','1')")
    con.commit(); con.close()

    from realmemory.store.sqlite_store import MemoryStore

    store = MemoryStore(db, dim=8)
    try:
        assert store.schema_version == "2"
    finally:
        store.close()


def test_remember_stamps_author(tmp_path):
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock(),
                         author="mtrunov")
    try:
        mid = h.remember("факт про тесты авторства ua901").memory_id
        rec = h.store.get(mid)
        assert rec.author == "mtrunov"
        packet = h.recall("факт про тесты авторства", k=1)
        assert packet.items[0].author == "mtrunov"
    finally:
        h.close()


def test_resolve_identity_prefers_explicit_then_env(monkeypatch):
    monkeypatch.delenv("REALMEMORY_IDENTITY", raising=False)
    assert resolve_identity("андрей") == "андрей"
    monkeypatch.setenv("REALMEMORY_IDENTITY", " maxim ")
    assert resolve_identity() == "maxim"


# -- политика -------------------------------------------------------------------------

def _policy(tmp_path, **over):
    policy = TeamPolicy(projects=[{"name": "proj"}][0:0])  # пусто по умолчанию
    policy.identity = over.get("identity", "me")
    set_project_shareable(policy, "proj", True)
    return policy


def test_policy_yaml_roundtrip_keeps_unknown_keys(tmp_path):
    path = tmp_path / "team.yaml"
    path.write_text("custom_future_key: hello\nidentity: someone\n",
                    encoding="utf-8")
    policy = load_policy(path)
    assert policy.raw.get("custom_future_key") == "hello"
    save_policy(policy, path)
    after = load_policy(path)
    assert after.raw.get("custom_future_key") == "hello"


def test_classify_ladder(tmp_path):
    import numpy as np

    from realmemory.types import MemoryRecord

    def rec(kind="semantic", rep=3, meta=None, text="обычный факт",
            scope="proj"):
        rng = np.random.default_rng(7)
        return MemoryRecord(
            id=1, text=text, kind=kind, status="active", meta=dict(meta or {}),
            embedding=rng.standard_normal(8).astype(np.float32),
            sdr=np.arange(4, dtype=np.int32), created_at=0.0, updated_at=0.0,
            reinforced_count=rep, last_reinforced_at=0.0, base_strength=0.5,
            valid_from=0.0, scope=scope)

    policy = _policy(None)
    d = classify(rec(), policy)
    assert d.status == ELIGIBLE

    assert classify(rec(kind="episodic"), policy).status == WRONG_KIND
    assert classify(rec(rep=1), policy).status == LOW_REINFORCEMENTS
    off = classify(rec(scope="other"), policy)
    assert off.status == PROJECT_OFF

    secret = classify(rec(meta={"tags": ["private"]}), policy)
    assert secret.status == BLOCKED_NEVER
    leaky = classify(rec(text="наш api_key = sk-123"), policy)
    assert leaky.status == BLOCKED_NEVER

    # никогда не побеждается ничем, даже недостатком подкреплений
    assert classify(rec(rep=0, meta={"tags": ["secret"]}),
                    policy).status == BLOCKED_NEVER


# -- registry ------------------------------------------------------------------------

class _Env:
    """Маленький контекст: личная база + политика с включённым проектом."""

    def __init__(self, tmp_path):
        self.hippo = Hippocampus.open(tmp_path / "rm", config=_cfg(),
                                      clock=FakeClock(), author="me")
        self.clock = FakeClock()
        self.policy = _policy(tmp_path)

    def remember(self, text, scope="proj", rep_before=3):
        mid = self.hippo.remember(text, scope=scope).memory_id
        for _ in range(rep_before):
            self.hippo.feedback([mid], 1.0)
        return mid


def test_publish_and_tombstone_roundtrip(tmp_path):
    env = _Env(tmp_path)
    h = env.hippo
    try:
        a = env.remember("решение о структуре хранения p301")
        b = env.remember("выбор протокола синхронизации p302")

        rows = registry.publish(h.store, [a, b], policy=env.policy,
                                now=100.0, identity="me")
        assert len(rows) == 2 and rows[0].author == "me"
        active = registry.active_publications(h.store)
        assert {p.trace_id for p in active} == {a, b}
        st = registry.sync_status(h.store)
        assert st["active"] == 2 and st["tombstones"] == 0

        # повторная публикация того же следа — новая строка истории
        again = registry.publish(h.store, [a], policy=env.policy,
                                 now=200.0, identity="me")
        assert again[0].publication_id != rows[0].publication_id

        n = registry.retract(h.store, now=300.0, trace_ids=[a])
        assert n == 2  # обе публикации следа получили tombstone
        st = registry.sync_status(h.store)
        # активной осталась только публикация следа b; обе строки следа a
        # переехали в tombstones
        assert st["active"] == 1 and st["tombstones"] == 2
    finally:
        h.close()


def test_publish_refuses_never_rules_whole_batch(tmp_path):
    env = _Env(tmp_path)
    h = env.hippo
    try:
        ok = env.remember("безопасный факт p303")
        bad = env.remember("тут лежит пароль = hunter2")
        with pytest.raises(registry.RegistryError, match="never"):
            registry.publish(h.store, [ok, bad], policy=env.policy,
                             now=10.0, identity="me")
        # пакет отменён целиком: ничего не опубликовано
        assert registry.sync_status(h.store)["active"] == 0
    finally:
        h.close()


def test_publish_refuses_foreign_authorship(tmp_path):
    env = _Env(tmp_path)
    h = env.hippo
    try:
        mid = env.remember("запись андрея для команды p304")
        rec = h.store.get(mid)
        h.store.update_trace(mid, rec.base_strength, rec.reinforced_count,
                             rec.last_reinforced_at)
        con = sqlite3.connect(str(h.path / "memory.db"))
        con.execute("UPDATE memories SET author='andrey' WHERE id=?", (mid,))
        con.commit(); con.close()
        with pytest.raises(registry.RegistryError, match="andrey"):
            registry.publish(h.store, [mid], policy={**env.policy.__dict__
                               } and _policy(tmp_path), now=5.0, identity="me")
    finally:
        h.close()


# -- TUI headless smoke ---------------------------------------------------------------

def test_tui_headless_smoke(tmp_path):
    from realmemory.team.tui import TeamApp

    env = _Env(tmp_path)
    try:
        tid = env.remember("решение о структуре хранения t901")
        app = TeamApp(env.hippo.path)
        async def scenario():
            async with app.run_test(size=(110, 34)) as pilot:
                await pilot.pause()
                assert app.current == "proj"
                table = app.query_one("#candidates")
                assert table.row_count >= 1
                assert tid in app.row_owner.values()          # проект выбран сам
                table = app.query_one("#candidates")
                assert table.row_count >= 1
                await pilot.press("s")                   # toggle политики → уведомление
                await pilot.pause()
                await pilot.press("p")                   # публикация без выбора
                await pilot.pause()
                app.exit()
        asyncio.run(scenario())
    finally:
        env.hippo.close()
