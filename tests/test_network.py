"""Сетевой слой команды: координатор (presence/publish/retract/search),
клиентский транспорт, синхронизация registry→координатор и командный recall.
Координатор поднимается в потоке на эфемерном порту."""
import base64
import threading

import numpy as np
import pytest

from realmemory import Hippocampus
from realmemory.team import registry
from realmemory.team.coordinator import make_server
from realmemory.team.policy import TeamPolicy, save_policy
from realmemory.team.transport import (
    AuthError,
    CoordinatorClient,
    EmbedderMismatch,
    TransportError,
)
from realmemory.timeprov import FakeClock


def _vec(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(384).astype(np.float32)


@pytest.fixture
def coord(tmp_path):
    srv = make_server(tmp_path / "coord", host="127.0.0.1", port=0,
                      token="sekret")
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    port = srv.server_address[1]
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def _client(url, token="sekret"):
    return CoordinatorClient(url, token=token, timeout_s=3.0)


# -- транспорт и аутентификация -----------------------------------------------------

def test_health_and_heartbeat_presence(coord):
    c = _client(coord)
    assert c.health() == {"ok": True}
    c.heartbeat("andrey", projects=["proj"])
    c.heartbeat("maxim")
    presence = {p["identity"]: p for p in c.presence()}
    assert set(presence) == {"andrey", "maxim"}
    assert all(p["online"] for p in presence.values())


def test_wrong_token_rejected(coord):
    with pytest.raises(AuthError):
        _client(coord, token="wrong").heartbeat("intruder")


def test_unreachable_is_transport_error(tmp_path):
    # порт 1 на localhost почти наверняка закрыт; короткий таймаут
    c = CoordinatorClient("http://127.0.0.1:1", token="x", timeout_s=1.5)
    with pytest.raises(TransportError):
        c.health()


# -- publish/retract/search ---------------------------------------------------------

def _pub_item(pid, seed=11, author="andrey", project="proj",
              text="решение через journalctl"):
    v = _vec(seed)
    return {"publication_id": pid, "project": project, "author": author,
            "text": text,
            "embedding_b64": base64.b64encode(v.tobytes()).decode(),
            "published_at": 100.0 + seed / 1000.0,
            "content_hash": f"h{seed}", "embedder":
                "fastembed:test-model@1.2"}


def test_publish_search_and_filters(coord):
    c = _client(coord)
    assert c.publish_batch([_pub_item("p1"), _pub_item("p2", seed=12)]) == 2
    assert c.publish_batch([_pub_item("p1")]) == 0  # идемпотентность по id

    q = _vec(11)
    hits = c.search(q, k=5, embedder="fastembed:test-model@1.2")
    assert hits and hits[0]["publication_id"] == "p1"
    assert all(h["score"] >= 0 for h in hits)

    only_maxim = c.search(q, k=5, embedder="fastembed:test-model@1.2",
                          author="maxim")
    assert only_maxim == []

    # чужой эмбеддер — честный отказ вместо мусорного косинуса
    with pytest.raises(EmbedderMismatch):
        c.search(_vec(13), k=5, embedder="hashing(dim=256,seed=7)")

    # отзыв убирает из поиска, tombstone остаётся в дампе
    assert c.retract_batch([{"publication_id": "p1", "revoked_at": 500.0}]) == 1
    hits_after = c.search(q, k=5, embedder="fastembed:test-model@1.2")
    assert [h["publication_id"] for h in hits_after] == ["p2"]
    dump = c.cache_dump()
    assert [t["publication_id"] for t in dump["tombstones"]] == ["p1"]


# -- интеграция: локальный registry → sync → recall_team -----------------------------

def _brain(tmp_path, author):
    from realmemory import Hippocampus
    from realmemory.timeprov import FakeClock
    from tests.test_team import _cfg

    return Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock(),
                            author=author)


def _policy_file(tmp_path, url):
    policy = TeamPolicy(identity="maxim", coordinator=url)
    save_policy(policy, tmp_path / "team.yaml")
    return policy


def test_sync_pushes_registry_and_recall_finds_it(tmp_path, coord,
                                                    monkeypatch):
    monkeypatch.setenv("REALMEMORY_TEAM_TOKEN", "sekret")
    h = _brain(tmp_path, "maxim")
    try:
        mid = h.remember("для кэша моделей используется fastembed ONNX локально",
                         scope="proj").memory_id
        # (follow-up: близкий факт коллеги связался бы, тут пишем сами себе)
        res = h.remember("выбор протокола командной памяти p401", scope="proj")
        del res
        rows = registry.publish(h.store, [mid], policy=_policy_file(
            tmp_path, coord), now=1000.0, identity="maxim")
        assert len(rows) == 1

        before = registry.sync_status(h.store)
        assert before["awaiting_sync"] == 1

        from realmemory.team.sync import push

        summary = push(h.store, _policy_file(tmp_path, coord))
        assert summary.published == 1 and summary.marked == 1
        after = registry.sync_status(h.store)
        assert after["awaiting_sync"] == 0
    finally:
        h.close()


def test_retract_travels_on_second_sync(tmp_path, coord, monkeypatch):
    monkeypatch.setenv("REALMEMORY_TEAM_TOKEN", "sekret")
    h = _brain(tmp_path, "maxim")
    try:
        mid = h.remember("временное решение которое отзовём p402",
                         scope="proj").memory_id
        pol = _policy_file(tmp_path, coord)
        registry.publish(h.store, [mid], policy=pol, now=100.0, identity="maxim")

        from realmemory.team.sync import push

        push(h.store, pol)
        n = registry.retract(h.store, now=200.0, trace_ids=[mid])
        assert n == 1
        summary = push(h.store, pol)
        assert summary.retracted == 1
        c = _client(coord)
        dump = c.cache_dump()
        assert dump["active"] == [] and len(dump["tombstones"]) == 1
    finally:
        h.close()


def test_network_failure_keeps_decisions_pending(tmp_path, coord):
    """Отказные тесты §7 спеки: сеть умерла до/во время — решения живы."""
    h = _brain(tmp_path, "maxim")
    try:
        mid = h.remember("важное решение команды p403",
                         scope="proj").memory_id
        registry.publish(h.store, [mid], policy=_policy_file(
            tmp_path, "http://127.0.0.1:9"), now=100.0, identity="maxim")

        from realmemory.team.sync import push
        from realmemory.team.transport import TransportError as _TE

        with pytest.raises((RuntimeError, _TE)):
            push(h.store, _policy_file(tmp_path, "http://127.0.0.1:9"),
                 timeout_s=1.0)
        st = registry.sync_status(h.store)
        assert st["awaiting_sync"] == 1  # решение не потеряно
    finally:
        h.close()


def test_mcp_tool_hidden_without_coordinator(tmp_path):
    """Kill-switch: без coordinator в политике тула recall_team не существует."""
    from fastmcp import FastMCP

    from realmemory.api.mcp_server import build_server
    from tests.test_team import _cfg

    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock())
    try:
        server = build_server(h)
        tools = asyncio.run(server.get_tools()) if hasattr(
            server, "get_tools") else {}
        names = set(tools or [])
        # recall_team появляется только при настроенном командном слое
        if isinstance(server, FastMCP):
            assert "recall_team" not in names
    finally:
        h.close()


import asyncio
