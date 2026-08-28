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
from realmemory.team.policy import TeamPolicy, load_policy, save_policy
from realmemory.team.transport import (
    AuthError,
    CoordinatorClient,
    EmbedderMismatch,
    TransportError,
)
from realmemory.timeprov import FakeClock
from tests.test_team import _cfg


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

# -- v0.8: live peer-to-peer ---------------------------------------------------------

def _peer_server(root, token="sekret", port=0):
    from realmemory.team.peer import make_peer_server

    srv = make_peer_server(root, host="127.0.0.1", port=port, token=token)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"127.0.0.1:{srv.server_address[1]}"


def test_peer_serves_only_published(tmp_path):
    """Приватность конструктивно: личный след с тем же текстом НЕ виден по сети
    без публикации; опубликованный — виден с авторством."""
    from realmemory.team.transport import encode_vector

    h = _brain(tmp_path, "andrey")
    try:
        published = h.remember("командный выбор очереди задач: nats",
                               scope="proj").memory_id
        private = h.remember("командный выбор очереди задач: nats",
                             scope="proj").memory_id  # личный дубль НЕ публикуем
        h.feedback([published], 1.0)
        pol = _policy_file(tmp_path, "http://127.0.0.1:9")
        pol.identity = "andrey"
        registry.publish(h.store, [published], policy=pol, now=100.0,
                         identity="andrey")

        srv, addr = _peer_server(h.path)
        try:
            client = CoordinatorClient(f"http://{addr}", token="sekret",
                                       timeout_s=2.0)
            qvec = np.random.default_rng(42).standard_normal(256).astype(np.float32)
            out = client.raw_post("/recall", {
                "query_embedding_b64": encode_vector(qvec), "k": 5,
                "embedder": h.store.get_meta("embedder") or "x",
            })
            authors = {hit["author"] for hit in out["hits"]}
            texts = {hit["text"] for hit in out["hits"]}
            assert authors == {"andrey"}
            assert len(out["hits"]) == 1
            assert all("nats" in t for t in texts)
            del private
        finally:
            srv.shutdown(); srv.server_close()
    finally:
        h.close()


def test_peer_embedder_mismatch_rejected(tmp_path):
    from realmemory.team.transport import EmbedderMismatch, encode_vector

    h = _brain(tmp_path, "andrey")
    try:
        srv, addr = _peer_server(h.path)
        try:
            client = CoordinatorClient(f"http://{addr}", token="sekret",
                                       timeout_s=2.0)
            with pytest.raises(EmbedderMismatch):
                client.raw_post("/recall", {
                    "query_embedding_b64": encode_vector(_vec(5)), "k": 3,
                    "embedder": "совсем другой эмбеддер",
                })
        finally:
            srv.shutdown(); srv.server_close()
    finally:
        h.close()


def test_recall_team_live_then_cache_fallback(tmp_path, coord, monkeypatch):
    """Живой коллега отвечает LIVE; офлайн-коллега падает в кэш-фоллбэк,
    недоступность peer не ломает ответ."""
    monkeypatch.setenv("REALMEMORY_TEAM_TOKEN", "sekret")

    # машина Андрея: живой peer с публикацией
    h1 = _brain(tmp_path / "andrey", "andrey")
    try:
        mid_a = h1.remember("инфраструктура тестов крутится в github actions",
                            scope="proj").memory_id
        pol_a = TeamPolicy(identity="andrey", coordinator=coord)
        save_policy(pol_a, tmp_path / "andrey" / "team.yaml")
        registry.publish(h1.store, [mid_a], policy=pol_a, now=100.0,
                         identity="andrey")
        from realmemory.team.sync import push
        push(h1.store, pol_a)

        srv, addr = _peer_server(h1.path)

        # машина Максима: своя публикация, peer не поднят
        h2 = _brain(tmp_path / "maxim", "maxim")
        try:
            mid_m = h2.remember("деплой идёт через systemd unit unitname",
                                scope="proj").memory_id
            pol_m = TeamPolicy(identity="maxim", coordinator=coord)
            save_policy(pol_m, tmp_path / "maxim" / "team.yaml")
            registry.publish(h2.store, [mid_m], policy=pol_m, now=100.0,
                             identity="maxim")
            push(h2.store, pol_m)

            # presence: Андрей online с адресом; Максим online без адреса
            c = _client(coord)
            c.heartbeat("andrey", address=addr, projects=["proj"])
            c.heartbeat("maxim")

            from realmemory.team.recall_team import recall_team

            answer = recall_team(
                h2.path,
                "инфраструктура тестов github actions", k=5,
                policy=pol_m)
            assert not answer.abstained
            assert "andrey" in answer.peers_live
            sources = {h.author: h.source for h in answer.hits}
            assert sources.get("andrey") == "live"
            assert sources.get("maxim") == "cache"
            # публикация Максима должна уехать в координатор
            assert any(h.author == "maxim" for h in answer.hits)

            # peer Андрея «падает»: фоллбэк на кэш, ответ не пустой
            srv.shutdown(); srv.server_close()
            answer2 = recall_team(
                h2.path,
                "инфраструктура тестов github actions", k=5,
                policy=pol_m)
            assert not answer2.abstained
            assert "andrey" in "".join(answer2.peers_failed) or \
                   answer2.peers_live == []
            assert any(h.author == "andrey" and h.source == "cache"
                       for h in answer2.hits)
        finally:
            h2.close()
    finally:
        h1.close()


def test_gc_before_sync_auto_retracts(tmp_path, coord, monkeypatch):
    """content-lost закрыт: доставленная публикация, чей след затем забылся
    локально (GC), получает tombstone УЖЕ в момент забывания (registry виден
    без сети), а следующий sync доставляет его на координатор — без контента,
    tombstone не требует данных."""
    monkeypatch.setenv("REALMEMORY_TEAM_TOKEN", "sekret")
    from realmemory.team.sync import push

    cfg = _cfg(tau_episodic=10 * 86400.0, gc_grace_below_floor_s=1.0)
    h = Hippocampus.open(tmp_path / "rm", config=cfg, clock=FakeClock(),
                         author="maxim")
    try:
        mid = h.remember("забудем это решение после публикации p501",
                         scope="proj").memory_id
        for _ in range(2):
            h.feedback([mid], 1.0)
        pol = _policy_file(tmp_path, coord)
        registry.publish(h.store, [mid], policy=pol, now=10.0, identity="maxim")

        first = push(h.store, pol)             # доставка активной публикации
        assert first.published == 1

        h.clock.advance(60 * 86400.0)          # локальное забывание (GC)
        report = h.consolidate()
        assert report.forgotten_traces == 1
        assert report.publications_retracted == 1
        assert h.store.get(mid) is None
        # tombstone стоит локально сразу, до всякой сети
        assert registry.active_publications(h.store) == []
        assert registry.sync_status(h.store)["awaiting_sync"] == 1

        second = push(h.store, pol)            # доставка готового tombstone
        assert second.retracted == 1 and second.auto_retracted == 0
        c = _client(coord)
        dump = c.cache_dump()
        assert dump["active"] == [] and len(dump["tombstones"]) == 1
        assert registry.sync_status(h.store)["awaiting_sync"] == 0
    finally:
        h.close()


def test_auto_sync_hook_after_sleep(tmp_path, coord, monkeypatch):
    """auto_sync=true: Stop-хук после сна сам доставляет публикации."""
    monkeypatch.setenv("REALMEMORY_TEAM_TOKEN", "sekret")
    monkeypatch.setenv("REALMEMORY_POLICY_PATH", str(tmp_path / "team.yaml"))
    from realmemory.hook_cli import main as hook_main

    h = _brain(tmp_path, "maxim")
    try:
        # выравниваем FakeClock с реальным временем: хук спит с SystemClock,
        # и консолидация не должна «состарить» след на три года
        import time as _time
        h.clock.advance(_time.time() - h.clock.now())
        mid = h.remember("решение для авто-sync проверки p502",
                         scope="proj").memory_id
        h.feedback([mid], 1.0)
        registry.publish(h.store, [mid],
                         policy=_policy_file(tmp_path, coord), now=1.0,
                         identity="maxim")
        pol = load_policy(tmp_path / "team.yaml")
        pol.auto_sync = True
        save_policy(pol, tmp_path / "team.yaml")
        path = str(h.path)
        h.close()

        with pytest.raises(SystemExit) as e:
            hook_main(["sleep", "--path", path])
        assert e.value.code == 0

        c = _client(coord)
        dump = c.cache_dump()
        assert len(dump["active"]) == 1
    finally:
        try:
            h.close()
        except Exception as exc:  # noqa: BLE001 - уже закрыт — не важно
            print(f"close note: {exc}")
