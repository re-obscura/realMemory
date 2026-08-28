"""Командный recall: live peer-to-peer с фоллбэком на кэш координатора.

Порядок честности ответа:
  1. presence координатора → кто online и по какому адресу;
  2. живой запрос каждому peer-у с КОРОТКИМ таймаутом: если коллега не
     отвечает — молча падаем на кэш, неудачи собираются в peers_failed;
  3. кэш координатора добирает всё, что не покрыто живыми ответами.
Дедупликация по (author, text); живое всегда приоритетнее кэша.
max_age_s — верхняя граница устаревания ХУДШЕГО попадания выдачи.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .policy import TeamPolicy, load_policy
from .sync import make_client
from .transport import CoordinatorClient, CoordinatorError, EmbedderMismatch, encode_vector

LIVE_TIMEOUT_S = 1.5
MAX_LIVE_PEERS = 6


@dataclass(frozen=True)
class TeamHit:
    publication_id: str = ""
    trace_id: int | None = None
    text: str = ""
    author: str = ""
    project: str = ""
    score: float = 0.0
    published_at: float = 0.0
    source: str = "cache"        # live | cache
    peer: str = ""               # identity живого источника


@dataclass(frozen=True)
class TeamAnswer:
    hits: tuple[TeamHit, ...] = ()
    abstained: bool = True
    max_age_s: float | None = None      # устаревание худшего попадания
    coordinator: str = ""
    presence_online: list[str] = field(default_factory=list)
    peers_live: list[str] = field(default_factory=list)
    peers_failed: list[str] = field(default_factory=list)


def _brain_meta(root_path) -> tuple[str, int]:
    """(имя эмбеддера базы, dim); пустое имя для несуществующего мозга."""
    db = Path(root_path) / "memory.db"
    if not db.exists():
        return "", 256
    con = sqlite3.connect(str(db))
    try:
        emb = con.execute(
            "SELECT value FROM db_meta WHERE key='embedder'").fetchone()
        row = con.execute(
            "SELECT length(embedding) FROM memories LIMIT 1").fetchone()
    finally:
        con.close()
    name = str(emb[0]) if emb else ""
    return name, (int(row[0]) // 4 if row else 256)


def embed_query_text(root_path, text: str) -> tuple[object, str]:
    """Запрос кодируется ТЕМ ЖЕ эмбеддером, которым писался локальный мозг:
    имя берём из db_meta (оно же маркер собственных публикаций). Несовпадение
    с чужими записями ловит получатель (peer/координатор) честным 409."""
    name, dim = _brain_meta(root_path)
    if name.startswith("fastembed:"):
        from ..encoding.embedder_local import FastEmbedProvider

        provider = FastEmbedProvider()
        return provider.embed_query(text), provider.name
    from ..encoding.embedder import HashingEmbedder

    embedder = HashingEmbedder(dim=dim)
    return embedder.embed(text), (name or f"hashing(dim={dim})")


def _query_live_peers(presence, qvec, k, local_name, author, project,
                      token, results: dict, failed: list[str]) -> None:
    """Опросить живых коллег (кроме себя) с коротким таймаутом."""
    targets = []
    for p in presence:
        if not p.get("online") or not p.get("address"):
            continue
        ident = p["identity"]
        if author is not None and ident != author:
            continue
        targets.append((ident, p["address"]))
    for ident, address in targets[:MAX_LIVE_PEERS]:
        peer_client = CoordinatorClient(f"http://{address}", token=token,
                                        timeout_s=LIVE_TIMEOUT_S)
        try:
            out = peer_client.raw_post("/recall", {
                "query_embedding_b64": encode_vector(qvec),
                "k": k, "embedder": local_name, "project": project,
            })
            for h in out.get("hits", []):
                key = (h.get("author", ""), h.get("text", ""))
                results[key] = TeamHit(
                    trace_id=int(h.get("trace_id") or 0),
                    text=h.get("text", ""), author=h.get("author", ""),
                    project=h.get("project", ""),
                    score=float(h.get("score", 0.0)),
                    published_at=float(h.get("published_at", 0.0)),
                    source="live", peer=ident)
        except EmbedderMismatch as exc:
            failed.append(f"{ident}: {exc}")
        except CoordinatorError as exc:
            failed.append(f"{ident}: {exc}")


def recall_team(root_path, query: str, *, k: int = 5,
                author: str | None = None, project: str | None = None,
                policy: TeamPolicy | None = None,
                policy_path=None) -> TeamAnswer:
    policy = policy or load_policy(policy_path)
    client = make_client(policy)

    _, _brain_dim = _brain_meta(root_path)
    del _brain_dim
    qvec, local_provider_name = embed_query_text(root_path, query)

    token = os.environ.get(policy.token_env or "", "").strip() or None

    presence: list[dict] = []
    online: list[str] = []
    try:
        presence = client.presence()
        online = [p["identity"] for p in presence if p.get("online")]
    except CoordinatorError:
        presence = []  # координатор недоступен: live-опрос невозможен

    live: dict[tuple[str, str], TeamHit] = {}
    failed: list[str] = []
    _query_live_peers(presence, qvec, k, local_provider_name, author,
                      project, token, live, failed)

    cache_hits = client.search(qvec, k=k, embedder=local_provider_name,
                               author=author, project=project)
    merged: dict[tuple[str, str], TeamHit] = dict(live)
    for h in cache_hits:
        key = (h.get("author", ""), h.get("text", ""))
        if key not in merged:
            merged[key] = TeamHit(
                publication_id=h["publication_id"], text=h.get("text", ""),
                author=h.get("author", ""), project=h.get("project", ""),
                score=float(h.get("score", 0.0)),
                published_at=float(h.get("published_at", 0.0)),
                source="cache")

    ordered = sorted(merged.values(),
                     key=lambda h: (0 if h.source == "live" else 1,
                                    -h.score))[:k]
    now = time.time()
    ages = [now - h.published_at for h in ordered
            if h.source == "cache" and h.published_at]
    return TeamAnswer(
        hits=tuple(ordered), abstained=not ordered,
        max_age_s=max(ages) if ages else None,
        coordinator=policy.coordinator or "",
        presence_online=online,
        peers_live=sorted({h.peer for h in ordered if h.source == "live"}),
        peers_failed=failed)
