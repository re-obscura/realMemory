"""Командный recall: поиск по кэшу опубликованного через координатора.

v0 намеренно ТОЛЬКО cached-канал: живой опрос инстансов коллег требует
сетевого входа на каждой машине — отдельный следующий этап (см. docs/TEAM.md).
Ответ честно помечает возраст данных (published_at самой свежей записи выдачи)
и падает понятной ошибкой при несовпадении эмбеддера команды.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .policy import TeamPolicy, load_policy
from .sync import make_client


@dataclass(frozen=True)
class TeamHit:
    publication_id: str
    text: str
    author: str
    project: str
    score: float
    published_at: float


@dataclass(frozen=True)
class TeamAnswer:
    hits: tuple[TeamHit, ...] = ()
    abstained: bool = True
    max_age_s: float | None = None     # возраст самого свежего попадания
    coordinator: str = ""
    presence_online: list[str] = field(default_factory=list)


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


def embed_query_text(text: str) -> tuple[object, str]:
    """Запрос кодируется боевым локальным эмбеддером (fastembed).

    Пarity проверяет координатор: имя модели запрашивающего сравнивается с
    именами в кэше, несовпадение = понятный отказ вместо мусорного косинуса."""
    from ..encoding.embedder_local import FastEmbedProvider

    provider = FastEmbedProvider()
    return provider.embed_query(text), provider.name


def recall_team(root_path, query: str, *, k: int = 5,
                author: str | None = None, project: str | None = None,
                policy: TeamPolicy | None = None,
                policy_path=None) -> TeamAnswer:
    policy = policy or load_policy(policy_path)
    client = make_client(policy)

    brain_name, brain_dim = _brain_meta(root_path)
    qvec, local_name = embed_query_text(text=query)
    # локальный мозг уже писал другим эмбеддером — предупредим заранее,
    # пока координатор не ответил своей более точной диагностикой
    if brain_name and local_name and brain_name != local_name:
        raise RuntimeError(
            f"локальная база писалась эмбеддером {brain_name}, а запрос "
            f"кодируется {local_name}; установите ту же модель/версию")

    hits = client.search(qvec, k=k, embedder=local_name or brain_name,
                         author=author, project=project)
    del brain_dim  # информация для будущего серверного валидатора размерности

    now = time.time()
    team_hits = tuple(
        TeamHit(publication_id=h["publication_id"], text=h["text"],
                author=h["author"], project=h["project"],
                score=float(h["score"]), published_at=float(h["published_at"]))
        for h in hits)
    ages = [now - h.published_at for h in team_hits]
    online: list[str] = []
    try:
        online = [p["identity"] for p in client.presence() if p.get("online")]
    except Exception:  # noqa: BLE001 - presence — украшение ответа, не критерий
        online = []
    return TeamAnswer(hits=team_hits, abstained=not team_hits,
                      max_age_s=max(ages) if ages else None,
                      coordinator=policy.coordinator or "",
                      presence_online=online)
