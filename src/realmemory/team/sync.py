"""Синхронизация registry с координатором: явный сетевой акт.

`push` читает из локального registry всё, что ещё не подтверждено сетью
(активные публикации И tombstones), отправляет двумя вызовами (/publish,
/retract) и помечает строки synced_at только после успеха. Падение сети в
середине не теряет решений: несинхронизированное доедет при следующем `sync`.

Публикации, чей след уже удалён GC-ем локально, отзываются автоматически
(tombstone не требует контента): автор больше не «стоит» за забытым фактом,
значит команда не должна его видеть. CLI честно считает такие отзывы.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

from ..store.sqlite_store import MemoryStore
from .policy import TeamPolicy
from .transport import CoordinatorClient, encode_vector


@dataclass
class SyncSummary:
    published: int = 0
    retracted: int = 0
    marked: int = 0
    auto_retracted: int = 0   # отзывы забытых локально следов (GC)


def embedder_name(store: MemoryStore) -> str:
    """Имя эмбеддера базы (тот же маркер, что хранит embedder identity)."""
    return store.get_meta("embedder") or "unknown"


def make_client(policy: TeamPolicy, timeout_s: float = 4.0) -> CoordinatorClient:
    if not policy.coordinator:
        raise RuntimeError(
            "координатор не настроен в team.yaml (ключ coordinator): "
            "сетевая синхронизация отключена — это kill-switch по умолчанию")
    token = os.environ.get(policy.token_env or "", "").strip() or None
    return CoordinatorClient(policy.coordinator, token=token,
                             timeout_s=timeout_s)


def push(store: MemoryStore, policy: TeamPolicy, *,
         timeout_s: float = 4.0) -> SyncSummary:
    client = make_client(policy, timeout_s=timeout_s)
    client.health()  # ранний ясный отказ вместо тихой половинчатой рассылки

    summary = SyncSummary()
    publish_items: list[dict] = []
    tombstones: list[dict] = []
    to_mark: list[str] = []

    emb_name = embedder_name(store)
    for (pid, trace_id, project, author, published_at, revoked_at,
         content_hash) in store.publications_unsynced():
        if revoked_at is not None:
            tombstones.append({"publication_id": pid,
                               "revoked_at": float(revoked_at)})
            to_mark.append(pid)
            continue
        rec = store.get(int(trace_id))
        if rec is None:
            # локальное забывание (GC) уже случилось: отзыв команды обязателен,
            # контент для него не нужен
            tombstones.append({"publication_id": pid,
                               "revoked_at": time.time()})
            to_mark.append(pid)
            summary.auto_retracted += 1
            continue
        text_bytes = (rec.text or "").encode("utf-8")
        digest = hashlib.sha256(text_bytes).hexdigest()[:16]
        publish_items.append({
            "publication_id": pid,
            "project": project,
            "author": author or rec.author or "",
            "text": rec.text,
            "embedding_b64": encode_vector(rec.embedding),
            "published_at": float(published_at),
            # локальный content_hash мог быть пуст у legacy-строк — досчитаем
            "content_hash": content_hash or digest,
            "embedder": emb_name,
        })
        to_mark.append(pid)

    # GC-автоотзыв: доставленные публикации, чей след локально забыт.
    # Автор больше не «стоит» за фактом — команда не должна его видеть.
    force_mark: list[str] = []
    for (pid, trace_id, _project, _author, _published_at, _revoked,
         _chash) in store.publications_synced_active():
        if store.get(int(trace_id)) is None:
            tombstones.append({"publication_id": pid,
                               "revoked_at": time.time()})
            force_mark.append(pid)
            summary.auto_retracted += 1

    if publish_items:
        summary.published = client.publish_batch(publish_items)
    if tombstones:
        summary.retracted = client.retract_batch(tombstones)
    if to_mark and (publish_items or tombstones):
        summary.marked += store.publication_mark_synced(to_mark, time.time())
    if force_mark:
        summary.marked += store.publication_mark_synced_force(force_mark,
                                                              time.time())
    return summary
