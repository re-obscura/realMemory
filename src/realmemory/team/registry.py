"""Registry публикаций: локальный журнал «что я решил показать команде».

Публикация и отзыв — записи в таблицу publications собственной базы.
Сетевая доставка (push к координатору, раздача tombstones) — отдельный
будущий шаг; до его появления registry честно ведёт счётчик записей,
ожидающих синхронизации, но никакие байты не покидают машину.

Правило авторства: публиковать можно свои следы (rec.author пуст для старых
записей — разрешается) либо совпадающие с identity. Чужая атрибуция — отказ.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..types import STATUS_ACTIVE, MemoryRecord


class RegistryError(Exception):
    """Отказ операции реестра с человекочитаемой причиной."""


@dataclass(frozen=True)
class PublicationRow:
    publication_id: str
    trace_id: int
    project: str
    author: str
    published_at: float
    content_hash: str = ""
    revoked_at: float | None = None


def _row_to_pub(row: tuple, with_revoked: bool = False) -> PublicationRow:
    if with_revoked:
        pid, tid, proj, auth, pub_at, rev_at = row
        return PublicationRow(str(pid), int(tid), str(proj), str(auth),
                              float(pub_at),
                              revoked_at=None if rev_at is None
                              else float(rev_at))
    pid, tid, proj, auth, pub_at, chash = row
    return PublicationRow(str(pid), int(tid), str(proj), str(auth),
                          float(pub_at), str(chash))


def publish(store, memory_ids, *, policy, now: float,
            identity: str | None = None) -> list[PublicationRow]:
    """Зарегистрировать публикацию следов. Возвращает созданные строки.

    Fail-closed по never-правилам политики: нарушающие кандидаты отклоняются
    целиком пакетом (ничего не публикуем частично «через один»).
    """
    from .policy import BLOCKED_NEVER, classify  # локальный импорт — циклы

    who = policy.identity or identity or ""
    ids = sorted({int(i) for i in memory_ids})
    if not ids:
        return []
    records: list[MemoryRecord] = []
    for mid in ids:
        rec = store.get(mid)
        if rec is None:
            raise RegistryError(f"след {mid} не существует")
        decision = classify(rec, policy)
        if decision.status == BLOCKED_NEVER:
            raise RegistryError(
                f"след {mid} запрещён never-правилами ({decision.reason}); "
                "публикация пакета отменена целиком")
        records.append(rec)

    created: list[PublicationRow] = []
    for rec in records:
        tid = int(rec.id or 0)
        if rec.author and who and rec.author != who:
            raise RegistryError(
                f"след {tid} принадлежит «{rec.author}» — публиковать "
                "чужую атрибуцию нельзя")
        digest = hashlib.sha256((rec.text or "").encode("utf-8")).hexdigest()[:16]
        pid = store.publication_add(tid, rec.scope, who or rec.author,
                                    float(now), digest)
        created.append(PublicationRow(pid, tid, rec.scope,
                                      who or rec.author, float(now), digest))
    store.event_append("team_publish", {
        "count": len(created),
        "ids": [p.publication_id for p in created],
        "identity": who,
    })
    return created


def retract(store, *, now: float, publication_ids=(), trace_ids=()) -> int:
    """Поставить tombstones на активные публикации."""
    touched = store.publication_retract(float(now),
                                        publication_ids=list(publication_ids or ()),
                                        trace_ids=list(trace_ids or ()))
    if touched:
        store.event_append("team_retract", {"touched": touched})
    return touched


def active_publications(store) -> list[PublicationRow]:
    return [_row_to_pub(r) for r in store.publications_active()]

def tombstoned_publications(store) -> list[PublicationRow]:
    return [_row_to_pub(r, with_revoked=True) for r in store.publications_tombstones()]


def sync_status(store) -> dict:
    """Наблюдаемость registry: awaiting_sync — решения без подтверждения
    сети; доставленные остаются в истории (active/tombstones)."""
    unsynced = store.publications_unsynced()
    return {
        "active": len(active_publications(store)),
        "tombstones": len(tombstoned_publications(store)),
        "awaiting_sync": len(unsynced),
    }


def stats_line(store) -> str:  # pragma: no cover - рендер-хелпер CLI
    st = sync_status(store)
    return (f"опубликовано {st['active']}, отозвано {st['tombstones']}, "
            f"ждёт синхронизации {st['awaiting_sync']}")


assert STATUS_ACTIVE == "active"  # контракт статуса, на который опираемся
