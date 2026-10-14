"""Однократный перенос наследия v0.3 (journal.jsonl, snapshot.pkl) в БД.

Выделено из фасада, чтобы pickle и правила разбора старых форматов не жили
в основном файле. Каждый импорт маркер-гейтован в db_meta и выполняется
один раз; битый снапшот не блокирует открытие базы — ошибка попадает
в журнал (legacy_snapshot_imported).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .store.sqlite_store import MemoryStore
from .timeprov import TimeProvider

_LEGACY_SNAPSHOT = "snapshot.pkl"
_LEGACY_JOURNAL = "journal.jsonl"


def migrate_legacy_files(root: Path, store: MemoryStore, clock: TimeProvider) -> None:
    """Однократный перенос наследия v0.3: журнал событий и снапшот сети."""
    jp = root / _LEGACY_JOURNAL
    if jp.exists() and store.get_meta("journal_imported") is None:
        imported = 0
        for line in jp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = str(event.pop("type", "unknown"))
            ts = float(event.pop("ts", event.pop("t", 0.0)))
            if etype == "write":
                # до v0.4 журналировались только вставки (гейт CREATE)
                event.setdefault("action", "create")
            store.event_append(etype, event, ts=ts)
            imported += 1
        store.set_meta("journal_imported", "1")
        jp.rename(jp.with_name(jp.name + ".imported"))
        store.event_append("legacy_journal_imported",
                           {"events": imported}, ts=float(clock.now()))

    sp = root / _LEGACY_SNAPSHOT
    if sp.exists() and store.get_meta("snapshot_imported") is None:
        try:
            with sp.open("rb") as f:
                payload = pickle.load(f)
            net = payload.get("network", {})
            keys = np.asarray(net.get("keys"), dtype=np.int64).reshape(-1)
            ws = np.asarray(net.get("weights"), dtype=np.float32).reshape(-1)
            last_tick = net.get("last_tick")
            store.edges_import(keys, ws, None if last_tick is None else float(last_tick))
            for src, dst, strength, created_at, source_ids in (
                payload.get("eligibility", {}).get("events", [])
            ):
                store.elig_add(np.asarray(src, dtype=np.int32),
                               np.asarray(dst, dtype=np.int32),
                               float(strength), float(created_at),
                               [int(i) for i in source_ids])
            status: dict[str, Any] = {"edges": int(keys.size)}
        except Exception as exc:  # noqa: BLE001 - битый снапшот не должен блокировать базу
            status = {"error": str(exc)}
        store.set_meta("snapshot_imported", "1")
        sp.rename(sp.with_name(sp.name + ".imported"))
        store.event_append("legacy_snapshot_imported", status,
                           ts=float(clock.now()))
