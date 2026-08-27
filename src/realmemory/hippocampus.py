"""Hippocampus — фасад realMemory («гиппокамп»).

Связывает кодирование (эмбеддер -> биполярный адрес + SDR), L1-адресацию,
L2-ассоциации, политики новизны/затухания и SQLite-хранилище в единые пути
remember/recall/feedback/consolidate. Контракт: docs/CONTRACTS.md.

Все мутабельное состояние (рёбра L2, eligibility-лог, журнал, счётчики) живёт
в SQLite и пишется сквозь — несколько процессов (MCP-сервер + хуки) работают
с одной базой без потери состояния. При открытии перестраиваются только
производные структуры: L1-бакеты и юнит-индекс — из БД детерминированно,
CSR-кэш рёбер L2 — из таблицы edges, кэш эмбеддингов точного скана — из блобов.

Ретривер recall по умолчанию — точный косинус-скан по кэшу (exact_scan_recall);
голосование L1 включается автоматически при превышении exact_scan_max_traces.
"""
from __future__ import annotations

import dataclasses
import json
import pickle
import re
import time as _time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np

from .config import MemoryConfig
from .core.addressing import SDRVotingIndex
from .core.assembly import AssemblyNetwork
from .core.plasticity import EligibilityLog
from .encoding.embedder import EmbeddingProvider, HashingEmbedder
from .encoding.sdr import SDREncoder
from .policies.decay import reinforce_values, retention, should_promote, weaken_value
from .policies.novelty import gate
from .store.sqlite_store import MemoryStore, build_fts_query, search_tokens
from .timeprov import SystemClock, TimeProvider
from .types import (
    KIND_EPISODIC,
    KIND_SEMANTIC,
    SCOPE_GLOBAL,
    SOURCE_ASSOCIATED,
    SOURCE_DIRECT,
    SOURCE_KEYWORD,
    STATUS_ACTIVE,
    ConsolidationReport,
    DecisionAction,
    MemoryRecord,
    RecalledMemory,
    RecallPacket,
    WriteDecision,
    WriteResult,
)

_DB_NAME = "memory.db"
_LEGACY_SNAPSHOT = "snapshot.pkl"
_LEGACY_JOURNAL = "journal.jsonl"

_NAMESPACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _resolve_root(path: str | Path, namespace: str | None) -> Path:
    """Каталог хранилища; namespace изолирует проекты внутри одного корня."""
    root = Path(path)
    if namespace is None:
        return root
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(
            "namespace: 1–64 символа из [A-Za-z0-9_.-], первый — буква или цифра"
        )
    return root / namespace


class _EventLog:
    """Фасад над таблицей events: прежний вызов append(event_type, **fields)."""

    def __init__(self, store: MemoryStore, clock: TimeProvider) -> None:
        self._store = store
        self._clock = clock

    def append(self, event_type: str, **fields: Any) -> None:
        self._store.event_append(event_type, fields, ts=float(self._clock.now()))

    def count(self) -> int:
        return self._store.event_count()


class Hippocampus:
    def __init__(
        self,
        path: str | Path = "rm_data",
        *,
        config: MemoryConfig | None = None,
        embedder: EmbeddingProvider | None = None,
        clock: TimeProvider | None = None,
        namespace: str | None = None,
        verify_embedder: bool = True,
        author: str = "",
    ) -> None:
        self.config = config or MemoryConfig.dev()
        self.config.validate()
        self.path = _resolve_root(path, namespace)
        self.path.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemClock()
        # кто записывает через этот инстанс; попадает в каждую новую запись
        # (командный слой, атрибуция). Пусто для безличных/тестовых сценариев.
        self.author = str(author or "").strip()
        self.embedder = embedder or HashingEmbedder(dim=self.config.dim)
        if self.embedder.dim != self.config.dim:
            raise ValueError(
                f"dim эмбеддера ({self.embedder.dim}) не совпадает с config.dim ({self.config.dim})"
            )
        self.store = MemoryStore(self.path / _DB_NAME, self.config.dim)
        if verify_embedder:
            # False для инструментов без эмбеддингов (хуки сна/брифа): они не
            # порождают векторов и не должны спорить с боевым эмбеддером базы
            self._check_embedder_identity()
        self._check_db_config()
        self._migrate_legacy_files()
        self.sdr_encoder = SDREncoder(
            self.config.dim, self.config.n_units, self.config.k_sparse, self.config.sdr_seed
        )
        self.index = SDRVotingIndex(self.config.n_units, bucket_cap=self.config.bucket_cap)
        # network — чистый in-memory CSR-кэш таблицы edges; истина в БД
        self.network = AssemblyNetwork(
            self.config.n_units,
            edge_min_weight=self.config.edge_min_weight,
            tau_edge_stable=self.config.tau_edge_stable,
            seed=self.config.sdr_seed + 1,
            max_pairs_per_bind=self.config.max_pairs_per_bind,
        )
        self.journal = _EventLog(self.store, self.clock)
        self._rng = np.random.default_rng(self.config.sdr_seed + 2)
        self._unit_index: dict[int, set[int]] = {}
        self._edges_rev_seen = -1
        self._trace_count = self.store.count()
        # кэш эмбеддингов активных следов (точный скан): амортизированные буферы,
        # растут вдвое — вставка не копирует весь массив на каждом remember
        self._emb_cap = 0
        self._emb_len = 0
        self._emb_buf = np.zeros((0, self.config.dim), dtype=np.float32)
        self._idbuf = np.zeros(0, dtype=np.int64)
        self._emb_pos: dict[int, int] = {}
        self._rebuild_volatile()

    # -- конструирование ---------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        config: MemoryConfig | None = None,
        embedder: EmbeddingProvider | None = None,
        clock: TimeProvider | None = None,
        namespace: str | None = None,
        verify_embedder: bool = True,
        author: str = "",
    ) -> Hippocampus:
        return cls(path, config=config, embedder=embedder, clock=clock,
                   namespace=namespace, verify_embedder=verify_embedder,
                   author=author)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- внутренние помощники ------------------------------------------------------

    def _check_embedder_identity(self) -> None:
        """База помнит, каким эмбеддером писалась: векторы разных эмбеддеров
        несравнимы по косинусу, смешение тихо ломает recall."""
        stored = self.store.get_meta("embedder")
        if stored is None:
            if self.store.count() > 0:
                # база до введения маркировки — принимаем текущий эмбеддер как есть
                self.store.event_append(
                    "embedder_identity_adopted", {"name": self.embedder.name},
                    ts=float(self.clock.now()),
                )
            self.store.set_meta("embedder", self.embedder.name)
            return
        if stored != self.embedder.name:
            raise RuntimeError(
                f"эмбеддер базы {self.path} — '{stored}', а открывается с "
                f"'{self.embedder.name}'. Старые и новые эмбеддинги несравнимы; "
                "откройте базу исходным эмбеддером или начните новую директорию памяти."
            )

    def _check_db_config(self) -> None:
        """Геометрия SDR/размерность фиксируются в базе при первом открытии;
        открытие с другой геометрией дало бы невалидные бакеты и рёбра."""
        stored = self.store.get_meta("config")
        mine = json.dumps(self.config.snapshot_fields(), sort_keys=True)
        if stored is None:
            self.store.set_meta("config", mine)
            return
        try:
            other = json.loads(stored)
        except ValueError:
            other = {}
        for key in ("dim", "n_units", "sdr_seed"):
            if key in other and other[key] != getattr(self.config, key):
                raise RuntimeError(
                    f"конфиг не совпадает с базой {self.path} по полю {key}: "
                    f"в базе {other[key]}, открывается с {getattr(self.config, key)}"
                )

    def _migrate_legacy_files(self) -> None:
        """Однократный перенос наследия v0.3 (journal.jsonl, snapshot.pkl) в БД."""
        jp = self.path / _LEGACY_JOURNAL
        if jp.exists() and self.store.get_meta("journal_imported") is None:
            imported = 0
            for line in jp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                etype = str(event.pop("type", "unknown"))
                ts = float(event.pop("ts", event.pop("t", 0.0)))
                if etype == "write":
                    # до v0.4 журналировались только вставки (гейт CREATE)
                    event.setdefault("action", "create")
                self.store.event_append(etype, event, ts=ts)
                imported += 1
            self.store.set_meta("journal_imported", "1")
            jp.rename(jp.with_name(jp.name + ".imported"))
            self.store.event_append("legacy_journal_imported",
                                    {"events": imported}, ts=float(self.clock.now()))

        sp = self.path / _LEGACY_SNAPSHOT
        if sp.exists() and self.store.get_meta("snapshot_imported") is None:
            try:
                with sp.open("rb") as f:
                    payload = pickle.load(f)
                net = payload.get("network", {})
                keys = np.asarray(net.get("keys"), dtype=np.int64).reshape(-1)
                ws = np.asarray(net.get("weights"), dtype=np.float32).reshape(-1)
                last_tick = net.get("last_tick")
                self.store.edges_import(keys, ws, None if last_tick is None else float(last_tick))
                for src, dst, strength, created_at, source_ids in (
                    payload.get("eligibility", {}).get("events", [])
                ):
                    self.store.elig_add(np.asarray(src, dtype=np.int32),
                                        np.asarray(dst, dtype=np.int32),
                                        float(strength), float(created_at),
                                        [int(i) for i in source_ids])
                status: dict[str, Any] = {"edges": int(keys.size)}
            except Exception as exc:  # noqa: BLE001 - битый снапшот не должен блокировать базу
                status = {"error": str(exc)}
            self.store.set_meta("snapshot_imported", "1")
            sp.rename(sp.with_name(sp.name + ".imported"))
            self.store.event_append("legacy_snapshot_imported", status,
                                    ts=float(self.clock.now()))

    def _rebuild_volatile(self) -> None:
        """Производные структуры при открытии: L1-бакеты, юнит-индекс, кэш
        эмбеддингов, CSR рёбер."""
        for rec in self.store.iter_active():
            self.index.write(rec.sdr, int(rec.id))
            for u in rec.sdr.tolist():
                self._unit_index.setdefault(int(u), set()).add(int(rec.id))
            self._cache_append(int(rec.id), rec.embedding)
        self._reload_network_cache()

    # -- кэш эмбеддингов (точный скан) ------------------------------------------

    def _grow_emb_cache(self, needed: int) -> None:
        if needed <= self._emb_cap:
            return
        new_cap = max(needed, max(64, self._emb_cap * 2))
        emb = np.zeros((new_cap, self.config.dim), dtype=np.float32)
        ids = np.zeros(new_cap, dtype=np.int64)
        if self._emb_len:
            emb[: self._emb_len] = self._emb_buf[: self._emb_len]
            ids[: self._emb_len] = self._idbuf[: self._emb_len]
        self._emb_buf, self._idbuf, self._emb_cap = emb, ids, new_cap

    def _cache_append(self, memory_id: int, emb: np.ndarray) -> None:
        """Добавить след в кэш точного скана. Superseded-следы остаются в буфере
        до перезапуска процесса — фильтруются по статусу после чтения записи."""
        self._grow_emb_cache(self._emb_len + 1)
        self._emb_buf[self._emb_len] = emb
        self._idbuf[self._emb_len] = int(memory_id)
        self._emb_pos[int(memory_id)] = self._emb_len
        self._emb_len += 1

    def _exact_mode(self) -> bool:
        return (
            bool(getattr(self.config, "exact_scan_recall", False))
            and self._trace_count <= self.config.exact_scan_max_traces
        )

    def _cosine_ranked(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(id, косинус) всех закэшированных следов по убыванию похожести.
        Нулевой запрос даёт нули — без распада на деление."""
        qn = float(np.linalg.norm(emb))
        qv = (emb / np.float32(qn)).astype(np.float32) if qn > 0.0 else emb.astype(np.float32)
        view = self._emb_buf[: self._emb_len]
        sims = view @ qv if self._emb_len else np.empty(0, dtype=np.float32)
        order = np.argsort(-sims, kind="stable")
        return self._idbuf[: self._emb_len][order], sims[order]

    def _reload_network_cache(self) -> None:
        keys, ws = self.store.edges_load()
        tick = self.store.get_meta("last_edge_tick")
        self.network.load_state_dict({
            "keys": keys,
            "weights": ws,
            "last_tick": float(tick) if tick is not None else None,
        })
        self._edges_rev_seen = self.store.edges_rev()

    def _sync_network_cache(self) -> None:
        """Чужие процессы могли доконсолидировать рёбра — обновляем кэш по версии."""
        if self.store.edges_rev() != self._edges_rev_seen:
            self._reload_network_cache()

    def _candidate_budget(self, base: int, *, divisor: int = 64, cap: int = 1500) -> int:
        """Бюджет кандидатов L1 растёт с размером базы: шум голосования
        («конкуренты» с одним общим словом) растёт с корпусом, и фиксированный
        k·oversample на десятках тысяч следов вытесняет цели со слабым
        пересечением юнитов. Пол = traces/divisor с потолком cap."""
        floor = min(cap, self._trace_count // divisor)
        return max(base, floor)

    def _encode(self, text: str, *, query: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Кодирование текста. Для поисковых запросов используется embed_query(),
        если эмбеддер его предоставляет (асимметричные модели вроде e5)."""
        if query:
            qfn = getattr(self.embedder, "embed_query", None)
            if qfn is not None:
                emb = np.asarray(qfn(text), dtype=np.float32)
                return emb, self.sdr_encoder.encode(emb)
        emb = self.embedder.embed(text)
        sdr = self.sdr_encoder.encode(emb)
        return emb, sdr

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _fts_candidates(self, text: str, limit: int) -> list[int]:
        """ID следа с точным совпадением токенов (keyword-канал FTS5)."""
        if not self.store.fts_enabled:
            return []
        expr = build_fts_query(text)
        if expr is None:
            return []
        return [rid for rid, _ in self.store.fts_match(expr, limit=limit)]

    def _probe(self, emb: np.ndarray, sdr: np.ndarray,
               scope: str | None = None,
               text: str | None = None) -> tuple[int | None, float, tuple[int, ...]]:
        """Лучший существующий след по косинусу среди кандидатов + FTS.
        Гейт сравнивает текст только со своим проектом и global — факты разных
        проектов не сливаются в REINFORCE/LINK; keyword-кандидаты ловят
        почти-дубликаты с точными токенами (ID, коды ошибок).

        Точный режим сканирует глобальный рейтинг косинусов и останавливается
        рано: как только набраны best и `near`, а следующий по списку косинус
        опустился ниже theta_link, дальше подходящих нет (рейтинг убывающий).
        Режим голосования сохранён для больших баз и сравнения режимов."""
        near_ids: list[tuple[float, int]] = []
        if self._exact_mode():
            ranked_ids, ranked_cos = self._cosine_ranked(emb)
            best_id: int | None = None
            best_cos = 0.0
            if ranked_ids.size:
                # интересуют только следы с косинусом >= theta_link; в убывающем
                # рейтинге это префикс известной длины — глубже смотреть незачем
                cnt_link = int(np.searchsorted(-ranked_cos, -self.config.theta_link,
                                               side="right"))
                budget = self._candidate_budget(
                    max(2, self.config.recall_oversample * 3), divisor=256,
                )
                probe_ids = [int(i) for i in ranked_ids[: min(cnt_link, budget)].tolist()]
                recs = self.store.get_many(probe_ids)
                if recs:
                    qn = float(np.linalg.norm(emb))
                    qv = (emb / np.float32(qn)).astype(np.float32) if qn else np.zeros_like(emb)
                    sims = np.stack([r.embedding for r in recs]) @ qv
                    for rec, sim in zip(recs, sims.tolist()):
                        if rec.status != STATUS_ACTIVE:
                            continue
                        if not self._scope_allows(rec.scope, scope, all_scopes=False):
                            continue
                        cos = max(0.0, sim)
                        if best_id is None:
                            best_id, best_cos = int(rec.id), cos
                        elif cos >= self.config.theta_link and len(near_ids) < 4:
                            near_ids.append((cos, int(rec.id)))
            return best_id, best_cos, tuple(i for _, i in sorted(near_ids, key=lambda t: -t[0]))

        qr = self.index.query(
            sdr,
            max_candidates=self._candidate_budget(
                max(2, self.config.recall_oversample * 3), divisor=256,
            ),
        )
        candidate_ids = list(qr.candidates.tolist())
        for rid in self._fts_candidates(
            text or "", limit=max(4, self.config.recall_oversample * 3)
        ):
            if rid not in candidate_ids:
                candidate_ids.append(rid)
        if not candidate_ids:
            return None, 0.0, ()
        scored: list[tuple[int, float]] = []
        recs = self.store.get_many(candidate_ids)
        if recs:
            qn = float(np.linalg.norm(emb))
            qv = (emb / np.float32(qn)).astype(np.float32) if qn else np.zeros_like(emb)
            sims = np.stack([r.embedding for r in recs]) @ qv
            for rec, sim in zip(recs, sims.tolist()):
                if rec.status != STATUS_ACTIVE:
                    continue
                if not self._scope_allows(rec.scope, scope, all_scopes=False):
                    continue
                scored.append((int(rec.id), max(0.0, sim)))
        if not scored:
            return None, 0.0, ()
        scored.sort(key=lambda t: -t[1])
        best_id, best_cos = scored[0]
        near = tuple(i for i, c in scored[1:] if c >= self.config.theta_link)[:4]
        return best_id, best_cos, near

    @staticmethod
    def _scope_allows(rec_scope: str, scope: str | None, all_scopes: bool) -> bool:
        """scope=None или all_scopes — без фильтра; иначе свой проект + global."""
        if all_scopes or scope is None:
            return True
        return rec_scope == scope or rec_scope == SCOPE_GLOBAL

    def _insert_memory(self, text, kind, meta, now, emb, sdr,
                       scope: str = SCOPE_GLOBAL, author: str | None = None) -> int:
        rec = MemoryRecord(
            id=None,
            text=text,
            kind=kind,
            status=STATUS_ACTIVE,
            meta=dict(meta or {}),
            embedding=np.asarray(emb, dtype=np.float32),
            sdr=np.asarray(sdr, dtype=np.int32),
            created_at=now,
            updated_at=now,
            reinforced_count=0,
            last_reinforced_at=now,
            base_strength=float(self.config.initial_strength),
            valid_from=now,
            scope=scope,
            author=(self.author if author is None else str(author).strip()),
        )
        mid = self.store.insert(rec)
        self._trace_count += 1
        self.index.write(sdr, mid)
        for u in np.asarray(sdr).tolist():
            self._unit_index.setdefault(int(u), set()).add(mid)
        self._cache_append(mid, rec.embedding)
        return mid

    def _sample_pairs(self, ua: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total = int(ua.size * ub.size)
        cap = min(self.config.max_pairs_per_bind, total)
        flat = self._rng.choice(total, size=cap, replace=False)
        ii = ua[flat // ub.size]
        jj = ub[flat % ub.size]
        keep = ii != jj
        return ii[keep].astype(np.int32), jj[keep].astype(np.int32)

    def _bind_sdrs(self, units_a, units_b, strength: float, now: float, source_ids) -> int:
        ua = np.unique(np.asarray(units_a, dtype=np.int64))
        ub = np.unique(np.asarray(units_b, dtype=np.int64))
        if ua.size == 0 or ub.size == 0 or strength <= 0:
            return 0
        ii, jj = self._sample_pairs(ua, ub)
        if ii.size == 0:
            return 0
        # write-through: bind сразу в БД, потерять его нельзя даже при падении процесса
        self.store.elig_add(ii, jj, strength, now, source_ids)
        return int(ii.size)

    def _reinforce(self, memory_id: int, now: float) -> None:
        rec = self.store.get(memory_id)
        if rec is None:
            return
        base, count = reinforce_values(rec.base_strength, rec.reinforced_count, self.config)
        new_kind = (
            KIND_SEMANTIC
            if should_promote(rec.kind, count, rec.created_at, now, self.config)
            else rec.kind
        )
        self.store.update_trace(memory_id, base, count, now, kind=new_kind)

    @property
    def pending_eligibility(self) -> int:
        """Незакоммиченных bind'ов в логе (видны всем процессам)."""
        return self.store.elig_pending()

    # -- запись --------------------------------------------------------------------

    def remember(
        self,
        text: str,
        *,
        kind: str = KIND_EPISODIC,
        meta: dict[str, Any] | None = None,
        when: float | None = None,
        force_new: bool = False,
        related_ids: Sequence[int] = (),
        scope: str = SCOPE_GLOBAL,
    ) -> WriteResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text должен быть непустой строкой")
        if kind not in (KIND_EPISODIC, KIND_SEMANTIC):
            raise ValueError(f"kind должен быть '{KIND_EPISODIC}' или '{KIND_SEMANTIC}'")
        if not isinstance(scope, str) or not _NAMESPACE_RE.fullmatch(scope):
            raise ValueError(
                "scope: 1–64 символа из [A-Za-z0-9_.-], первый — буква или цифра"
            )
        related_ids = tuple(dict.fromkeys(int(i) for i in related_ids))
        known: dict[int, MemoryRecord] = {}
        for rid in related_ids:
            rec = self.store.get(rid)
            if rec is None:
                raise KeyError(f"related_id {rid} не существует")
            known[rid] = rec
        now = float(when) if when is not None else float(self.clock.now())
        emb, sdr = self._encode(text)
        best_id, best_cos, near_ids = self._probe(emb, sdr, scope=scope, text=text)

        action = DecisionAction.CREATE if force_new else gate(best_cos, self.config)
        target = best_id if action is DecisionAction.REINFORCE else None
        link_ids: tuple[int, ...] = ()
        if action is DecisionAction.LINK and best_id is not None and best_cos >= self.config.theta_link:
            link_ids = (best_id,) + near_ids
        all_related = tuple(dict.fromkeys(link_ids + related_ids))

        decision = WriteDecision(
            action=action,
            target_id=target,
            related_ids=all_related,
            novelty=1.0 - best_cos,
            best_cosine=best_cos,
        )
        if action is DecisionAction.REINFORCE:
            self._reinforce(best_id, now)
            mid = best_id
            created = False
        elif action is DecisionAction.LINK:
            mid = self._insert_memory(text, kind, meta, now, emb, sdr, scope=scope)
            for rid in all_related:
                other = known.get(rid)
                other_sdr = other.sdr if other is not None else self._sdr_of(rid)
                self._bind_sdrs(sdr, other_sdr, strength=1.0, now=now, source_ids=(mid, rid))
            created = True
        else:
            mid = self._insert_memory(text, kind, meta, now, emb, sdr, scope=scope)
            for rid in all_related:
                self._bind_sdrs(sdr, self._sdr_of(rid), strength=0.5, now=now, source_ids=(mid, rid))
            created = True

        self.journal.append("write", id=mid, kind=kind, action=action.value,
                            created=created, chars=len(text), scope=scope, t=now)
        return WriteResult(memory_id=mid, decision=decision, created=created)

    def _sdr_of(self, memory_id: int) -> np.ndarray:
        rec = self.store.get(memory_id)
        return rec.sdr if rec is not None else np.empty(0, dtype=np.int32)

    def link_memories(self, ids: Sequence[int], strength: float = 1.0) -> int:
        uniq = tuple(dict.fromkeys(int(i) for i in ids))
        if len(uniq) < 2:
            raise ValueError("нужны минимум два следа")
        recs: dict[int, MemoryRecord] = {}
        for i in uniq:
            rec = self.store.get(i)
            if rec is None:
                raise KeyError(f"след {i} не существует")
            recs[i] = rec
        added = 0
        for a in range(len(uniq)):
            for b in range(a + 1, len(uniq)):
                ia, ib = uniq[a], uniq[b]
                added += self._bind_sdrs(recs[ia].sdr, recs[ib].sdr, strength,
                                         self.clock.now(), source_ids=(ia, ib))
        self.journal.append("link", ids=list(uniq), pairs_added=added, strength=strength)
        return added

    def update_fact(
        self,
        old_id: int,
        new_text: str,
        meta: dict[str, Any] | None = None,
    ) -> WriteResult:
        old = self.store.get(old_id)
        if old is None:
            raise KeyError(f"след {old_id} не существует")
        merged_meta = {**old.meta, **(meta or {}), "supersedes": int(old_id)}
        res = self.remember(new_text, kind=old.kind, meta=merged_meta, force_new=True,
                            scope=old.scope)
        now = float(self.clock.now())
        self.store.mark_superseded(old_id, res.memory_id, now)
        self._bind_sdrs(old.sdr, self._sdr_of(res.memory_id),
                        strength=0.75, now=now, source_ids=(old_id, res.memory_id))
        self.journal.append("supersede", old=old_id, new=res.memory_id, t=now)
        return res

    # -- чтение ---------------------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        include_superseded: bool = False,
        scope: str | None = None,
        all_scopes: bool = False,
    ) -> RecallPacket:
        """Поиск по памяти. scope='<проект>' видит следы проекта и global;
        scope=None или all_scopes=True — вся память без фильтра."""
        t0 = _time.perf_counter()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query должен быть непустой строкой")
        if k < 1:
            raise ValueError("k должен быть >= 1")
        now = float(self.clock.now())
        emb, sdr = self._encode(query, query=True)

        # Два движка кандидатов:
        #  exact — точный косинус-скан по кэшу эмбеддингов: полнота без обрыва
        #    голосования, порядок обхода — глобальный рейтинг похожести;
        #  votes — исторический L1-поиск голосами юнитов + FTS (большие базы).
        use_exact = self._exact_mode()
        engine = "exact" if use_exact else "votes"
        kw_ids = set(self._fts_candidates(query, limit=k * self.config.recall_oversample))
        query_tokens = search_tokens(query) or None

        items: list[tuple[float, float, float, MemoryRecord, str]] = []
        seen: set[int] = set()
        has_full_match = False
        w_votes_eff = 1.0 if use_exact else self.config.w_votes
        qr_active = 0
        votes_map: dict[int, int] = {}

        def conf_direct(cos: float, votes_norm: float, ret: float) -> float:
            return cos * (w_votes_eff + (1 - w_votes_eff) * min(1.0, votes_norm)) * (
                0.3 + 0.7 * ret
            )

        def ret_factor(ret: float) -> float:
            return 0.3 + 0.7 * ret

        def evaluate(rec: MemoryRecord, cos: float) -> None:
            """Ветвление скоринга одного кандидата (общее для обоих проходов).
            Косинус приходит предвычисленным: точный скан считает его gemv,
            FTS-довесок и голосовательное окно — векторно по стопке строк."""
            nonlocal has_full_match
            if rec.status != STATUS_ACTIVE and not include_superseded:
                return
            if not self._scope_allows(rec.scope, scope, all_scopes):
                return
            cos = max(0.0, cos)
            ret = retention(rec.base_strength, rec.last_reinforced_at, now, rec.kind,
                            self.config)
            if ret < self.config.min_retention_recall:
                return
            # полный keyword-матч (все токены запроса есть в следе): точное
            # совпадение надёжнее слабого косинуса; источник остаётся direct,
            # если косинус сам прошёл порог, keyword — только когда семантики
            # не хватило и держит нас токен
            full_match = False
            doc_tokens: set[str] | None = None
            if int(rec.id) in kw_ids and query_tokens:
                doc_tokens = search_tokens(rec.text)
                full_match = bool(query_tokens <= doc_tokens)
            if cos >= self.config.cos_min_recall:
                vnorm = votes_map.get(int(rec.id), 0) / max(1, qr_active)
                c = conf_direct(cos, vnorm, ret)
                if full_match:
                    c = max(c, self.config.w_keyword * ret_factor(ret))
                    has_full_match = True
                source = SOURCE_DIRECT
            elif full_match:
                c = self.config.w_keyword * ret_factor(ret)
                source = SOURCE_KEYWORD
                has_full_match = True
            elif int(rec.id) in kw_ids and query_tokens and doc_tokens is not None:
                overlap = len(doc_tokens & query_tokens) / len(query_tokens)
                if overlap <= 0.0:
                    return
                c = self.config.w_keyword * overlap * ret_factor(ret)
                source = SOURCE_KEYWORD
            else:
                return
            items.append((c, cos, ret, rec, source))
            seen.add(int(rec.id))

        if not use_exact:
            qr = self.index.query(
                sdr,
                max_candidates=self._candidate_budget(k * self.config.recall_oversample),
            )
            qr_active = int(qr.active_locations)
            votes_map = {
                int(p): int(v) for p, v in zip(qr.candidates.tolist(), qr.votes.tolist())
            }
            candidates = list(votes_map.keys()) + sorted(
                rid for rid in kw_ids if rid not in votes_map
            )
            recs = self.store.get_many(candidates)
            if recs:
                qn = float(np.linalg.norm(emb))
                qv = (emb / np.float32(qn)).astype(np.float32) if qn else np.zeros_like(emb)
                sims = np.stack([r.embedding for r in recs]) @ qv
                for rec, sim in zip(recs, sims.tolist()):
                    evaluate(rec, sim)

        if use_exact:
            ranked_ids, ranked_cos = self._cosine_ranked(emb)
            # словарь id -> косинус из глобального рейтинга: позиция в буфере
            # не равна позиции в отсортированном порядке
            rank_cos_by_id = dict(zip(ranked_ids.tolist(), ranked_cos.tolist()))
            window_target = self._candidate_budget(k * self.config.recall_oversample)
            step = 64
            start = 0
            while start < len(ranked_ids):
                stop_ids = [int(i) for i in ranked_ids[start:start + step].tolist()]
                for rec in self.store.get_many(stop_ids):
                    evaluate(rec, float(rank_cos_by_id.get(int(rec.id), 0.0)))
                start += step
                # ранний выход между подбатчами: топ-k укомплектован, объём
                # разведки не меньше окна голосования, а сильнейший из остатка
                # не может вытеснить слабейшего в выбранном
                # (у direct-канала c <= cos при w_votes_eff = 1)
                if len(items) >= k and start >= window_target:
                    next_best_cos = float(ranked_cos[start]) if start < len(ranked_cos) else 0.0
                    if next_best_cos <= min(it[0] for it in items):
                        break
            # проход B: все FTS-кандидаты вне уже оценённых — keyword-буст
            # обязан сохранить шансы независимо от глубины в рейтинге
            kw_extras = sorted(rid for rid in kw_ids if rid not in seen)
            if kw_extras:
                qn = float(np.linalg.norm(emb))
                qv = (emb / np.float32(qn)).astype(np.float32) if qn else np.zeros_like(emb)
                for rec in self.store.get_many(kw_extras):
                    cos = float(np.dot(rec.embedding, qv)) if rec.embedding.size else 0.0
                    evaluate(rec, cos)

        items.sort(key=lambda t: (-t[0], t[3].id))
        items = items[:k]

        # относительное воздержание точного скана: центр анизотропной массы
        # (медиана рейтинга корпуса) сам масштабируется с эмбеддером/размерностью,
        # поэтому правило переносимо туда, где абсолютные пороги не выживают.
        # Ответ обязан возвышаться над медианой хотя бы на margin; полный
        # keyword-матч отменяет правило — точный токен сам по себе надёжен.
        if (
            use_exact
            and items
            and not has_full_match
            and self.config.exact_abstain_rel_margin > 0
            and ranked_cos.size
        ):
            stride = max(1, ranked_cos.size // 512)
            null_median = float(np.median(ranked_cos[::stride]))
            if items[0][1] < null_median + self.config.exact_abstain_rel_margin:
                latency_ms = (_time.perf_counter() - t0) * 1000.0
                packet = RecallPacket(query=query, items=(), abstained=True,
                                      latency_ms=latency_ms)
                self.journal.append(
                    "recall",
                    k=k, items=0, abstained=True, latency_ms=round(latency_ms, 3),
                    top_conf=None, below_null=True,
                    null_median=round(null_median, 4), engine=engine, t=now,
                )
                return packet

        # воздержание «нет выраженного лидера»: top1 ниже сильного порога и вся
        # прямая волна лежит в узком коридоре — это шум анизотропии, а не ответ;
        # полный keyword-матч отменяет правило (точный токен сам по себе надёжен).
        # Правило opt-in: включается профилем эмбеддера (cos_min_strong_recall > 0),
        # потому что абсолютные пороги непереносимы между моделями/размерностями —
        # на hashing dim=2048 дефолт ложно съедал корректные ответы на 30k+ фактов
        if (
            items
            and self.config.cos_min_strong_recall > 0
            and not has_full_match
            and items[0][1] < self.config.cos_min_strong_recall
            and (items[0][1] - min(it[1] for it in items)) < self.config.abstain_spread_cos
        ):
            latency_ms = (_time.perf_counter() - t0) * 1000.0
            packet = RecallPacket(query=query, items=(), abstained=True,
                                  latency_ms=latency_ms)
            self.journal.append(
                "recall",
                k=k, items=0, abstained=True, latency_ms=round(latency_ms, 3),
                top_conf=None, flat_noise=True, engine=engine, t=now,
            )
            return packet

        # волна ассоциаций: spread от SDR топ-следов по пластичным рёбрам;
        # сигнал связи — достижимые юниты, поэтому косинусный фильтр не применяется
        if 0 < len(items) < k and sdr.size:
            self._sync_network_cache()
            seeds = np.concatenate([it[3].sdr for it in items[: min(3, len(items))]])
            units, _scores = self.network.spread(
                seeds,
                depth=self.config.spread_depth,
                alpha=self.config.spread_alpha,
                top_m=self.config.spread_top_m,
                eps=self.config.spread_eps,
            )
            assoc_ids: list[int] = []
            for u in units.tolist():
                for mid in self._unit_index.get(int(u), ()):  # юнит -> следы
                    if mid not in seen:
                        assoc_ids.append(mid)
            assoc_ids = list(dict.fromkeys(assoc_ids))[: k - len(items)]
            extras: list[tuple[float, float, float, MemoryRecord, str]] = []
            for rec in self.store.get_many(assoc_ids):
                if rec.status != STATUS_ACTIVE and not include_superseded:
                    continue
                if not self._scope_allows(rec.scope, scope, all_scopes):
                    continue
                cos = max(0.0, self._cosine(emb, rec.embedding))
                ret = retention(rec.base_strength, rec.last_reinforced_at, now, rec.kind, self.config)
                if ret < self.config.min_retention_recall:
                    continue
                c = self.config.assoc_confidence_penalty * _assoc_cos_floor(cos) * (0.3 + 0.7 * ret)
                extras.append((c, cos, ret, rec, SOURCE_ASSOCIATED))
                seen.add(int(rec.id))
            extras.sort(key=lambda t: (-t[0], t[3].id))
            items += extras[: k - len(items)]

        out = tuple(
            RecalledMemory(
                memory_id=int(rec.id),
                text=rec.text,
                kind=rec.kind,
                cosine=round(cos, 6),
                confidence=round(c, 6),
                retention=round(ret, 6),
                source=source,
                created_at=rec.created_at,
                updated_at=rec.updated_at,
                meta=dict(rec.meta),
                scope=rec.scope,
                author=rec.author,
            )
            for c, cos, ret, rec, source in items
        )
        latency_ms = (_time.perf_counter() - t0) * 1000.0
        packet = RecallPacket(query=query, items=out, abstained=len(out) == 0,
                              latency_ms=latency_ms)
        # наблюдаемость: каждое обращение — для последующего анализа трендов
        self.journal.append(
            "recall",
            k=k,
            items=len(out),
            abstained=packet.abstained,
            latency_ms=round(latency_ms, 3),
            top_conf=out[0].confidence if out else None,
            engine=engine,
            t=now,
        )
        return packet

    # -- обратная связь и консолидация ---------------------------------------------

    def feedback(self, ids: Sequence[int], reward: float) -> int:
        """Позитивный reward подкрепляет следы (bump + сброс таймера), негативный —
        ослабляет их к забыванию (без сброса таймера и счётчика продвижения),
        ноль нейтрален. Возвращает число затронутых следов."""
        if not -1.0 <= reward <= 1.0:
            raise ValueError("reward должен быть в [-1, 1]")
        uniq = tuple(dict.fromkeys(int(i) for i in ids))
        factor = max(0.0, 1.0 + reward)
        touched = self.store.elig_reward(uniq, factor)
        if touched > 0:
            self.store.bump_meta_int("pending_reward_touches", touched)
        now = float(self.clock.now())
        reinforced = weakened = 0
        for i in uniq:
            rec = self.store.get(i)
            if rec is None:
                continue
            if reward > 0.0:
                self._reinforce(i, now)
                reinforced += 1
            elif reward < 0.0:
                self.store.adjust_base(i, weaken_value(rec.base_strength, reward), now)
                weakened += 1
        self.journal.append(
            "feedback",
            ids=list(uniq),
            reward=reward,
            reinforced=reinforced,
            weakened=weakened,
            eligible_touched=touched,
        )
        return reinforced + weakened

    def consolidate(self) -> ConsolidationReport:
        """«Сон»: выкачать eligibility в стабильные рёбра (с reward-усилением и
        распадом), повысить дозревшие эпизоды до семантических, удалить давно
        забытые ниже recall-пола (после grace-периода), записать метрики.
        Всё состояние уже в БД, отдельного сохранения не требуется; параллельные
        консолидации других процессов сериализуются транзакцией."""
        t0 = _time.perf_counter()
        now = float(self.clock.now())
        # страховочная копия до любых изменений; троттлинг по wall-clock — сны
        # после каждого ответа агента не обязаны копировать всю базу. Сбой
        # бэкапа не отменяет «сон», но остаётся в журнале.
        if self.config.backups_keep > 0:
            try:
                copied = self.store.backup(
                    keep=self.config.backups_keep,
                    min_interval_s=self.config.backup_min_interval_s,
                )
                if copied is None:
                    self.journal.append(
                        "backup_skipped", interval_s=self.config.backup_min_interval_s
                    )
            except Exception as exc:  # noqa: BLE001 - сон важнее идеального бэкапа
                self.journal.append("backup_error", error=str(exc))
        # ротация журнала после бэкапа: обрезанные события остаются в копии
        journal_pruned = 0
        if self.config.journal_max_events > 0:
            journal_pruned = self.store.events_prune(self.config.journal_max_events)
            if journal_pruned:
                self.journal.append("journal_rotated",
                                    pruned=journal_pruned, kept=self.config.journal_max_events)
        rewards_applied = self.store.consume_meta_int("pending_reward_touches")
        rows = self.store.elig_drain()
        committed = pruned = 0
        if rows:
            staging = EligibilityLog(self.config.tau_eligibility)
            staging.load_state_dict({
                "tau": self.config.tau_eligibility,
                "events": rows,
            })
            src, dst, w = staging.commit(now)
            committed, pruned = self.store.edges_apply(
                src, dst, w, now,
                tau=self.config.tau_edge_stable,
                min_weight=self.config.edge_min_weight,
                stride=self.config.n_units,
            )
        else:
            # часы сети двигаются даже пустым сном: слабые рёбра обязаны угасать
            committed, pruned = self.store.edges_apply(
                np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32),
                now, tau=self.config.tau_edge_stable,
                min_weight=self.config.edge_min_weight, stride=self.config.n_units,
            )
        self._reload_network_cache()

        # один проход по активным следам решает обе судьбы: дозревшие эпизоды
        # повышаются до семантических, давно забытые ниже recall-пола удаляются
        promote_ids: list[int] = []
        forgotten_ids: list[int] = []
        grace_ok = now - self.config.gc_grace_below_floor_s
        for rec in self.store.iter_active():
            if should_promote(rec.kind, rec.reinforced_count, rec.created_at, now, self.config):
                promote_ids.append(int(rec.id))
            elif (
                self.config.gc_enabled
                and rec.last_reinforced_at <= grace_ok
                and retention(rec.base_strength, rec.last_reinforced_at, now,
                              rec.kind, self.config) < self.config.min_retention_recall
            ):
                forgotten_ids.append(int(rec.id))
        for rid in promote_ids:
            promoted_rec = self.store.get(rid)
            if promoted_rec is not None:
                self.store.update_trace(
                    rid, promoted_rec.base_strength, promoted_rec.reinforced_count,
                    promoted_rec.last_reinforced_at, kind=KIND_SEMANTIC,
                )
        if forgotten_ids:
            self.store.forget_traces(forgotten_ids)
            self._evict_forgotten(set(forgotten_ids))
            self._trace_count = self.store.count()
        report = ConsolidationReport(
            edges_committed=committed,
            edges_pruned=pruned,
            promoted_to_semantic=len(promote_ids),
            rewards_applied=rewards_applied,
            journal_pruned=journal_pruned,
            forgotten_traces=len(forgotten_ids),
            elapsed_ms=(_time.perf_counter() - t0) * 1000.0,
        )
        self.store.set_meta("last_consolidate_at", repr(now))
        self.journal.append("consolidate", **dataclasses.asdict(report))
        self.journal.append("metrics", **self.metrics_snapshot(now=now))
        return report

    def _evict_forgotten(self, ids: set[int]) -> None:
        """Вычистить удалённые следы из производных кэшей процесса.

        Эмбеддинговый буфер уплотняется свопом с хвоста (O(1) на след, порядок
        строк не сохраняется — рейтинги строятся на каждом recall заново).
        L1-бакеты не трогаются: устаревшие указатели отбрасываются на чтении
        (store.get → None), palimpsest вытеснит их под давлением ёмкости."""
        for u in list(self._unit_index.keys()):
            survivors = self._unit_index[u] - ids
            if survivors:
                self._unit_index[u] = survivors
            else:
                del self._unit_index[u]
        for mid in ids:
            pos = self._emb_pos.pop(mid, None)
            if pos is None or pos >= self._emb_len:
                continue
            last = self._emb_len - 1
            if pos != last:
                self._emb_buf[pos] = self._emb_buf[last]
                moved_id = int(self._idbuf[last])
                self._idbuf[pos] = moved_id
                self._emb_pos[moved_id] = pos
            self._emb_len -= 1

    def metrics_snapshot(self, now: float | None = None) -> dict[str, Any]:
        """Полный срез состояния для анализа «со временем»: счётчики, гистерезис
        затухания, возраст следов. Вызывается на каждом consolidate(); те же
        данные доступны по требованию (тул dream_log / CLI report)."""
        now = float(now) if now is not None else float(self.clock.now())
        rets: list[float] = []
        ages: list[float] = []
        for rec in self.store.iter_active():
            rets.append(
                retention(rec.base_strength, rec.last_reinforced_at, now, rec.kind, self.config)
            )
            ages.append(max(0.0, now - rec.created_at))
        r = np.asarray(rets, dtype=np.float64) if rets else np.empty(0)
        a = np.asarray(ages, dtype=np.float64) if ages else np.empty(0)
        day = 86400.0
        return {
            "ts": now,
            **self.stats(),
            "retention_mean": round(float(r.mean()), 4) if r.size else 1.0,
            "retention_below_02": int((r < 0.2).sum()),
            "age_days_mean": round(float(a.mean()) / day, 2) if a.size else 0.0,
            "index_load_factor": round(self.index.load_factor(), 3),
        }

    # -- статистика ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Глобальная статистика из БД: одинакова для всех процессов в любой момент."""
        decisions: dict[str, int] = {a.value: 0 for a in DecisionAction}
        for act, n in self.store.gate_decisions().items():
            decisions[act] = decisions.get(act, 0) + n
        recalls, avg_recall_ms = self.store.recall_stats()
        edge_count, total_weight = self.store.edges_stats(
            float(self.clock.now()), self.config.tau_edge_stable
        )
        return {
            "memories_active": self.store.count(status=STATUS_ACTIVE),
            "memories_superseded": self.store.count(status="superseded"),
            "kind_episodic": self.store.count(kind=KIND_EPISODIC),
            "kind_semantic": self.store.count(kind=KIND_SEMANTIC),
            "edges": edge_count,
            "total_edge_weight": total_weight,
            "pending_eligibility": self.store.elig_pending(),
            "decisions": decisions,
            "writes": sum(decisions.values()),
            "recalls": recalls,
            "avg_recall_ms": avg_recall_ms,
            "journal_events": self.store.event_count(),
        }


def _assoc_cos_floor(cos: float) -> float:
    """Для associated-элементов косинус может быть ~0: связь даётся юнитами.
    Пол гарантирует ненулевой вклад уверенности от самой связи."""
    return max(cos, 0.05)
