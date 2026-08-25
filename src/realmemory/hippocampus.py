"""Hippocampus — фасад realMemory («гиппокамп»).

Связывает кодирование (эмбеддер -> биполярный адрес + SDR), L1-адресацию,
L2-ассоциации, политики новизны/затухания и SQLite-хранилище в единые пути
remember/recall/feedback/consolidate. Контракт: docs/CONTRACTS.md.

Восстановление состояния при открытии: L1-бакеты и юнит-индекс всегда
перестраиваются из БД (детерминированно); рёбра L2, eligibility-лог и счётчики —
из snapshot.pkl.
"""
from __future__ import annotations

import dataclasses
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
from .store.journal import Journal
from .store.sqlite_store import MemoryStore
from .timeprov import SystemClock, TimeProvider
from .types import (
    KIND_EPISODIC,
    KIND_SEMANTIC,
    SOURCE_ASSOCIATED,
    SOURCE_DIRECT,
    STATUS_ACTIVE,
    ConsolidationReport,
    DecisionAction,
    MemoryRecord,
    RecalledMemory,
    RecallPacket,
    WriteDecision,
    WriteResult,
)

SNAPSHOT_VERSION = 1
_DB_NAME = "memory.db"
_SNAPSHOT_NAME = "snapshot.pkl"
_JOURNAL_NAME = "journal.jsonl"

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
    ) -> None:
        self.config = config or MemoryConfig.dev()
        self.config.validate()
        self.path = _resolve_root(path, namespace)
        self.path.mkdir(parents=True, exist_ok=True)
        self.clock = clock or SystemClock()
        self.embedder = embedder or HashingEmbedder(dim=self.config.dim)
        if self.embedder.dim != self.config.dim:
            raise ValueError(
                f"dim эмбеддера ({self.embedder.dim}) не совпадает с config.dim ({self.config.dim})"
            )
        self.store = MemoryStore(self.path / _DB_NAME, self.config.dim)
        self.sdr_encoder = SDREncoder(
            self.config.dim, self.config.n_units, self.config.k_sparse, self.config.sdr_seed
        )
        self.index = SDRVotingIndex(self.config.n_units, bucket_cap=self.config.bucket_cap)
        self.network = AssemblyNetwork(
            self.config.n_units,
            edge_min_weight=self.config.edge_min_weight,
            tau_edge_stable=self.config.tau_edge_stable,
            seed=self.config.sdr_seed + 1,
            max_pairs_per_bind=self.config.max_pairs_per_bind,
        )
        self.eligibility = EligibilityLog(self.config.tau_eligibility)
        self.journal = Journal(self.path / _JOURNAL_NAME)
        if verify_embedder:
            # False для инструментов без эмбеддингов (хуки сна/брифа): они не
            # порождают векторов и не должны спорить с боевым эмбеддером базы
            self._check_embedder_identity()
        self._rng = np.random.default_rng(self.config.sdr_seed + 2)
        self._unit_index: dict[int, set[int]] = {}
        self._decision_counts: dict[str, int] = {a.value: 0 for a in DecisionAction}
        self._stats: dict[str, Any] = {"writes": 0, "recalls": 0, "sum_recall_ms": 0.0}
        self._pending_rewards = 0
        self._load_state()

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
    ) -> Hippocampus:
        return cls(path, config=config, embedder=embedder, clock=clock,
                   namespace=namespace, verify_embedder=verify_embedder)

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
                self.journal.append("embedder_identity_adopted", name=self.embedder.name)
            self.store.set_meta("embedder", self.embedder.name)
            return
        if stored != self.embedder.name:
            raise RuntimeError(
                f"эмбеддер базы {self.path} — '{stored}', а открывается с "
                f"'{self.embedder.name}'. Старые и новые эмбеддинги несравнимы; "
                "откройте базу исходным эмбеддером или начните новую директорию памяти."
            )

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

    def _probe(self, emb: np.ndarray, sdr: np.ndarray) -> tuple[int | None, float, tuple[int, ...]]:
        """Лучший существующий след по косинусу среди кандидатов L1."""
        qr = self.index.query(sdr, max_candidates=max(2, self.config.recall_oversample * 3))
        if qr.candidates.size == 0:
            return None, 0.0, ()
        scored: list[tuple[int, float]] = []
        for rec in self.store.get_many(qr.candidates.tolist()):
            if rec.status != STATUS_ACTIVE:
                continue
            scored.append((int(rec.id), max(0.0, self._cosine(emb, rec.embedding))))
        if not scored:
            return None, 0.0, ()
        scored.sort(key=lambda t: -t[1])
        best_id, best_cos = scored[0]
        near = tuple(i for i, c in scored[1:] if c >= self.config.theta_link)[:4]
        return best_id, best_cos, near

    def _insert_memory(self, text, kind, meta, now, emb, sdr) -> int:
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
        )
        mid = self.store.insert(rec)
        self.index.write(sdr, mid)
        for u in np.asarray(sdr).tolist():
            self._unit_index.setdefault(int(u), set()).add(mid)
        self.journal.append("write", id=mid, kind=rec.kind, chars=len(text), t=now)
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
        self.eligibility.add(ii, jj, strength, now, source_ids)
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
    ) -> WriteResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text должен быть непустой строкой")
        if kind not in (KIND_EPISODIC, KIND_SEMANTIC):
            raise ValueError(f"kind должен быть '{KIND_EPISODIC}' или '{KIND_SEMANTIC}'")
        related_ids = tuple(dict.fromkeys(int(i) for i in related_ids))
        known: dict[int, MemoryRecord] = {}
        for rid in related_ids:
            rec = self.store.get(rid)
            if rec is None:
                raise KeyError(f"related_id {rid} не существует")
            known[rid] = rec
        now = float(when) if when is not None else float(self.clock.now())
        emb, sdr = self._encode(text)
        best_id, best_cos, near_ids = self._probe(emb, sdr)

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
            mid = self._insert_memory(text, kind, meta, now, emb, sdr)
            for rid in all_related:
                other = known.get(rid)
                other_sdr = other.sdr if other is not None else self._sdr_of(rid)
                self._bind_sdrs(sdr, other_sdr, strength=1.0, now=now, source_ids=(mid, rid))
            created = True
        else:
            mid = self._insert_memory(text, kind, meta, now, emb, sdr)
            for rid in all_related:
                self._bind_sdrs(sdr, self._sdr_of(rid), strength=0.5, now=now, source_ids=(mid, rid))
            created = True

        self._decision_counts[action.value] += 1
        self._stats["writes"] += 1
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
        res = self.remember(new_text, kind=old.kind, meta=merged_meta, force_new=True)
        now = float(self.clock.now())
        self.store.mark_superseded(old_id, res.memory_id, now)
        self._bind_sdrs(old.sdr, self._sdr_of(res.memory_id),
                        strength=0.75, now=now, source_ids=(old_id, res.memory_id))
        self.journal.append("supersede", old=old_id, new=res.memory_id, t=now)
        return res

    # -- чтение ---------------------------------------------------------------------

    def recall(self, query: str, *, k: int = 5, include_superseded: bool = False) -> RecallPacket:
        t0 = _time.perf_counter()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query должен быть непустой строкой")
        if k < 1:
            raise ValueError("k должен быть >= 1")
        now = float(self.clock.now())
        emb, sdr = self._encode(query, query=True)

        qr = self.index.query(sdr, max_candidates=k * self.config.recall_oversample)
        votes_map = {
            int(p): int(v) for p, v in zip(qr.candidates.tolist(), qr.votes.tolist())
        }
        items: list[tuple[float, float, float, MemoryRecord, str]] = []
        seen: set[int] = set()

        def conf_direct(cos: float, votes_norm: float, ret: float) -> float:
            return cos * (self.config.w_votes + (1 - self.config.w_votes) * min(1.0, votes_norm)) * (
                0.3 + 0.7 * ret
            )

        for rec in self.store.get_many(list(votes_map.keys())):
            if rec.status != STATUS_ACTIVE and not include_superseded:
                continue
            cos = max(0.0, self._cosine(emb, rec.embedding))
            if cos < self.config.cos_min_recall:
                continue
            ret = retention(rec.base_strength, rec.last_reinforced_at, now, rec.kind, self.config)
            if ret < self.config.min_retention_recall:
                continue
            vnorm = votes_map.get(int(rec.id), 0) / max(1, qr.active_locations)
            c = conf_direct(cos, vnorm, ret)
            items.append((c, cos, ret, rec, SOURCE_DIRECT))
            seen.add(int(rec.id))
        items.sort(key=lambda t: (-t[0], t[3].id))
        items = items[:k]

        # волна ассоциаций: spread от SDR топ-следов по пластичным рёбрам;
        # сигнал связи — достижимые юниты, поэтому косинусный фильтр не применяется
        if 0 < len(items) < k and sdr.size:
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
            )
            for c, cos, ret, rec, source in items
        )
        latency_ms = (_time.perf_counter() - t0) * 1000.0
        self._stats["recalls"] += 1
        self._stats["sum_recall_ms"] += latency_ms
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
        touched = self.eligibility.reward(uniq, reward)
        self._pending_rewards += touched
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

    def consolidate(self, save: bool = True) -> ConsolidationReport:
        t0 = _time.perf_counter()
        now = float(self.clock.now())
        e0 = self.network.edge_count
        src, dst, w = self.eligibility.commit(now)
        self.network.commit_eligibility(src, dst, w, now)
        e1 = self.network.edge_count
        edges_pruned = max(0, e0 + int(w.size) - e1)
        promoted = 0
        for rec in self.store.iter_active():
            if should_promote(rec.kind, rec.reinforced_count, rec.created_at, now, self.config):
                self.store.update_trace(
                    rec.id, rec.base_strength, rec.reinforced_count, rec.last_reinforced_at,
                    kind=KIND_SEMANTIC,
                )
                promoted += 1
        rewards_applied = self._pending_rewards
        self._pending_rewards = 0
        report = ConsolidationReport(
            edges_committed=int(w.size),
            edges_pruned=edges_pruned,
            promoted_to_semantic=promoted,
            rewards_applied=rewards_applied,
            elapsed_ms=(_time.perf_counter() - t0) * 1000.0,
        )
        self.journal.append("consolidate", **dataclasses.asdict(report))
        self.journal.append("metrics", **self.metrics_snapshot(now=now))
        if save:
            self.save()
        return report

    def metrics_snapshot(self, now: float | None = None) -> dict[str, Any]:
        """Полный срез состояния для анализа «со временем»: счётчики, гистерезис
        затухания, возраст следов. Вызывается на каждом consolidate(); те же
        данные доступны по требованию (тул memory_report / CLI report)."""
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

    # -- статистика и снапшоты --------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        recalls = max(1, self._stats["recalls"])
        return {
            "memories_active": self.store.count(status=STATUS_ACTIVE),
            "memories_superseded": self.store.count(status="superseded"),
            "kind_episodic": self.store.count(kind=KIND_EPISODIC),
            "kind_semantic": self.store.count(kind=KIND_SEMANTIC),
            "edges": self.network.edge_count,
            "total_edge_weight": round(self.network.total_weight, 4),
            "pending_eligibility": self.eligibility.pending_count,
            "decisions": dict(self._decision_counts),
            "writes": self._stats["writes"],
            "recalls": self._stats["recalls"],
            "avg_recall_ms": round(self._stats["sum_recall_ms"] / recalls, 3),
            "journal_events": self.journal.count(),
        }

    def save(self) -> None:
        payload = {
            "version": SNAPSHOT_VERSION,
            "config": self.config.snapshot_fields(),
            "network": self.network.state_dict(),
            "eligibility": self.eligibility.state_dict(),
            "stats": dict(self._stats),
            "decisions": dict(self._decision_counts),
            "pending_rewards": self._pending_rewards,
        }
        final = self.path / _SNAPSHOT_NAME
        tmp = final.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(final)
        self.journal.append("snapshot", t=float(self.clock.now()))

    def _load_state(self) -> None:
        for rec in self.store.iter_active():
            self.index.write(rec.sdr, int(rec.id))
            for u in rec.sdr.tolist():
                self._unit_index.setdefault(int(u), set()).add(int(rec.id))
        snap_path = self.path / _SNAPSHOT_NAME
        if not snap_path.exists():
            return
        try:
            with snap_path.open("rb") as f:
                payload = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            raise RuntimeError(f"повреждён снапшот {snap_path}: {exc}") from exc
        if payload.get("version") != SNAPSHOT_VERSION:
            raise RuntimeError("несовместимая версия снапшота")
        cfg = payload.get("config", {})
        for key in ("dim", "n_units", "sdr_seed"):
            if key in cfg and cfg[key] != getattr(self.config, key):
                raise RuntimeError(f"конфиг не совпадает со снапшотом по полю {key}")
        self.network.load_state_dict(payload["network"])
        self.eligibility.load_state_dict(payload["eligibility"])
        stats = payload.get("stats", {})
        self._stats.update({k: stats[k] for k in self._stats if k in stats})
        self._decision_counts.update(payload.get("decisions", {}))
        self._pending_rewards = int(payload.get("pending_rewards", 0))


def _assoc_cos_floor(cos: float) -> float:
    """Для associated-элементов косинус может быть ~0: связь даётся юнитами.
    Пол гарантирует ненулевой вклад уверенности от самой связи."""
    return max(cos, 0.05)
