"""Гиперпараметры realMemory. Все временные величины — секунды."""
from __future__ import annotations

from dataclasses import dataclass, fields

SECONDS_PER_DAY = 86400.0


@dataclass
class MemoryConfig:
    # -- кодирование -------------------------------------------------------
    dim: int = 256  # размерность эмбеддера
    n_units: int = 1024  # юнитов в SDR-пространстве (L1-бакеты и L2-связи)
    k_sparse: int = 64  # on-битов на паттерн
    sdr_seed: int = 29

    # -- L1 SDRVotingIndex ----------------------------------------------------
    bucket_cap: int = 64  # горизонт palimpsest ≈ bucket_cap / (k·load/N)
    recall_oversample: int = 6  # кандидатов из L1 на один запрошенный k

    # -- гейт новизны (косинусная близость) ---------------------------------
    theta_reinforce: float = 0.45
    theta_link: float = 0.22
    cos_min_recall: float = 0.12
    # воздержание: top1 ниже сильного порога при плоском разбросе выдачи
    # (нет выраженного лидера) — надёжных воспоминаний нет;
    # полный keyword-матч отменяет правило (точный токен надёжен сам по себе)
    cos_min_strong_recall: float = 0.35
    abstain_spread_cos: float = 0.20

    # -- следы и затухание (секунды) -----------------------------------------
    tau_episodic: float = 30 * SECONDS_PER_DAY
    tau_semantic: float = 365 * SECONDS_PER_DAY
    min_retention_recall: float = 0.05
    initial_strength: float = 0.5  # base_strength нового следа; подкрепления растят к cap
    reinforce_bump: float = 1.25
    strength_cap: float = 1.0
    promote_after_reinforcements: int = 5
    promote_min_age_s: float = 60 * SECONDS_PER_DAY

    # -- связи L2 (секунды) ---------------------------------------------------
    tau_eligibility: float = 14 * SECONDS_PER_DAY
    tau_edge_stable: float = 90 * SECONDS_PER_DAY
    edge_min_weight: float = 0.02
    max_pairs_per_bind: int = 256

    # -- эксплуатация -----------------------------------------------------------
    backups_keep: int = 10  # копий в <каталог базы>/backups перед каждым «сном»; 0 — выключить

    # -- spread (ассоциативное распространение) -------------------------------
    spread_depth: int = 2
    spread_alpha: float = 0.5
    spread_top_m: int = 32
    spread_eps: float = 0.01

    # -- ранжирование recall ---------------------------------------------------
    w_votes: float = 0.5
    assoc_confidence_penalty: float = 0.8
    w_keyword: float = 0.65  # вес keyword-канала FTS5 (точные токены при низком косинусе)

    def validate(self) -> None:
        if self.dim <= 0 or self.n_units <= 0:
            raise ValueError("dim и n_units должны быть положительными")
        if not 0 < self.k_sparse <= self.n_units:
            raise ValueError("требуется 0 < k_sparse <= n_units")
        if not self.theta_reinforce > self.theta_link > self.cos_min_recall >= 0.0:
            raise ValueError("требуется theta_reinforce > theta_link > cos_min_recall >= 0")
        if self.cos_min_strong_recall < self.cos_min_recall:
            raise ValueError("cos_min_strong_recall должен быть >= cos_min_recall")
        if self.abstain_spread_cos <= 0:
            raise ValueError("abstain_spread_cos должен быть положительным")
        if self.bucket_cap <= 0:
            raise ValueError("bucket_cap должен быть положительным")
        if self.tau_episodic <= 0 or self.tau_semantic <= self.tau_episodic:
            raise ValueError("требуется tau_semantic > tau_episodic > 0")
        if self.tau_eligibility <= 0 or self.tau_edge_stable <= 0:
            raise ValueError("tau_eligibility и tau_edge_stable должны быть положительными")
        if not 0.0 <= self.w_votes <= 1.0:
            raise ValueError("w_votes должен быть в [0, 1]")
        if not 0.0 < self.initial_strength <= self.strength_cap <= 1.0:
            raise ValueError("требуется 0 < initial_strength <= strength_cap <= 1")
        if not 0.0 < self.assoc_confidence_penalty <= 1.0:
            raise ValueError("assoc_confidence_penalty должен быть в (0, 1]")
        if not 0.0 < self.w_keyword <= 1.0:
            raise ValueError("w_keyword должен быть в (0, 1]")
        if self.recall_oversample < 1 or self.max_pairs_per_bind < 1:
            raise ValueError("recall_oversample и max_pairs_per_bind должны быть >= 1")
        if self.backups_keep < 0:
            raise ValueError("backups_keep должен быть >= 0")

    @classmethod
    def dev(cls) -> MemoryConfig:
        """Малые размеры для тестов и демо (быстро, всё влезает в память ноутбука)."""
        return cls()

    @classmethod
    def production(cls) -> MemoryConfig:
        """Целевые масштабы фазы 3: ~10^5–10^6 следов, десятки МБ весов."""
        cfg = cls(
            dim=1024,
            n_units=8192,
            k_sparse=128,
            bucket_cap=256,
        )
        cfg.validate()
        return cfg

    def snapshot_fields(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_snapshot(cls, data: dict) -> MemoryConfig:
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.validate()
        return cfg
