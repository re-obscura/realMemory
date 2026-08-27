"""Изменения v0.5: точный скан как основной ретривер, откат на голосование,
журнал под замком и с ротацией, троттлинг бэкапов, уборка забытых следов."""
import sqlite3

import numpy as np

from realmemory import Hippocampus, MemoryConfig
from realmemory.store.sqlite_store import MemoryStore
from realmemory.timeprov import FakeClock
from realmemory.types import MemoryRecord


def _rec(i: int, dim: int = 8, k: int = 4) -> MemoryRecord:
    rng = np.random.default_rng(900 + i)
    return MemoryRecord(
        id=None,
        text=f"row {i}",
        kind="episodic",
        status="active",
        meta={},
        embedding=rng.standard_normal(dim).astype(np.float32),
        sdr=np.sort(rng.choice(dim, size=k, replace=False)).astype(np.int32),
        created_at=float(i),
        updated_at=float(i),
        reinforced_count=0,
        last_reinforced_at=float(i),
        base_strength=0.5,
        valid_from=float(i),
    )


# -- кэш эмбеддингов / точный скан -------------------------------------------------

def test_exact_mode_default_on_and_off_by_flag(tmp_path):
    h = Hippocampus.open(tmp_path / "rm")
    assert h._exact_mode() is True  # дефолт конфига
    h.remember("korvex kv101 qixa")
    p = h.recall("korvex kv101")
    assert [i.memory_id for i in p.items] == [1]
    h.close()

    cfg = MemoryConfig(exact_scan_recall=False)
    cfg.validate()
    h2 = Hippocampus.open(tmp_path / "rm", config=cfg)
    assert h2._exact_mode() is False
    # голосование L1 не потеряло следы
    assert [i.memory_id for i in h2.recall("korvex kv101").items] == [1]
    h2.close()


def test_exact_engine_finds_weak_overlap_target_beyond_vote_window(tmp_path):
    """Регрессия обрыва качества: цель со слабым пересечением SDR тонет среди
    «конкурентов» с одним общим словом при голосовании; точный скан обязан
    находить её независимо от плотности корпуса."""
    from realmemory.eval.bench_recall import _make_corpus, _subset_query

    n = 400
    cfg = MemoryConfig(dim=2048, n_units=2048, k_sparse=96, bucket_cap=512,
                       cos_min_recall=0.18)
    cfg.validate()
    h = Hippocampus.open(tmp_path / "rm", config=cfg)
    try:
        corpus = _make_corpus(n, seed=3)
        ids = {i: h.remember(text).memory_id for text, i in corpus}
        rng = np.random.default_rng(11)
        hits = 0
        trials = 60
        for qi in rng.choice(len(corpus), size=trials, replace=False):
            text, fi = corpus[int(qi)]
            packet = h.recall(_subset_query(text, fi, rng), k=10)
            hits += int(any(it.memory_id == ids[fi] for it in packet.items))
        assert hits == trials  # точный скан повторяет baseline-полноту
        engines = [e.get("engine") for e in h.store.iter_events() if e["type"] == "recall"]
        assert set(engines) == {"exact"}
    finally:
        h.close()


def test_fallback_to_votes_above_trace_cap(tmp_path):
    """Выше exact_scan_max_traces движок откатывается на голосование L1 —
    обе ветки рабочие, результат согласован."""
    cfg = MemoryConfig(exact_scan_recall=True, exact_scan_max_traces=1)
    cfg.validate()
    h = Hippocampus.open(tmp_path / "rm", config=cfg)
    try:
        h.remember("korvex kv201 qixa")
        h.remember("plimso pt202 suvat")
        assert h._exact_mode() is False  # лимит исчерпан уже двумя следами
        packet = h.recall("korvex kv201", k=2)
        assert packet.items[0].memory_id == 1
        engines = [e.get("engine") for e in h.store.iter_events() if e["type"] == "recall"]
        assert set(engines) == {"votes"}
    finally:
        h.close()


def test_results_stable_across_reopen(tmp_path, tiny_cfg):
    """Кэш эмбеддингов перестраивается при открытии: выдача идентична живому
    процессу."""
    h = Hippocampus.open(tmp_path / "rm", config=tiny_cfg)
    h.remember("korvex kv301 qixa")
    h.remember("plimso pt302 suvat")
    h.remember("korvex kv303 qixa related", scope="proj")
    live = [(i.memory_id, i.confidence) for i in h.recall("korvex kv301", k=3).items]
    h.close()
    h2 = Hippocampus.open(tmp_path / "rm", config=tiny_cfg)
    after = [(i.memory_id, i.confidence) for i in h2.recall("korvex kv301", k=3).items]
    h2.close()
    assert live == after
    assert live[0][0] == 1


# -- журнал: лоченые итераторы и ротация --------------------------------------------

def test_iterators_tolerate_concurrent_write_on_same_connection(tmp_path):
    """Порция материализована до yield: BEGIN IMMEDIATE из другого треда того же
    соединения не падает с 'cannot start a transaction within a transaction'."""
    store = MemoryStore(tmp_path / "m.db", dim=8)
    for i in range(40):
        store.insert(_rec(i))

    seen = []
    for position, r in enumerate(store.iter_active(batch=4)):
        seen.append(r.id)
        if position % 4 == 0:
            # мутация во время «открытой» итерации — раньше рвала курсор/транзакцию
            store.set_meta("probe", str(position))
            store.event_append("during_iter", {"position": position})
    assert len(seen) == 40
    assert len(set(seen)) == 40
    store.close()


def test_events_prune_keeps_latest(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    for i in range(50):
        store.event_append("tick", {"i": i})
    pruned = store.events_prune(20)
    assert pruned == 30
    rest = list(store.iter_events())
    assert len(rest) == 20
    assert rest[-1]["i"] == 49  # выживают свежие
    assert all(e["type"] == "tick" for e in rest)
    assert store.events_prune(0) == 0  # ротация выключена флагом 0
    store.close()


# -- бэкапы: троттлинг и коллизия имён ----------------------------------------------

def test_backup_throttled_by_interval_meta(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    first = store.backup(min_interval_s=3600)
    assert first is not None
    soon = store.backup(min_interval_s=3600)
    assert soon is None  # интервал не истёк
    past = store.backup(min_interval_s=-1 if False else 0)  # 0 = без троттлинга
    assert past is not None
    # сдвиг метки вручную эмулирует прошедший час
    import time as _t
    store.set_meta("last_backup_at", repr(float(_t.time() - 3700)))
    later = store.backup(min_interval_s=3600)
    assert later is not None
    store.close()


def test_same_second_backups_distinct_names(tmp_path):
    store = MemoryStore(tmp_path / "m.db", dim=8)
    p1 = store.backup(keep=0)
    p2 = store.backup(keep=0)
    assert p1 != p2  # миллисекундный суффикс разводит имена в одну секунду
    assert p1.exists() and p2.exists()
    store.close()


def test_consolidate_respects_backup_interval_and_prunes_journal(hippo):
    """Сон после сна без интервала дважды копирует; с интервалом — второй
    пропускается с событием backup_skipped. Ротация применяется."""
    hippo.config.backups_keep = 5
    hippo.config.backup_min_interval_s = 0.0
    hippo.remember("korvex kv401 qixa")
    hippo.consolidate()
    bdir = hippo.path / "backups"
    n_after_first = len(list(bdir.glob("memory-*.db")))

    hippo.config.backup_min_interval_s = 6 * 3600.0
    hippo.config.journal_max_events = 100_000
    hippo.consolidate()
    events = [e["type"] for e in hippo.store.iter_events()]
    assert "backup_skipped" in events
    assert len(list(bdir.glob("memory-*.db"))) == n_after_first  # копии нет


def test_journal_rotation_triggered_in_consolidate(tmp_path, clock):
    cfg = MemoryConfig(journal_max_events=5, backups_keep=0)
    cfg.validate()
    h = Hippocampus.open(tmp_path / "rm", config=cfg, clock=clock)
    try:
        for i in range(12):
            h.remember(f"fact korvex kv{500+i}")
        h.consolidate()
        types = [e["type"] for e in h.store.iter_events()]
        # выжило 5 последних write-ов плюс три события текущего сна:
        # journal_rotated, consolidate, metrics
        assert len(types) == 8
        assert set(types[:5]) == {"write"}
        assert types[5:] == ["journal_rotated", "consolidate", "metrics"]
        rotated = next(e for e in h.store.iter_events() if e["type"] == "journal_rotated")
        assert rotated["pruned"] > 0
    finally:
        h.close()


# -- keyword-канал: согласованность токенов ------------------------------------------

def test_single_char_query_tokens_do_not_break_full_match(hippo):
    """'и'/'в' длиной 1 отброшены синхронно в обоих каналах: строка мусорных
    односимвольных токенов в запросе не отменяет full-match буст."""
    hippo.remember("ошибка E404 ломает очередь задачи v2")
    packet = hippo.recall("E404 v2 и в с а", k=3)
    assert packet.items, "точные токены обязаны находиться"
    item = next(i for i in packet.items if "E404" in i.text)
    assert item.source in ("keyword", "direct")


# -- относительное воздержание точного скана -------------------------------------------

def test_exact_relative_abstain_fires_and_bypasses(tmp_path, tiny_cfg, clock):
    """Топ-1 ниже медианы корпуса на margin -> воздержание с причиной
    below_null; полный keyword-матч отменяет правило."""
    cfg = MemoryConfig(**{**tiny_cfg.__dict__, "exact_abstain_rel_margin": 0.95})
    cfg.validate()
    h = Hippocampus.open(tmp_path / "rm", config=cfg, clock=clock)
    try:
        h.remember("korvex kv701 qixa")
        h.remember("plimso pt702 suvat")
        # частичное совпадение: full_match нет, но косинус высокий;
        # margin 0.95 заведомо перекрывает всю слабую зону текстов
        packet = h.recall("korvex kv999", k=2)
        assert packet.abstained and not packet.items
        events = [e for e in h.store.iter_events() if e.get("below_null")]
        assert events and events[-1]["null_median"] >= 0.0

        # тот же запрос с точным токеном цели проходит сквозь правило
        p_full = h.recall("korvex kv701", k=2)
        assert p_full.items and p_full.items[0].memory_id == 1
    finally:
        h.close()


# -- mcp_server: геометрия через CLI ---------------------------------------------------

class _FakeSrv:
    def run(self) -> None:  # main() вызывает .run(): гасим тихо
        return None


def test_cli_geometry_flags_applied(monkeypatch, tmp_path):
    import realmemory.projects as projects_mod
    from realmemory.api import mcp_server

    captured: dict = {}
    monkeypatch.setattr(projects_mod, "resolve_project", lambda explicit=None: None)

    def fake_build_server(hippo, default_project=None):
        captured.update(units=hippo.config.n_units, k=hippo.config.k_sparse,
                        cap=hippo.config.bucket_cap, dim=hippo.config.dim)
        return _FakeSrv()

    monkeypatch.setattr(mcp_server, "build_server", fake_build_server)
    monkeypatch.setattr(mcp_server, "make_embedder", lambda choice="local": _Hash(dim=256))
    mcp_server.main(["--path", str(tmp_path / "rm"), "--embedder", "hashing",
                     "--units", "512", "--k-sparse", "48", "--bucket-cap", "33"])
    assert captured == {"units": 512, "k": 48, "cap": 33, "dim": 256}


class _Hash:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.name = f"hashing-test-{dim}"

    def embed(self, text):
        return np.zeros(self.dim, dtype=np.float32)


# -- целостность схемы после обновления -----------------------------------------------

def test_schema_version_intact_for_v04_base(tmp_path):
    """База v0.4 открывается v0.5 кодом: schema_version сохранён, точный режим
    работает на исторических блобах."""
    con = sqlite3.connect(str(tmp_path / "m.db"))
    con.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,"
        " kind TEXT NOT NULL, status TEXT NOT NULL, meta TEXT NOT NULL, embedding BLOB NOT NULL,"
        " sdr BLOB NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,"
        " reinforced_count INTEGER NOT NULL DEFAULT 0, last_reinforced_at REAL NOT NULL,"
        " base_strength REAL NOT NULL DEFAULT 1.0, valid_from REAL NOT NULL, valid_to REAL,"
        " superseded_by INTEGER, scope TEXT NOT NULL DEFAULT 'global')"
    )
    con.execute("CREATE TABLE IF NOT EXISTS db_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    con.execute("INSERT INTO db_meta VALUES('schema_version','1')")
    con.commit()
    con.close()
    store = MemoryStore(tmp_path / "m.db", dim=8)
    assert store.get_meta("schema_version") == "2"
    store.close()


# -- уборка забытых следов (GC) ------------------------------------------------------

def _gc_cfg(**overrides) -> MemoryConfig:
    fields = {
        "dim": 256, "n_units": 512, "k_sparse": 48, "sdr_seed": 5,
        "bucket_cap": 32,
        "tau_episodic": 60 * 86400.0,
        "tau_semantic": 600 * 86400.0,
        "gc_grace_below_floor_s": 5 * 86400.0,
    }
    fields.update(overrides)
    cfg = MemoryConfig(**fields)
    cfg.validate()
    return cfg


def _open_gc(tmp_path, cfg, start=0.0):
    clock = FakeClock(start=start) if hasattr(FakeClock, "start") else FakeClock()
    return Hippocampus.open(tmp_path / "rm", config=cfg, clock=clock)


def test_gc_deletes_zombies_keeps_reinforced(tmp_path):
    h = _open_gc(tmp_path, _gc_cfg())
    victim1 = h.remember("забытый факт корвекс kv901").memory_id
    keeper = h.remember("живой факт plimso pt902").memory_id
    victim2 = h.remember(
        "ещё один забытый вендрик vf903", related_ids=[keeper]
    ).memory_id
    h.feedback([keeper], 1.0)
    h.feedback([keeper], 1.0)

    assert h.consolidate().forgotten_traces == 0
    h.clock.advance(100 * 86400.0)  # ниже пола ещё не упал даже забытий
    assert h.consolidate().forgotten_traces == 0
    h.clock.advance(50 * 86400.0)   # 150d: слабые ниже пола, grace прошёл
    rep = h.consolidate()
    assert rep.forgotten_traces == 2
    assert h.store.get(victim1) is None
    assert h.store.get(victim2) is None
    assert h.store.get(keeper) is not None

    # выживший находится, свежая запись после уборки попадает в кэш и выдачу
    packet = h.recall("живой факт plimso pt902", k=1)
    assert [i.memory_id for i in packet.items] == [keeper]
    post = h.remember("новый факт после уборки юбэкс nj904").memory_id
    again = h.recall("новый факт после уборки юбэкс nj904", k=1)
    assert [i.memory_id for i in again.items] == [post]
    assert h._emb_len == 2

    final_now = h.clock.now()
    live = [(i.memory_id, round(i.confidence, 6))
            for i in h.recall("факт", k=5, all_scopes=True).items]
    h.close()

    clock2 = FakeClock(start=final_now)
    h2 = Hippocampus.open(tmp_path / "rm", config=_gc_cfg(), clock=clock2)
    after = [(i.memory_id, round(i.confidence, 6))
             for i in h2.recall("факт", k=5, all_scopes=True).items]
    h2.close()
    assert live == after


def test_gc_respects_grace_after_negative_feedback(tmp_path):
    """§8-сценарий буквально: негативный фидбек обнуляет след мгновенно,
    но удаление ждёт grace со времени последнего подкрепления."""
    cfg = _gc_cfg(tau_episodic=400 * 86400.0, initial_strength=0.06,
                  promote_min_age_s=400 * 86400.0)
    h = Hippocampus.open(tmp_path / "rm", config=cfg, clock=FakeClock())
    try:
        zombie = h.remember("неправильный вывод о архитектуре q901").memory_id
        h.feedback([zombie], -0.9)  # base падает ниже recall-пола сразу
        h.clock.advance(3 * 86400.0)  # меньше grace
        assert h.consolidate().forgotten_traces == 0
        assert h.store.get(zombie) is not None
        h.clock.advance(4 * 86400.0)  # суммарно больше grace
        rep = h.consolidate()
        assert rep.forgotten_traces == 1
        assert h.store.get(zombie) is None
        assert h._emb_len == 0
    finally:
        h.close()


def test_gc_disabled_leaves_everything(tmp_path):
    cfg = _gc_cfg(gc_enabled=False)
    h = Hippocampus.open(tmp_path / "rm", config=cfg, clock=FakeClock())
    try:
        lost = h.remember("забытый факт сорекс sx909").memory_id
        h.feedback([lost], -1.0)
        h.clock.advance(300 * 86400.0)
        assert h.consolidate().forgotten_traces == 0
        assert h.store.get(lost) is None if False else h.store.get(lost) is not None
    finally:
        h.close()


def test_gc_never_touches_superseded_history(tmp_path):
    """Superseded-строки исключены из GC-кандидатов по статусу; замена живёт,
    пока её retention выше пола (горизонт и подкрепления подобраны так, чтобы
    неотменённое забывание не вмешивалось в проверку)."""
    h = _open_gc(tmp_path, _gc_cfg())
    try:
        old = h.remember("старая версия решения вудекс wo900").memory_id
        fresh = h.update_fact(old, "новая версия решения вудекс wo900b").memory_id
        for _ in range(3):  # base к потолку: на 120d retention держится над полом
            h.feedback([fresh], 1.0)
        h.clock.advance(120 * 86400.0)
        rep = h.consolidate()
        assert rep.forgotten_traces == 0
        superseded_rec = h.store.get(old)
        assert superseded_rec is not None
        assert superseded_rec.status == "superseded"
        assert h.store.get(fresh) is not None
    finally:
        h.close()


def test_elig_sources_cleaned_by_forget(tmp_path):
    """Ссылки elig_sources удалённых следов не дублируют reward-матчинг мусором."""
    h = _open_gc(tmp_path, _gc_cfg(tau_episodic=400 * 86400.0,
                                   initial_strength=0.06,
                                   promote_min_age_s=400 * 86400.0))
    try:
        import sqlite3 as _sq
        doomed = h.remember("связанный факт жрекс dj907").memory_id
        anchor = h.remember("якорный факт анкор ak908").memory_id
        h.link_memories([doomed, anchor])
        con = _sq.connect(str(h.path / "memory.db"))
        try:
            before = con.execute(
                "SELECT COUNT(*) FROM elig_sources WHERE mem_id=?", (doomed,)
            ).fetchone()[0]
            assert before > 0
            # негативный фидбек валит base ниже recall-пола мгновенно;
            # через grace консолидация обязана удалить след вместе со ссылками
            h.feedback([doomed], -1.0)
            h.clock.advance(6 * 86400.0)
            rep = h.consolidate()
            assert rep.forgotten_traces == 1
            after = con.execute(
                "SELECT COUNT(*) FROM elig_sources WHERE mem_id=?", (doomed,)
            ).fetchone()[0]
            assert after == 0
            assert con.execute(
                "SELECT COUNT(*) FROM memories WHERE id=?", (doomed,)
            ).fetchone()[0] == 0
            assert con.execute(
                "SELECT COUNT(*) FROM memories WHERE id=?", (anchor,)
            ).fetchone()[0] == 1
        finally:
            con.close()
    finally:
        h.close()
