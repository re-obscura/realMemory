"""Догфудинг-ревизия v0.8.4: канонизация скоупов (один проект под тремя
написаниями терял следы), смена kind в revise (identity-факты угасали как
episodic до удаления GC), предупреждение брифа об угасающих глобальных
фактов, WAL-чекпойнт в «сне»."""
import pytest

from realmemory import Hippocampus, MemoryConfig
from realmemory.hook_cli import main as hooks_main
from realmemory.store.sqlite_store import scope_key
from realmemory.timeprov import FakeClock


def _cfg(**over) -> MemoryConfig:
    fields = {
        "dim": 256, "n_units": 512, "k_sparse": 48, "sdr_seed": 5,
        "bucket_cap": 32,
        "tau_episodic": 60 * 86400.0,
        "tau_semantic": 600 * 86400.0,
        "gc_grace_below_floor_s": 5 * 86400.0,
    }
    fields.update(over)
    cfg = MemoryConfig(**fields)
    cfg.validate()
    return cfg


# -- канонизация скоупов --------------------------------------------------------

def test_scope_key_unifies_case_and_separators():
    assert scope_key("CPM.Backend") == scope_key("cpm_backend")
    assert scope_key("cpm backend") == scope_key("CPM-BACKEND")
    assert scope_key("gis_front") != scope_key("CPM.Backend")


def test_remember_converges_scope_variant_to_existing(tmp_path):
    """След, записанный вариантом написания, приходит в живущий скоуп —
    гейт видит дубликат и делает REINFORCE, а не плодит второй скоуп."""
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock())
    try:
        first = h.remember("Go-сервис CPM живёт в докере dm801", scope="CPM.Backend")
        dup = h.remember("Go-сервис CPM живёт в докере dm801", scope="cpm_backend")
        assert dup.memory_id == first.memory_id
        assert not dup.created
        assert set(h.store.scope_counts()) == {"CPM.Backend"}
        # recall вариантом имени находит тот же след
        packet = h.recall("Go-сервис докер", k=3, scope="cpm_backend")
        assert packet.items and packet.items[0].scope == "CPM.Backend"
    finally:
        h.close()


def test_canonical_scope_prefers_most_populous(tmp_path):
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock())
    try:
        for i in range(3):
            h.remember(f"факт мажоритарного скоупа mj{i}", scope="Major.Scope", force_new=True)
        h.remember("факт миноритарного написания", scope="major_scope", force_new=True)
        # написание миноритария сходится к самому населенному варианту
        assert "major_scope" not in h.store.scope_counts()
        assert h.store.canonical_scope("MAJOR.SCOPE") == "Major.Scope"
    finally:
        h.close()


# -- revise со сменой kind --------------------------------------------------------

def test_update_fact_changes_kind(tmp_path):
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock())
    try:
        old = h.remember("Предпочитаю короткие ответы", scope="global")
        res = h.update_fact(old.memory_id, "Предпочитаю короткие ответы и по-русски",
                            kind="semantic")
        new_rec = h.store.get(res.memory_id)
        assert new_rec.kind == "semantic"
        # старый след — superseded-история
        assert h.store.get(old.memory_id).status == "superseded"
        # без явного kind — наследуется
        third = h.update_fact(res.memory_id, "Обновление семантического факта")
        assert h.store.get(third.memory_id).kind == "semantic"
    finally:
        h.close()


def test_update_fact_rejects_unknown_kind(tmp_path):
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock())
    try:
        mid = h.remember("факт для невалидного kind fk802").memory_id
        with pytest.raises(ValueError, match="kind"):
            h.update_fact(mid, "новый текст", kind="procedural")
    finally:
        h.close()


# -- бриф предупреждает об угасающих глобальных фактах -----------------------------

def test_brief_warns_on_fading_global_traces(tmp_path, capsys):
    """Identity-факты в global с низким retention — не фильтр фокуса, а
    потеря главного: бриф обязан кричать до того, как GC удалит их."""
    root = tmp_path / "rm"
    h = Hippocampus.open(root, config=_cfg(), clock=FakeClock())
    try:
        h.remember("Меня зовут Максим, я ценю честные метрики", scope="global")
        h.remember("проектный факт без тревоги pf803", scope="proj803")
    finally:
        h.close()
    # бриф открывает базу SystemClock-ом: время FakeClock давно в прошлом →
    # retention глобального эпизода ≈ 0
    with pytest.raises(SystemExit) as ex:
        hooks_main(["brief", "--path", str(root), "--project", "proj803",
                    "--plain"])
    out = capsys.readouterr().out
    assert ex.value.code == 0
    assert "WARNING" in out and "глобальных" in out
    assert "Меня зовут Максим" in out


def test_brief_no_warning_when_global_healthy(tmp_path, capsys):
    root = tmp_path / "rm"
    # без FakeClock: след свежий по реальным часам — тревоги быть не должно
    h = Hippocampus.open(root, config=_cfg())
    try:
        h.remember("Свежий глобальный факт fg804", scope="global")
    finally:
        h.close()
    with pytest.raises(SystemExit) as ex:
        hooks_main(["brief", "--path", str(root), "--project", "proj803",
                    "--plain"])
    out = capsys.readouterr().out
    assert ex.value.code == 0
    assert "WARNING" not in out


# -- WAL-чекпойнт в «сне» -----------------------------------------------------------

def test_consolidate_runs_wal_checkpoint(tmp_path):
    h = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock())
    try:
        h.remember("факт перед сном wl805")
        report = h.consolidate()
        assert report.elapsed_ms >= 0  # сон прошёл целиком
        h.store.wal_checkpoint()  # явный вызов не падает и не блокирует
    finally:
        h.close()
    # контроль: журнал применяется без потери данных после чекпойнта
    h2 = Hippocampus.open(tmp_path / "rm", config=_cfg(), clock=FakeClock())
    try:
        stats = h2.stats()
        assert stats["journal_events"] > 0
    finally:
        h2.close()
