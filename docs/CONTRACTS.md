# Контракты модулей realMemory v0.4

Этот документ фиксирует публичные интерфейсы. Правило: изменения сигнатур
публичных API требуют правки этого файла в том же коммите. Всё, что не описано
здесь, — приватное и может меняться свободно.

Общие соглашения:
- время — `float`, секунды epoch (UTC); все таймауты конфига — секунды;
- эмбеддинги — `np.ndarray[float32]` формы `(dim,)`;
- SDR — `np.ndarray[int32]`, отсортированные уникальные индексы on-битов;
- скоупы — `scope: str`, `'global'` или имя проекта (`[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`);
- детерминизм: все случайные структуры параметризуются seed'ами из конфига;
  одинаковый конфиг + одинаковая последовательность вызовов → одинаковое состояние;
- ошибки: невалидный вход → `ValueError`; отсутствующий id → `KeyError`;
  повреждённое хранилище → `StorageError`.

---

## types.py

```python
KIND_EPISODIC = "episodic"; KIND_SEMANTIC = "semantic"
STATUS_ACTIVE = "active"; STATUS_SUPERSEDED = "superseded"
SCOPE_GLOBAL = "global"

SOURCE_DIRECT = "direct"        # семантика прошла cos_min_recall
SOURCE_ASSOCIATED = "associated"# волна ассоциаций по рёбрам L2
SOURCE_KEYWORD = "keyword"      # точный токен при косинусе ниже порога

class DecisionAction(Enum): CREATE | REINFORCE | LINK

@dataclass(frozen=True) WriteDecision:
    action: DecisionAction
    target_id: int | None        # существующий след для REINFORCE/LINK
    related_ids: tuple[int, ...] # к кому привязывать новый след
    novelty: float               # 1 − best_cosine
    best_cosine: float

@dataclass(frozen=True) WriteResult:
    memory_id: int
    decision: WriteDecision
    created: bool                # True, если след создан

@dataclass(frozen=True) RecalledMemory:
    memory_id: int; text: str; kind: str
    cosine: float                # [0..1], точная близость к запросу
    confidence: float            # композит (см. Hippocampus.recall)
    retention: float             # [0..1]
    source: str                  # "direct" | "associated" | "keyword"
    created_at: float; updated_at: float; meta: dict
    scope: str                   # скоуп следа

@dataclass(frozen=True) RecallPacket:
    query: str; items: tuple[RecalledMemory, ...]
    abstained: bool; latency_ms: float

@dataclass(frozen=True) ConsolidationReport:
    edges_committed: int; edges_pruned: int
    promoted_to_semantic: int; rewards_applied: int; elapsed_ms: float

@dataclass MemoryRecord:  # внутренняя единица хранения
    id, text, kind, status, meta,
    embedding, sdr,
    created_at, updated_at, reinforced_count, last_reinforced_at,
    base_strength, valid_from, valid_to, superseded_by,
    scope = SCOPE_GLOBAL
```

Инварианты: `0 ≤ cosine, confidence, retention ≤ 1`; abstained означает «нет
надёжных воспоминаний» (пусто после фильтров либо сработало правило плоского
шума — см. recall); у суперседнутого следа `superseded_by != None` и `valid_to != None`.

## config.py — `MemoryConfig`

Единый dataclass всех гиперпараметров (см. докстринги полей). Ключевые группы:
кодирование (`dim, n_units, k_sparse, sdr_seed`), L1 (`bucket_cap,
recall_oversample`), гейт (`theta_reinforce ≥ theta_link > cos_min_recall`),
воздержание (`cos_min_strong_recall ≥ cos_min_recall`, `abstain_spread_cos`),
затухание (`tau_episodic < tau_semantic`, `min_retention_recall`,
`initial_strength ≤ strength_cap`), связи (`tau_eligibility < tau_edge_stable`,
`edge_min_weight`, `max_pairs_per_bind`), spread (`depth, alpha, top_m, eps`),
ранжирование (`w_votes, assoc_confidence_penalty, w_keyword`),
эксплуатация (`backups_keep`: 0 выключает копии перед «сном»).

Фабрики: `MemoryConfig.dev()` — малые размеры для тестов/демо;
`MemoryConfig.production()` — целевые масштабы фазы 3.
Инвариант `validate()`: `theta_reinforce > theta_link > cos_min_recall`;
`k_sparse ≤ n_units`; радиусы ∈ (0, 0.5].

## timeprov.py

```python
class TimeProvider(Protocol): def now(self) -> float
class SystemClock(TimeProvider)          # time.time()
class FakeClock(TimeProvider)            # .advance(seconds) для тестов
```

## encoding.embedder

```python
class EmbeddingProvider(Protocol):
    name: str                                      # стабильная идентичность (модель+версия);
                                                   # база записывает её в db_meta и
                                                   # отказывается открываться с другой
    dim: int
    def embed(self, text: str) -> np.ndarray       # факты; L2-нормализованный или нулевой
    def embed_query(self, text: str) -> np.ndarray # опционально; фасад применяет для
                                                   # поисковых запросов, если есть
class HashingEmbedder(dim=256, seed=7)              # дефолт: feature hashing слов+3-грамм,
                                                    # blake2b, детерминирован кроссплатформенно;
                                                    # name = "hashing(dim=D,seed=S)"
class FastEmbedProvider(model_name=DEFAULT_MODEL, dim=None, cache_dir=None)
                                                    # локальный ONNX (extra [local]);
                                                    # кэш по умолчанию ~/.cache/realmemory/fastembed;
                                                    # e5-префиксы включаются только для e5-моделей;
                                                    # name включает версию fastembed;
                                                    # ClassVar recommended_thresholds —
                                                    # профиль порогов под анизотропию модели,
                                                    # применяется mcp_server.main() до открытия базы
```

Контракт `embed`: пустой/blank текст → нулевой вектор; одинаковый текст →
бит-в-бит одинаковый вектор; близкие тексты (лексически) → высокая косинусная
близость; непересекающиеся словари → |cos| < 0.15.

## encoding.sdr

```python
class BipolarProjector(dim, n_bits, seed):          # утилита, ядром не используется
    def project(vec) -> np.ndarray[int8]            # sign(W·v̂)
class SDREncoder(dim, n_units, k, seed):
    def encode(vec) -> np.ndarray[int32]            # top-k по W·v̂, отсортирован;
                                                    # нулевой вход → пустой массив
def overlap(a, b) -> int
def overlap_fraction(a, b) -> float                 # |A∩B| / min(|A|,|B|)
def calibrate_sparse(encoder, n_samples=120, seed=99) -> CalibrationStats(mu, sigma)
```

Монотонность (проверяется тестом): рост косинусной близости входов →
неубывание overlap их SDR в статистическом смысле.

## core.addressing — SDRVotingIndex (L1)

```python
@dataclass(frozen=True) QueryResult:
    candidates: np.ndarray[int64]   # указатели, убывание голосов
    votes: np.ndarray[int32]        # голоса, параллельно candidates
    active_locations: int           # юниты запроса с непустыми бакетами

class SDRVotingIndex(n_units, bucket_cap):
    def write(sdr, pointer: int) -> int             # указатель в бакеты on-битов;
                                                    # без дубликатов; cap → FIFO вытеснение;
                                                    # вернул #затронутых бакетов
    def query(sdr, max_candidates) -> QueryResult   # голоса = #общих юнитов
    def load_factor() -> float                      # средняя длина бакета
    def state_dict() / load_state_dict(state)
```

Селективность (проверяется тестом): идентичный паттерн → k голосов;
коррелированный с долей ρ общих юнитов → ~ρ·k; случайный → ~k²/N.
Контракт точного запроса: `write(A, p)` затем `query(A)` всегда возвращает
`p` первым при отсутствии вытеснения.

## core.plasticity

```python
@dataclass EligibilityEvent: src_units, dst_units: np.ndarray[int32]
                            strength: float; created_at: float; source_ids: frozenset[int]

class EligibilityLog(tau):
    def add(src_units, dst_units, strength, now, source_ids) -> None
    def reward(source_ids: Iterable[int], reward: float, now) -> int  # strength *= max(0,(1+r)); вернул #событий
    def commit(now) -> (src, dst, weights)   # np-массивы, слитые по ключам (сумма),
                                             # каждый вклад затухает exp(-(now-created)/tau);
                                             # вклады < 1e-12 отбрасываются; лог очищается
    def pending_count -> int
    def state_dict() / load_state_dict(state)

def merge_pairs(src, dst, w) -> (src2, dst2, w2)     # агрегация дубликатов пар суммой
```

Контракт reward: положительный reward только усиливает; отрицательный может
обнулить вклад (не ниже 0).

## core.assembly — AssemblyNetwork (L2)

```python
class AssemblyNetwork(n_units, edge_min_weight, tau_edge_stable, seed, max_pairs_per_bind):
    def bind(units_a, units_b, strength, now) -> int       # #НОВЫХ направленных рёбер
                                                           # (слияние дубликатов не считается);
                                                           # запись двигает сетевые часы —
                                                           # распад существующих рёбер непрерывен
    def commit_eligibility(src, dst, w, now) -> None       # decay_tick + накопление
    def decay_tick(now) -> int                             # эксп. распад всех рёбер, обрезка; #оставшихся
    def spread(query_units, depth, alpha, top_m, eps) -> (units, scores)  # убывание scores;
                                                                          # query-юниты входят с 1.0,
                                                                          # суммы могут превышать 1
    def neighbors(unit) -> Iterator[(unit, weight)]
    def edge_count / total_weight -> float
    def state_dict() / load_state_dict(state)
```

Контракт: bind симметричен (оба направления); после `decay_tick(t2)`
вес ребра w умножен ровно на `exp(-(t2-t1)/tau)` относительно предыдущего тика;
spread не возвращает юниты с score < eps.

## policies.novelty

```python
def gate(best_cosine: float, cfg: MemoryConfig) -> DecisionAction
# ≥ theta_reinforce → REINFORCE; ≥ theta_link → LINK; иначе CREATE
```

## policies.decay

Чистые функции над примитивами записи:

```python
def retention(base_strength, last_reinforced_at, now, kind, cfg) -> float
    # base * exp(-(now-last)/tau(kind)), клип [0,1]; tau_semantic > tau_episodic
def reinforce_values(base, count, cfg) -> (base', count')
    # base = min(base*bump, cap); count += 1
def weaken_value(base, reward) -> base'
    # reward ∈ [-1, 0): base' = max(0, base*(1+reward)); таймеры не трогаются
def should_promote(kind, count, created_at, now, cfg) -> bool
    # эпизод → семантика: count >= promote_after_reinforcements и возраст >= promote_min_age_s
```

## store.sqlite_store — MemoryStore

```python
class MemoryStore(path, dim):                        # контекстный менеджер; WAL + busy_timeout;
                                                     # мутации в BEGIN IMMEDIATE — безопасно
                                                     # для нескольких процессов
    # следы
    def insert(rec: MemoryRecord) -> int                     # присваивает id
    def get(id) -> MemoryRecord | None
    def get_many(ids) -> list[MemoryRecord]                  # порядок как в ids, отсутствующие пропущены
    def update_trace(id, base_strength, reinforced_count, last_reinforced_at, kind=None) -> None
    def adjust_base(id, base_strength, updated_at) -> None   # ослабление без сброса таймера
    def mark_superseded(id, by_id, when) -> None
    def iter_active(batch=256) -> Iterator[MemoryRecord]
    def count(status=None, kind=None, scope=None) -> int
    def scope_counts() -> dict[str, int]                     # активные по скоупам
    def all_active_ids() -> np.ndarray[int64]
    def top_by_reinforcements(limit=10) -> list[MemoryRecord]   # для отчёта
    def stale_episodic(limit=10) -> list[MemoryRecord]          # кандидаты на забывание
    # мета и счётчики (db_meta)
    def get_meta(key) -> str | None / set_meta(key, value)
    def bump_meta_int(key, delta=1) -> int                   # атомарный инкремент, новое значение
    def consume_meta_int(key) -> int                         # прочитать и обнулить
    # рёбра L2 (таблица edges: key=src*n_units+dst)
    def edges_rev() -> int                                   # версия для инвалидации CSR-кэша
    def edges_load() -> (keys int64, ws float32)             # отсортированы по key
    def edges_apply(src, dst, w, now, tau, min_weight, stride) -> (committed, pruned)
                                                             # распад от last_edge_tick + обрезка +
                                                             # вливание батча + тик, одна транзакция
    def edges_import(keys, ws, last_tick)                    # миграция legacy-снапшота без распада
    def edges_stats(now, tau) -> (count, effective_total_weight)
    # eligibility write-through (таблицы eligibility/elig_sources)
    def elig_add(src, dst, strength, created_at, source_ids) -> None
    def elig_reward(mem_ids, factor) -> int                  # strength *= factor по источникам
    def elig_drain() -> list[event]                          # выкачать с удалением,
                                                             # формат EligibilityLog.load_state_dict
    def elig_pending() -> int
    # журнал пластичности (таблица events)
    def event_append(event_type, fields, ts) -> None
    def event_count() -> int
    def iter_events() -> Iterator[dict]                      # {"ts","type",**data} по seq
    def gate_decisions() -> dict[str, int]                   # из write-событий по $.action
    def recall_stats() -> (count, avg_latency_ms)
class StorageError(Exception)
```

    def max_updated_at() -> float | None                         # для троттлинга сна
    def backup(dest_dir=None, keep=10) -> Path                   # консистентная копия
                                                                 # (sqlite backup API) + ротация;
                                                                 # вызывается на consolidate()
class StorageError(Exception)
```

Сериализация: embedding→float32 blob, sdr→int32 blob.
`iter_active` не возвращает superseded. Базы до появления колонок мигрируют
ALTER'ом при открытии (scope → default 'global'); изменению схемы предшествует
автоматический бэкап. `db_meta['schema_version']` — версия схемы.
`db_meta['embedder']` — имя провайдера, создавшего векторы; `db_meta['config']`
— геометрия (dim/n_units/sdr_seed); Hippocampus отклоняет несовместимое
открытие (RuntimeError).

## hippocampus — Hippocampus (фасад)

```python
class Hippocampus:
    @classmethod
    def open(cls, path: str, *, config: MemoryConfig | None = None,
             embedder: EmbeddingProvider | None = None,
             clock: TimeProvider | None = None,
             namespace: str | None = None,
             verify_embedder: bool = True) -> "Hippocampus"
        # открывает или создаёт базу в каталоге path (path/namespace при namespace,
        # [A-Za-z0-9][A-Za-z0-9_.-]{0,63}, иначе ValueError);
        # перестраивает производные структуры (L1-бакеты, юнит-индекс, CSR рёбер);
        # проверяет идентичность эмбеддера и геометрию конфига;
        # однократно импортирует наследие v0.3 (journal.jsonl, snapshot.pkl);
        # verify_embedder=False — для инструментов без эмбеддингов (хуки),
        # проверка не выполняется и метка не перезаписывается

    def remember(self, text, *, kind=KIND_EPISODIC, meta=None, when=None,
                 force_new=False, related_ids: Sequence[int] = (),
                 scope: str = SCOPE_GLOBAL) -> WriteResult
        # гейт сравнивает текст только со своим скоупом и global;
        # ValueError на blank text, неизвестный kind, невалидный scope;
        # KeyError если id из related_ids не существует

    def recall(self, query, *, k=5, include_superseded=False,
               scope: str | None = None, all_scopes: bool = False) -> RecallPacket
        # запрос кодируется embed_query(), когда эмбеддер его предоставляет;
        # scope='<проект>' — следы проекта + global; None/all_scopes — вся память;
        # keyword-канал FTS5: полный матч всех токенов бустит confidence,
        # частичный при cos < cos_min_recall возвращается как source='keyword';
        # воздержание дополнительно срабатывает при «плоском шуме»: top1 direct
        # < cos_min_strong_recall, разброс косинусов выдачи < abstain_spread_cos
        # и нет полного keyword-матча

    def link_memories(self, ids: Sequence[int], strength=1.0) -> int
        # pairwise bind между SDR следов; KeyError на неизвестный id

    def feedback(self, ids: Sequence[int], reward: float) -> int
        # reward > 0 — подкрепление (bump, сброс таймера); reward < 0 — ослабление
        # base*(1+reward) без сброса таймера и счётчика продвижения;
        # reward == 0 — нейтрально; всегда плюс усиление/ослабление свежих
        # eligibility-событий; возвращает число затронутых следов

    def update_fact(self, old_id: int, new_text: str, meta=None) -> WriteResult
        # force_new + mark_superseded(old→new); новый след наследует scope старого

    def consolidate() -> ConsolidationReport     # всё состояние уже в БД, сохранять нечего;
                                                 # параллельные «сны» сериализуются транзакцией
    def stats() -> dict                          # глобально из БД: одинаково для всех процессов
    def metrics_snapshot(now=None) -> dict       # срез для dream_log/отчёта
    def pending_eligibility -> int               # незакоммиченные bind'ы (общие для процессов)
```

Формула уверенности recall (зафиксирована):
`confidence = cosine · (w_votes + (1−w_votes)·votes_norm) · (0.3 + 0.7·retention)`;
для full keyword-match confidence поднимается максимум до
`w_keyword · (0.3 + 0.7·retention)`; для associated-элементов дополнительно
×`assoc_confidence_penalty`, косинус заменяется полом 0.05 (связь сама является
сигналом релевантности, поэтому косинусный фильтр к ассоциативной волне не
применяется — только retention).
Абстейн: (а) нет кандидатов, прошедших фильтры direct-волны
`cos ≥ cos_min_recall` и `retention ≥ min_retention_recall`; (б) правило
плоского шума из докстринга recall.

## api.mcp_server

```python
def build_server(hippo, default_project=None):  # ImportError("pip install 'realmemory[mcp]'") без fastmcp
def make_embedder(choice):              # "local" | "hashing" | "auto"
def main(argv=None):                    # CLI: --path, --namespace, --project, --embedder; stdio MCP
                                        # default_project = resolve_project(--project):
                                        # явный аргумент > REALMEMORY_PROJECT >
                                        # ZCODE_PROJECT_DIR/CLAUDE_PROJECT_DIR > cwd c .git/.zcode
```

Тулы (имена — когнитивные действия, описания на английском):
`recall(query, k, include_superseded, project)` ·
`memorize(text, kind, related_ids, project)` ·
`reflect(memory_ids, reward)` → `{touched}` · `revise(old_id, new_text)` ·
`introspect()` (включает текущий проект и разбивку по скоупам) · `dream_log()` —
тонкие обёртки над фасадом, JSON-сериализация пакетов.
Политика скоупов в описаниях тулов: факты о рабочем проекте — с project=<имя>,
пользовательские предпочтения/идентичность — без project → global.

## hook_cli (автоматизация для агентов)

```python
# python -m realmemory.hook_cli <cmd> --path P [--namespace NS]
brief  [--project P] [--top N] [--episodic-top M] [--plain]
                                # SessionStart: strict JSON
                                # {"hookSpecificOutput":{"hookEventName":
                                #  "SessionStart","additionalContext":str}};
                                # семантические факты (•) затем прочные эпизоды (·),
                                # score эпизода = retention·(1+ln(1+подкреплений));
                                # фильтр по скоупу проекта+global, бюджет ~600 символов;
                                # конфиг читается из db_meta, модель НЕ грузится
sleep  [--verbose]
                                # Stop: consolidate(); троттлинг по состоянию базы —
                                # пропуск если нет записей после last_consolidate_at
                                # и пуст eligibility; вывод пустой, код возврата всегда 0
```

## report (python -m realmemory.report)

```python
def build_report(path, namespace=None) -> dict    # следы по типам/статусам, top_reinforced,
                                                  # stale_episodic, индекс L1, snapshot-счётчики,
                                                  # статистика recall/feedback/консолидаций из журнала,
                                                  # metrics-временной ряд
def render(report) -> str          # человекочитаемый текст
def main(argv=None)                # CLI: --path, --namespace, --json out.json
```

Наблюдаемость фасада: каждый recall пишет в журнал событие
`{latency_ms, items, abstained, top_conf}`; каждый consolidate — полный
`metrics`-срез (`Hippocampus.metrics_snapshot()`).

## eval.bench_recall / eval.bench_real

```python
# bench_recall: синтетика, hashing-эмбеддер
def run(n_facts=1500, n_queries=200, seed=0, config=None) -> dict  # метрики (см. вывод)

# bench_real: fixtures/bench_real.json, fastembed обязателен
# python -m realmemory.eval.bench_real [--k 10] [--verbose] [--json out.json]
```

Метрики-контракт bench_recall: `pipeline_hits@k`, `baseline_hits@k` (точный
косинус тем же эмбеддером), латентности p50/p95, writes/sec, abstention на noise.
Метрики bench_real: hits@10/MRR раздельно для переформулировок и точных токенов,
воздержание на шуме, гейт раздельно для базы и дубликатов-переформулировок,
калибровочные распределения (нуль «факт—сосед» без дупл., сигнал дубликатов,
max-cos запросов). Ссылки в фикстуре — по строковым id фактов.
