# Контракты модулей realMemory v0.1

Этот документ фиксирует публичные интерфейсы. Правило: изменения сигнатур
публичных API требуют правки этого файла в том же коммите. Всё, что не описано
здесь, — приватное и может меняться свободно.

Общие соглашения:
- время — `float`, секунды epoch (UTC); все таймауты конфига — секунды;
- эмбеддинги — `np.ndarray[float32]` формы `(dim,)`;
- SDR — `np.ndarray[int32]`, отсортированные уникальные индексы on-битов;
- детерминизм: все случайные структуры параметризуются seed'ами из конфига;
  одинаковый конфиг + одинаковая последовательность вызовов → одинаковое состояние;
- ошибки: невалидный вход → `ValueError`; отсутствующий id → `KeyError`;
  повреждённое хранилище → `StorageError`.

---

## types.py

```python
KIND_EPISODIC = "episodic"; KIND_SEMANTIC = "semantic"
STATUS_ACTIVE = "active"; STATUS_SUPERSEDED = "superseded"

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
    source: str                  # "direct" | "associated"
    created_at: float; updated_at: float; meta: dict

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
    base_strength, valid_from, valid_to, superseded_by
```

Инварианты: `0 ≤ cosine, confidence, retention ≤ 1`; `abstained == (len(items)==0)`;
у суперседнутого следа `superseded_by != None` и `valid_to != None`.

## config.py — `MemoryConfig`

Единый dataclass всех гиперпараметров (см. докстринги полей). Ключевые группы:
кодирование (`dim, n_units, k_sparse, sdr_seed`), L1 (`bucket_cap,
recall_oversample`), гейт (`theta_reinforce ≥ theta_link > cos_min_recall`),
затухание (`tau_episodic < tau_semantic`, `min_retention_recall`,
`initial_strength ≤ strength_cap`), связи (`tau_eligibility < tau_edge_stable`,
`edge_min_weight`, `max_pairs_per_bind`), spread (`depth, alpha, top_m, eps`),
ранжирование (`w_votes, assoc_confidence_penalty`).

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
                                                    # name включает версию fastembed
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
class MemoryStore(path, dim):                        # контекстный менеджер
    def insert(rec: MemoryRecord) -> int                     # присваивает id
    def get(id) -> MemoryRecord | None
    def get_many(ids) -> list[MemoryRecord]                  # порядок как в ids, отсутствующие пропущены
    def update_trace(id, base_strength, reinforced_count, last_reinforced_at, kind=None) -> None
    def adjust_base(id, base_strength, updated_at) -> None   # ослабление без сброса таймера
    def mark_superseded(id, by_id, when) -> None
    def get_meta(key) -> str | None                          # таблица db_meta (key,value)
    def set_meta(key, value) -> None                         # upsert
    def iter_active(batch=256) -> Iterator[MemoryRecord]
    def count(status=None, kind=None) -> int
    def all_active_ids() -> np.ndarray[int64]
    def top_by_reinforcements(limit=10) -> list[MemoryRecord]   # для отчёта
    def stale_episodic(limit=10) -> list[MemoryRecord]          # кандидаты на забывание
class StorageError(Exception)
```

Сериализация: embedding→float32 blob, sdr→int32 blob.
`iter_active` не возвращает superseded.
`db_meta['embedder']` — имя провайдера, создавшего векторы; Hippocampus
отклоняет открытие с другим именем (RuntimeError).

## store.journal — Journal

```python
class Journal(path):
    def append(event_type: str, **fields) -> None    # одна JSON-строка + flush
    def events() -> Iterator[dict]
```

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
        # восстанавливает L1/L2 состояние; проверяет идентичность эмбеддера;
        # verify_embedder=False — для инструментов без эмбеддингов (хуки),
        # проверка не выполняется и метка не перезаписывается

    def remember(self, text, *, kind=KIND_EPISODIC, meta=None, when=None,
                 force_new=False, related_ids: Sequence[int] = ()) -> WriteResult
        # ValueError на blank text и неизвестный kind; KeyError если id из related_ids
        # не существует

    def recall(self, query, *, k=5, include_superseded=False) -> RecallPacket
        # запрос кодируется embed_query(), когда эмбеддер его предоставляет

    def link_memories(self, ids: Sequence[int], strength=1.0) -> int
        # pairwise bind между SDR следов; KeyError на неизвестный id

    def feedback(self, ids: Sequence[int], reward: float) -> int
        # reward > 0 — подкрепление (bump, сброс таймера); reward < 0 — ослабление
        # base*(1+reward) без сброса таймера и счётчика продвижения;
        # reward == 0 — нейтрально; всегда плюс усиление/ослабление свежих
        # eligibility-событий; возвращает число затронутых следов

    def update_fact(self, old_id: int, new_text: str, meta=None) -> WriteResult
        # force_new + mark_superseded(old→new); KeyError на old_id

    def consolidate(self, save=True) -> ConsolidationReport
    def stats(self) -> dict
    def save(self) -> None          # снапшот нейронного состояния
```

Формула уверенности recall (зафиксирована):
`confidence = cosine · (w_votes + (1−w_votes)·votes_norm) · (0.3 + 0.7·retention)`,
для associated-элементов дополнительно ×`assoc_confidence_penalty`, косинус
заменяется полом 0.05 (связь сама является сигналом релевантности, поэтому
косинусный фильтр к ассоциативной волне не применяется — только retention).
Абстейн: нет кандидатов, прошедших фильтры direct-волны
`cos ≥ cos_min_recall` и `retention ≥ min_retention_recall`.
Ассоциативная волна использует только рёбра, закоммиченные консолидацией:
связи до первого «сна» живут в eligibility-логе и в recall не участвуют.

## api.mcp_server

```python
def build_server(hippo):                # ImportError("pip install 'realmemory[mcp]'") без пакета fastmcp
def make_embedder(choice):              # "local" | "hashing" | "auto"
def main(argv=None):                    # CLI: --path, --namespace, --embedder {auto,local,hashing}; stdio MCP
```

Тулы (имена — когнитивные действия, описания на английском):
`recall(query, k, include_superseded)` · `memorize(text, kind, related_ids)` ·
`reflect(memory_ids, reward)` → `{touched}` · `revise(old_id, new_text)` ·
`introspect()` · `dream_log()` — тонкие обёртки над фасадом,
JSON-сериализация пакетов; recall отдаёт items с `meta` и датами.

## hook_cli (автоматизация для агентов)

```python
# python -m realmemory.hook_cli <cmd> --path P [--namespace NS]
brief  [--top N] [--plain]      # SessionStart: strict JSON
                                # {"hookSpecificOutput":{"hookEventName":
                                #  "SessionStart","additionalContext":str}};
                                # конфиг читается из snapshot.pkl, модель НЕ грузится;
                                # additionalContext на английском: объёмы + подсказка тулов
sleep  [--min-interval-s 1800] [--verbose]
                                # Stop: consolidate(save=True) c троттлингом —
                                # пропуск если <min-interval-s и db не новее снапшота;
                                # вывод пустой, код возврата всегда 0
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

## eval.bench_recall

```python
def run(n_facts=1500, n_queries=200, seed=0, config=None) -> dict  # метрики (см. вывод)
if __name__ == "__main__": argparse CLI
```

Метрики-контракт: `pipeline_hits@k`, `baseline_hits@k` (точный косинус тем же
эмбеддером), латентности p50/p95 recall, writes/sec, abstention на noise-запросах.
Успех фазы 0: pipeline ≥ baseline − ε на статическом recall и abstention ≥ 90%
на noise.
