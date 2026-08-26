# realMemory Module Contracts v0.4

This document fixes public interfaces. Rule: any change to a public API
signature requires editing this file in the same commit. Anything not described
here is private and may change freely.

Conventions:
- time — `float`, epoch seconds (UTC); all config timeouts are seconds;
- embeddings — `np.ndarray[float32]` of shape `(dim,)`;
- SDR — `np.ndarray[int32]`, sorted unique indices of on-bits;
- scopes — `scope: str`, either `'global'` or a project name
  (`[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`);
- determinism: all random structures are parameterized by config seeds; same
  config + same call sequence → same state;
- errors: invalid input → `ValueError`; missing id → `KeyError`;
  corrupted storage → `StorageError`.

---

## types.py

```python
KIND_EPISODIC = "episodic"; KIND_SEMANTIC = "semantic"
STATUS_ACTIVE = "active"; STATUS_SUPERSEDED = "superseded"
SCOPE_GLOBAL = "global"

SOURCE_DIRECT = "direct"         # semantics passed cos_min_recall
SOURCE_ASSOCIATED = "associated" # association wave over L2 edges
SOURCE_KEYWORD = "keyword"       # exact token below the cosine floor

class DecisionAction(Enum): CREATE | REINFORCE | LINK

@dataclass(frozen=True) WriteDecision:
    action: DecisionAction
    target_id: int | None        # existing trace for REINFORCE/LINK
    related_ids: tuple[int, ...] # what to link a new trace to
    novelty: float               # 1 − best_cosine
    best_cosine: float

@dataclass(frozen=True) WriteResult:
    memory_id: int
    decision: WriteDecision
    created: bool                # True if a trace was created

@dataclass(frozen=True) RecalledMemory:
    memory_id: int; text: str; kind: str
    cosine: float                # [0..1], exact proximity to the query
    confidence: float            # composite (see Hippocampus.recall)
    retention: float             # [0..1]
    source: str                  # "direct" | "associated" | "keyword"
    created_at: float; updated_at: float; meta: dict
    scope: str                   # trace scope

@dataclass(frozen=True) RecallPacket:
    query: str; items: tuple[RecalledMemory, ...]
    abstained: bool; latency_ms: float

@dataclass(frozen=True) ConsolidationReport:
    edges_committed: int; edges_pruned: int
    promoted_to_semantic: int; rewards_applied: int; elapsed_ms: float

@dataclass MemoryRecord:  # internal storage unit
    id, text, kind, status, meta,
    embedding, sdr,
    created_at, updated_at, reinforced_count, last_reinforced_at,
    base_strength, valid_from, valid_to, superseded_by,
    scope = SCOPE_GLOBAL
```

Invariants: `0 ≤ cosine, confidence, retention ≤ 1`; abstained means "no
trustworthy memories" (empty after filters or the flat-noise rule fired — see
recall); a superseded trace has `superseded_by != None` and `valid_to != None`.

## config.py — `MemoryConfig`

A single dataclass of all hyperparameters (see field docstrings). Key groups:
encoding (`dim, n_units, k_sparse, sdr_seed`), L1 (`bucket_cap,
recall_oversample`), gate (`theta_reinforce ≥ theta_link > cos_min_recall`),
abstention (`cos_min_strong_recall ≥ cos_min_recall`, `abstain_spread_cos`),
decay (`tau_episodic < tau_semantic`, `min_retention_recall`,
`initial_strength ≤ strength_cap`), links (`tau_eligibility < tau_edge_stable`,
`edge_min_weight`, `max_pairs_per_bind`), spread (`depth, alpha, top_m, eps`),
ranking (`w_votes, assoc_confidence_penalty, w_keyword`),
operations (`backups_keep`: 0 disables pre-sleep copies).

Factories: `MemoryConfig.dev()` — small sizes for tests/demos;
`MemoryConfig.production()` — phase-3 target scales.
`validate()` invariants: `theta_reinforce > theta_link > cos_min_recall`;
`k_sparse ≤ n_units`.

## timeprov.py

```python
class TimeProvider(Protocol): def now(self) -> float
class SystemClock(TimeProvider)          # time.time()
class FakeClock(TimeProvider)            # .advance(seconds) for tests
```

## encoding.embedder

```python
class EmbeddingProvider(Protocol):
    name: str                                      # stable identity (model+version);
                                                   # the database records it in db_meta and
                                                   # refuses to open with another one
    dim: int
    def embed(self, text: str) -> np.ndarray       # facts; L2-normalized or zero
    def embed_query(self, text: str) -> np.ndarray # optional; used by the facade for
                                                   # search queries when present
class HashingEmbedder(dim=256, seed=7)              # default: feature hashing of words+3-grams,
                                                    # blake2b, deterministic cross-platform;
                                                    # name = "hashing(dim=D,seed=S)"
class FastEmbedProvider(model_name=DEFAULT_MODEL, dim=None, cache_dir=None)
                                                    # local ONNX (extra [local]);
                                                    # default cache ~/.cache/realmemory/fastembed;
                                                    # e5 prefixes only for e5 models;
                                                    # name includes the fastembed version;
                                                    # ClassVar recommended_thresholds — a threshold
                                                    # profile per model anisotropy, applied by
                                                    # mcp_server.main() before opening the database
```

`embed` contract: empty/blank text → zero vector; identical text → bit-identical
vector; lexically close texts → high cosine similarity; disjoint vocabularies
→ |cos| < 0.15.

## encoding.sdr

```python
class BipolarProjector(dim, n_bits, seed):          # utility, unused by the core
    def project(vec) -> np.ndarray[int8]            # sign(W·v̂)
class SDREncoder(dim, n_units, k, seed):
    def encode(vec) -> np.ndarray[int32]            # top-k by W·v̂, sorted;
                                                    # zero input → empty array
def overlap(a, b) -> int
def overlap_fraction(a, b) -> float                 # |A∩B| / min(|A|,|B|)
def calibrate_sparse(encoder, n_samples=120, seed=99) -> CalibrationStats(mu, sigma)
```

Monotonicity (test-checked): increasing cosine proximity of inputs →
non-decreasing SDR overlap in the statistical sense.

## core.addressing — SDRVotingIndex (L1)

```python
@dataclass(frozen=True) QueryResult:
    candidates: np.ndarray[int64]   # pointers, votes descending
    votes: np.ndarray[int32]        # votes, parallel to candidates
    active_locations: int           # query units with non-empty buckets

class SDRVotingIndex(n_units, bucket_cap):
    def write(sdr, pointer: int) -> int             # pointer into on-bit buckets;
                                                    # no duplicates; cap → FIFO eviction;
                                                    # returns #touched buckets
    def query(sdr, max_candidates) -> QueryResult   # votes = #shared units
    def load_factor() -> float                      # mean bucket length
    def state_dict() / load_state_dict(state)
```

Selectivity (test-checked): identical pattern → k votes; correlated with share ρ
of shared units → ~ρ·k; random → ~k²/N.
Exact-query contract: `write(A, p)` then `query(A)` always returns `p` first
absent eviction.

## core.plasticity

```python
@dataclass EligibilityEvent: src_units, dst_units: np.ndarray[int32]
                            strength: float; created_at: float; source_ids: frozenset[int]

class EligibilityLog(tau):
    def add(src_units, dst_units, strength, now, source_ids) -> None
    def reward(source_ids: Iterable[int], reward: float, now) -> int  # strength *= max(0,(1+r)); returns #events
    def commit(now) -> (src, dst, weights)   # np arrays merged by key (sum),
                                             # each contribution decayed exp(-(now-created)/tau);
                                             # contributions < 1e-12 dropped; log cleared
    def pending_count -> int
    def state_dict() / load_state_dict(state)

def merge_pairs(src, dst, w) -> (src2, dst2, w2)     # duplicate-pair aggregation by sum
```

Reward contract: positive reward only strengthens; negative may zero out a
contribution (not below 0).

## core.assembly — AssemblyNetwork (L2)

```python
class AssemblyNetwork(n_units, edge_min_weight, tau_edge_stable, seed, max_pairs_per_bind):
    def bind(units_a, units_b, strength, now) -> int       # #NEW directed edges
                                                           # (merging duplicates does not count);
                                                           # writes advance the network clock —
                                                           # existing edges decay continuously
    def commit_eligibility(src, dst, w, now) -> None       # decay_tick + accumulation
    def decay_tick(now) -> int                             # exponential decay of all edges, pruning; #left
    def spread(query_units, depth, alpha, top_m, eps) -> (units, scores)  # scores descending;
                                                                          # query units enter at 1.0,
                                                                          # sums may exceed 1
    def neighbors(unit) -> Iterator[(unit, weight)]
    def edge_count / total_weight -> float
    def state_dict() / load_state_dict(state)
```

Contract: bind is symmetric (both directions); after `decay_tick(t2)` edge
weight w is multiplied by exactly `exp(-(t2-t1)/tau)` relative to the previous
tick; spread returns no units with score < eps.

Note: within Hippocampus the persistent truth for L2 edges is the SQLite
`edges` table; AssemblyNetwork instances act as pure in-memory CSR caches.

## policies.novelty

```python
def gate(best_cosine: float, cfg: MemoryConfig) -> DecisionAction
# ≥ theta_reinforce → REINFORCE; ≥ theta_link → LINK; else CREATE
```

## policies.decay

Pure functions over write primitives:

```python
def retention(base_strength, last_reinforced_at, now, kind, cfg) -> float
    # base * exp(-(now-last)/tau(kind)), clipped to [0,1]; tau_semantic > tau_episodic
def reinforce_values(base, count, cfg) -> (base', count')
    # base = min(base*bump, cap); count += 1
def weaken_value(base, reward) -> base'
    # reward ∈ [-1, 0): base' = max(0, base*(1+reward)); timers untouched
def should_promote(kind, count, created_at, now, cfg) -> bool
    # episode → semantic: count >= promote_after_reinforcements and age >= promote_min_age_s
```

## store.sqlite_store — MemoryStore

```python
class MemoryStore(path, dim):                        # context manager; WAL + busy_timeout;
                                                     # mutations in BEGIN IMMEDIATE — safe
                                                     # for multiple concurrent processes
    # traces
    def insert(rec: MemoryRecord) -> int                     # assigns id
    def get(id) -> MemoryRecord | None
    def get_many(ids) -> list[MemoryRecord]                  # order as given, missing skipped
    def update_trace(id, base_strength, reinforced_count, last_reinforced_at, kind=None) -> None
    def adjust_base(id, base_strength, updated_at) -> None   # weakening without timer reset
    def mark_superseded(id, by_id, when) -> None
    def iter_active(batch=256) -> Iterator[MemoryRecord]
    def count(status=None, kind=None, scope=None) -> int
    def scope_counts() -> dict[str, int]                     # active traces per scope
    def all_active_ids() -> np.ndarray[int64]
    def top_by_reinforcements(limit=10) -> list[MemoryRecord]   # for reports
    def stale_episodic(limit=10) -> list[MemoryRecord]          # forgetting candidates
    def max_updated_at() -> float | None                         # sleep throttling
    def backup(dest_dir=None, keep=10) -> Path                   # consistent copy
                                                                 # (sqlite backup API) + rotation;
                                                                 # called on consolidate()
    # meta and counters (db_meta)
    def get_meta(key) -> str | None / set_meta(key, value)
    def bump_meta_int(key, delta=1) -> int                   # atomic increment, returns new value
    def consume_meta_int(key) -> int                         # read and reset to zero
    # L2 edges (table edges: key=src*n_units+dst)
    def edges_rev() -> int                                   # version for CSR-cache invalidation
    def edges_load() -> (keys int64, ws float32)             # sorted by key
    def edges_apply(src, dst, w, now, tau, min_weight, stride) -> (committed, pruned)
                                                             # decay from last_edge_tick + prune +
                                                             # batch merge + tick, single transaction
    def edges_import(keys, ws, last_tick)                    # legacy snapshot migration, no decay
    def edges_stats(now, tau) -> (count, effective_total_weight)
    # eligibility write-through (tables eligibility/elig_sources)
    def elig_add(src, dst, strength, created_at, source_ids) -> None
    def elig_reward(mem_ids, factor) -> int                  # strength *= factor via sources
    def elig_drain() -> list[event]                          # drain with deletion,
                                                             # matches EligibilityLog.load_state_dict
    def elig_pending() -> int
    # plasticity journal (table events)
    def event_append(event_type, fields, ts) -> None
    def event_count() -> int
    def iter_events() -> Iterator[dict]                      # {"ts","type",**data} ordered by seq
    def gate_decisions() -> dict[str, int]                   # from write events via $.action
    def recall_stats() -> (count, avg_latency_ms)
class StorageError(Exception)
```

Serialization: embedding→float32 blob, sdr→int32 blob.
`iter_active` skips superseded. Databases predating columns migrate via ALTER
at open (scope → default 'global'); every schema change is preceded by an
automatic backup. `db_meta['schema_version']` records the schema version.
`db_meta['embedder']` — the provider name that created the vectors;
`db_meta['config']` — geometry (dim/n_units/sdr_seed); Hippocampus rejects
incompatible opens (RuntimeError).

## hippocampus — Hippocampus (facade)

```python
class Hippocampus:
    @classmethod
    def open(cls, path: str, *, config: MemoryConfig | None = None,
             embedder: EmbeddingProvider | None = None,
             clock: TimeProvider | None = None,
             namespace: str | None = None,
             verify_embedder: bool = True) -> "Hippocampus"
        # opens or creates the database in directory path (path/namespace when namespace
        # given, [A-Za-z0-9][A-Za-z0-9_.-]{0,63}, else ValueError);
        # rebuilds derived structures (L1 buckets, unit index, edge CSR cache);
        # checks embedder identity and config geometry;
        # imports v0.3 legacy (journal.jsonl, snapshot.pkl) once;
        # verify_embedder=False — for tools without embeddings (hooks),
        # the check is skipped and the marker is not overwritten

    def remember(self, text, *, kind=KIND_EPISODIC, meta=None, when=None,
                 force_new=False, related_ids: Sequence[int] = (),
                 scope: str = SCOPE_GLOBAL) -> WriteResult
        # the gate compares text only within its scope + global;
        # ValueError on blank text, unknown kind, invalid scope;
        # KeyError when a related_ids id does not exist

    def recall(self, query, *, k=5, include_superseded=False,
               scope: str | None = None, all_scopes: bool = False) -> RecallPacket
        # the query is encoded with embed_query() when the embedder provides it;
        # scope='<project>' — traces of the project + global; None/all_scopes — everything;
        # L1 candidate budget is adaptive: floor = traces/64 (cap 1500), because
        # vote noise grows with corpus size;
        # FTS5 keyword channel: full token match boosts confidence,
        # partial match below the cosine floor comes back as source='keyword';
        # abstention additionally fires on "flat noise" (opt-in via the embedder
        # profile, cos_min_strong_recall > 0): top-1 direct below the floor,
        # cosine spread of the direct wave < abstain_spread_cos and no full
        # keyword match

    def link_memories(self, ids: Sequence[int], strength=1.0) -> int
        # pairwise bind between trace SDRs; KeyError on unknown id

    def feedback(self, ids: Sequence[int], reward: float) -> int
        # reward > 0 — reinforcement (bump, timer reset); reward < 0 — weakening
        # base*(1+reward) without touching the timer or promotion counter;
        # reward == 0 — neutral; also amplifies/weakens fresh eligibility events;
        # returns number of affected traces

    def update_fact(self, old_id: int, new_text: str, meta=None) -> WriteResult
        # force_new + mark_superseded(old→new); replacement inherits old scope

    def consolidate() -> ConsolidationReport     # all state already in the DB, nothing to save;
                                                 # parallel sleeps serialize via transaction
    def stats() -> dict                          # global, computed from the DB: identical
                                                 # for every process at any moment
    def metrics_snapshot(now=None) -> dict       # slice for dream_log/reports
    def pending_eligibility -> int               # uncommitted binds (shared across processes)
```

Recall confidence formula (fixed):
`confidence = cosine · (w_votes + (1−w_votes)·votes_norm) · (0.3 + 0.7·retention)`;
a full keyword match raises it up to `w_keyword · (0.3 + 0.7·retention)`;
associated items additionally ×`assoc_confidence_penalty` with cosine floored
at 0.05 (the link itself is a relevance signal, so no cosine filter applies to
the association wave — only retention).
Abstention: (a) no candidate passed the direct-wave filters
`cos ≥ cos_min_recall` and `retention ≥ min_retention_recall`; (b) the flat
-noise rule from the recall docstring.

## api.mcp_server

```python
def build_server(hippo, default_project=None):  # ImportError("pip install 'realmemory[mcp]'") without fastmcp
def make_embedder(choice):              # "local" | "hashing" | "auto"
def main(argv=None):                    # CLI: --path, --namespace, --project, --embedder; stdio MCP
                                        # default_project = resolve_project(--project):
                                        # explicit arg > REALMEMORY_PROJECT >
                                        # ZCODE_PROJECT_DIR/CLAUDE_PROJECT_DIR > cwd with .git/.zcode
```

Tools (names are cognitive actions, descriptions in English):
`recall(query, k, include_superseded, project)` ·
`memorize(text, kind, related_ids, project)` ·
`reflect(memory_ids, reward)` → `{touched}` · `revise(old_id, new_text)` ·
`introspect()` (includes current project and scope breakdown) · `dream_log()` —
thin wrappers over the facade with JSON serialization.
Scope policy in tool descriptions: facts about the current workspace go with
project=<name>; user-global preferences/identity omit project → global.

## hook_cli (agent automation)

```python
# python -m realmemory.hook_cli <cmd> --path P [--namespace NS]
brief  [--project P] [--top N] [--episodic-top M] [--plain]
                                # SessionStart: strict JSON
                                # {"hookSpecificOutput":{"hookEventName":
                                #  "SessionStart","additionalContext":str}};
                                # semantic facts (•) then durable episodic ones (·),
                                # episode score = retention·(1+ln(1+reinforcements));
                                # project+global scope filter, ~600 char budget;
                                # config read from db_meta, model NOT loaded
sleep  [--verbose]
                                # Stop: consolidate(); throttled by database state —
                                # skip when no writes since last_consolidate_at
                                # and eligibility is empty; empty output, always exit code 0
```

Failures print to stderr and append a `hook_error` event; the exit code stays 0
so a broken hook never breaks the agent session.

## report (python -m realmemory.report)

```python
def build_report(path, namespace=None) -> dict    # traces by type/scope/status, top_reinforced,
                                                  # stale_episodic, L1 index, gate decisions,
                                                  # recall/feedback/consolidation statistics,
                                                  # hook failures, metrics time series
def render(report) -> str          # human-readable text
def main(argv=None)                # CLI: --path, --namespace, --json out.json
```

Facade observability: every recall appends `{latency_ms, items, abstained,
top_conf}`; every consolidation appends a full `metrics` snapshot
(`Hippocampus.metrics_snapshot()`).

## eval.bench_recall / eval.bench_real

```python
# bench_recall: synthetic, hashing embedder
def run(n_facts=1500, n_queries=200, seed=0, config=None) -> dict  # metrics (see output)

# bench_real: fixtures/bench_real.json, fastembed required
# python -m realmemory.eval.bench_real [--k 10] [--verbose] [--json out.json]
```

bench_recall metric contract: `pipeline_hits@k`, `baseline_hits@k` (exact
cosine with the same embedder), p50/p95 latencies, writes/sec, noise abstention.
bench_real metrics: hits@10/MRR separately for paraphrases and exact tokens,
noise abstention, the write gate split into base vs duplicate paraphrase facts,
and calibration distributions (null "fact—nearest neighbor" without duplicates,
duplicate signal, per-type query max-cos). Fixture references use string fact ids.
