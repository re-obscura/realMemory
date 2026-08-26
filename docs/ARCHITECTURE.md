# realMemory Architecture

Status: v0.4 (single SQLite store shared by all processes, global/project
scopes, hybrid FTS5 search, thresholds calibrated on real text, 122 tests,
phase-0 gate PASS).
This document records design decisions and their rationale; changes go through
an edit of this file in the same commit.

## 1. Positioning

A local single-user persistent memory service for LLM agents. The LLM stays
frozen; realMemory is a separate mutable module living across sessions and
projects, learning continuously via local plasticity rules, without gradient
descent.

Non-goals for v1: multi-tenant SaaS, RAG over large corpora, fine-tuning LLM weights.

## 2. Biology → system map

| Biology | Component |
|---|---|
| Entorhinal cortex | Frozen text embedder (`EmbeddingProvider`) |
| Dentate gyrus (pattern separation) | SDR: k on-bits out of N units (`SDREncoder`) |
| CA3 (attractor, completion) | Assembly network L2 with plastic links + spread/cleanup |
| CA1 (novelty detection) | Write gate by cosine proximity to known traces |
| Hippocampus (fast trace) | Trace pointers in L1 unit buckets (`SDRVotingIndex`) |
| Cortex (slow memory) | Semantic traces: promoted status → slow decay |
| Sharp-wave replay ("sleep") | Offline consolidator: trace commit, decay, promotion |
| Neuromodulation (dopamine) | Third factor: `feedback()` amplifies fresh eligibility events |

**What the metaphor is and is not.** The *dynamics* are borrowed from
computational neuroscience — novelty-gated writes, eligibility traces with a
third factor, replay-style consolidation (sources in
[RESEARCH.md §4](RESEARCH.md)). There are no capacity or energy claims: an
SNN-style store does **not** beat a vector database on either, and explicit
kill criteria demote the plastic layer if it ever loses to plain retrieval
heuristics ([RESEARCH.md §5](RESEARCH.md)).

## 3. Core levels

### L1 — SDRVotingIndex (voting addressing)

**Purpose:** recall candidates without scanning the whole base; O(k) writes
without re-indexing.

**Mechanics:** a trace writes a pointer (its SQLite id) into the buckets of all
its k on-bits; a query sums hits over its own on-bits and returns top-k
pointers by votes. Correlated patterns share some units, so the target pointer
collects ~ρ·k votes versus ~k²/N for random ones — robust-LSH behavior.
Neuromorphic reading: units are neurons, bucket writes are axonal pointers,
votes are presynaptic coincidence summation, top-k is WTA.
Bucket overflow eviction is FIFO (palimpsest under pressure); eviction horizon
≈ `bucket_cap / (corpus·k/N)` traces.

**Negative result (recorded honestly):** the first implementation was classic
Kanerva SDM over dense bipolar addresses with a Hamming access radius. It
failed the benchmark on semantically close queries: hits@10 = 0.495 vs 1.00
for exact cosine (1500 facts). The cause is Hamming-metric concentration:
distances of random pairs are anticorrelated, so the "access spheres" of
patterns with cos≈0.6 barely intersect; a radius catching near-duplicates
catches nothing moderately similar. Voting over shared units of a sparse code
has no such defect — it measures proximity directly. Lesson: in concentrated
metrics, threshold balls are a poor carrier of "closeness"; intersection of
sparse representations is a good one.

### L2 — AssemblyNetwork (assembly network)

**Space:** the same N_units; traces share units superpositionally.

**Links:** directed edges key=(i·N+j) → float32 weight (dict + lazy CSR).
Accumulation rule — BCPNN-lite: normalized co-occurrence (the full Bayesian
form is phase 3); palimpsest comes from decay and pruning.

**Operations:**
- `bind(A, B, strength)` — links on-bits of two patterns both ways; network
  clock advances on any write: existing edges decay continuously
  (materialized lazily from `last_edge_tick`), weak edges get pruned at
  consolidation;
- `spread(units, depth)` — activation propagation over edges with attenuation:
  associative multi-hop traversal of the "what was recalled together" graph;
- `commit_eligibility(...)` — merges buffered links during sleep.

**Eligibility (third factor):** binds accumulate in an eligibility table with
timestamps and source ids; each event's contribution decays exp(−Δt/τ_e) by
commit time; `feedback()` multiplies the strength of not-yet-committed events
(three-factor learning, Frémaux & Gerstner 2016).

## 4. Data paths

### Write

```
text → embedder → emb → SDREncoder → sdr
SDRVotingIndex.query(sdr, oversample) + FTS5 token candidates
   → candidates → exact cosine rerank over embeddings → best_cosine
Novelty gate (probe sees only this scope + global):
   best_cos ≥ θ_reinforce → REINFORCE (bump base_strength, counter, timer;
                             enough reinforcements+age ⇒ semantic status,
                             decay τ ×12 — mechanical semantization)
   best_cos ≥ θ_link      → CREATE + plastic links to active traces
   otherwise              → CREATE (+ links to related_ids, if passed)
```

Thresholds are per-embedder profiles (anisotropy differs between models);
for MiniLM they were derived from bench_real distributions (§7.2).

### Read

```
query → sdr → L1 candidates (oversample×k) ┐
FTS5: query tokens → bm25 candidates       ├─ candidate union
→ fetch → exact cosine rerank ←────────────┘
per-trace channel:
   full keyword-match (all query tokens present in the trace text)
      → confidence boosted up to w_keyword·(0.3+0.7·retention);
        source stays "direct" if cosine itself passed cos_min_recall,
        otherwise "keyword"
   cos ≥ cos_min_recall → direct: conf = cos·(w_v+(1−w_v)·votes_norm)·(0.3+0.7·ret)
   partial FTS-match below the cosine floor → keyword:
      conf = w_keyword·overlap·(0.3+0.7·retention)
   else dropped; everywhere retention ≥ min_retention_recall, scope filter
abstention "flat noise": top-1 direct < cos_min_strong_recall,
   direct-wave cosine spread < abstain_spread_cos and no full keyword match
   ⇒ empty packet with abstained=true (embedder anisotropy noise is
   indistinguishable from "no answer" — abstaining is more honest)
→ association wave (if direct hits < k): spread from SDRs of top traces over
  L2 edges → units → inverted unit→traces index → extra candidates;
  NO cosine filter here (the link itself is a relevance signal), only
  retention applies; confidence × assoc_confidence_penalty, cosine floored
  at 0.05 so confidence does not vanish
→ RecallPacket(items, abstained, latency)
```

Abstention: no candidate passed the filters, or the flat-noise rule fired ⇒
`abstained=true` — the system says "no trustworthy memories" instead of
hallucinating. Rule parameters were derived from bench_real: noise queries
have top-1 p50=0.29 with flat spreads, paraphrase targets have top-1 p10=0.40
with a clear leader.

Scopes: direct and association waves see only traces of the current project +
`global` (scope=None/all_scopes removes the filter). The write gate probes in
the same narrowed view — identical facts from different projects never merge
into REINFORCE.

### Update / contradictions

`update_fact(old_id, new_text)` creates a new trace force_new; the old one gets
`status='superseded'`, `valid_to`, a `superseded_by` link (bi-temporal), and the
old↔new association is kept. Superseded traces are excluded from recall by
default, available via `include_superseded=True`; the replacement inherits the
old trace's scope. Automatic contradiction detection (co-activation of
conflicting assemblies + LLM-judge during sleep) — phase 2.

### Sleep (consolidate)

1. drain the eligibility table → stable L2 edges (with reward amplification);
2. decay stable edges over the elapsed time since `last_edge_tick`, prune weak
   ones (lazy decay: effective weight is computed from the tick label and
   materialized only inside the consolidation transaction — O(E) once per
   sleep, not per write);
3. scan episodes: promotion to semantic by the promotion rule;
4. consolidation events plus a full metrics snapshot into the journal.

Note: links between a write and the first "sleep" exist only in the eligibility
table and do not participate in the recall association wave (they commit during
consolidation). A deliberate v0 compromise; hot propagation of uncommitted
links is a phase-1 candidate.

Parallel sleeps: the MCP server and the Stop hook may consolidate at the same
time; BEGIN IMMEDIATE serializes the transactions, the decay Δt is measured
from last_edge_tick after acquiring the lock — no double decay, no lost events.

## 5. Storage

All state lives in one SQLite database (`memory.db`, WAL + busy_timeout +
BEGIN IMMEDIATE) — the single source of truth for any number of processes:
a long-running MCP server and short-lived hooks read and write concurrently,
nobody clobbers anybody.

| Table | Contents |
|---|---|
| `memories` | texts, embeddings (float32 blob), SDRs (int32 blob), trace metrics, supersede chains, `scope` |
| `edges` | stable L2 edges: key = src·n_units+dst, weight |
| `eligibility` / `elig_sources` | uncommitted binds (write-through on write) and their source traces |
| `events` | plasticity journal: writes/recalls/feedback/consolidations/metrics |
| `memories_fts` | external-content FTS5 index over texts (unicode61), sync triggers |
| `db_meta` | embedder marker, config geometry, last_edge_tick, counters |

Recovery invariants: L1 buckets, the unit index and the edge CSR cache are
derived structures, deterministically rebuilt from the database at open;
everything else lives only in tables. There are no snapshot files (before
v0.4 there was a snapshot.pkl with a last-writer-wins race between processes);
v0.3 legacy is imported once, guarded by flags in db_meta.

Embedder identity: at first open the database records the provider `name`
(model + library version) in `db_meta`; opening with another name is rejected —
vectors from different models/versions are not cosine-comparable. The same
table stores geometry (dim/n_units/sdr_seed): incompatible opens are rejected
too. Pre-marking databases adopt the current embedder once (an
`embedder_identity_adopted` event).

Project isolation: `scope` on every trace (`global` or a project name) plus
`Hippocampus.open(path, namespace=...)` for hard file-level isolation. The
project is auto-detected via `projects.resolve_project()`: explicit argument →
`REALMEMORY_PROJECT` → `ZCODE_PROJECT_DIR`/`CLAUDE_PROJECT_DIR` (injected by
ZCode hooks automatically) → working directory containing a `.git`/`.zcode`
marker.

Backups: before every sleep the database is copied (sqlite backup API) into
`backups/` with rotation (`backups_keep`), and every schema migration takes an
automatic safety copy first. `db_meta['schema_version']` records the schema
version.

## 6. Integrations

- **MCP server** (`realmemory.api.mcp_server`): tools `recall`, `memorize`,
  `reflect`, `revise`, `introspect`, `dream_log` — named as cognitive actions;
  optional extra `[mcp]`; run
  `python -m realmemory.api.mcp_server --path ./rm_data [--namespace ns] [--project p]`.
- Direct Python API (`Hippocampus`) — the primary contract for tests and SDK use.

## 7. Results (real runs)

### 7.1 Synthetic (bench_recall: hashing embedder, dim=2048, disjoint vocabularies)

Queries are word subsets of facts; baseline is exact cosine with the same
embedder. Config: dim=2048, n_units=2048, k=96, bucket_cap=512, cos_min=0.18.

| Metric | 1500 facts | 5000 facts |
|---|---|---|
| pipeline hits@10 | **1.000** | 0.997 |
| baseline hits@10 (exact cos) | 1.000 | 1.000 |
| abstention on noise queries | **1.00** | 0.95 |
| recall latency p50 / p95, ms | 2.5 / 3.1 | 3.8 / 5.0 |
| writes/sec | 419 | 321 |

Phase-0 gate (pipeline ≥ baseline−0.02 and abstention ≥ 0.9): **PASS** at both scales.

### 7.3 Scale sweep (bench_recall, same synthetic setup)

Capacity parameters scaled with the corpus (n_units=16384 beyond 5k facts —
otherwise palimpsest eviction starts, which is forgetting-by-design, not a bug).

| Corpus | pipeline hits@10 | abstention | recall p50/p95, ms | writes/sec | reopen rebuild, s |
|---|---|---|---|---|---|
| 1 500 | 1.000 | 1.00 | 2.5 / 3.1 | 419 | — |
| 10 000 | 0.99 | 1.00 | 13.2 / 13.9 | 97 | 1.8 |
| 30 000 | 0.91 | 0.95 | 25.9 / 27.4 | 69 | 7.5 |
| 50 000 | 0.94 | 0.95 | 37.4 / 40.2 | 58 | 14.2 |

Current state includes two mitigations: the flat-noise heuristic is opt-in
(see below) and the L1 candidate budget grows with the corpus
(floor = traces/64, cap 1500 — vote noise scales with corpus size).

Honest findings:

- Latency, throughput and reopen-rebuild scale gracefully to 50k traces;
  rebuild is roughly linear in corpus size.
- A **recall-quality cliff appears between 10k and 30k** with this setup.
  Root cause was traced with instrumentation, in two layers:
  1. *Fixed*: the flat-noise abstention heuristic shipped with defaults tuned
     for anisotropic semantic embedders and misfired on the hashing embedder
     (dim=2048 compresses the whole cosine band), discarding correct answers
     wholesale (~4pp of misses at 30k, plus silently inflating abstention to
     1.00). It is now **opt-in via the embedder profile**
     (`cos_min_strong_recall > 0`); the fastembed profile keeps it on.
  2. *Open, phase 1*: vote-based candidate generation has an intrinsic
     coverage ceiling on weak-overlap subset queries. Measured at 30k:
     target-in-top-N votes = 0.89 @N=60, 0.93 @300, 0.96 @600, **0.98 @1200**,
     saturating — ~2% of targets share so few SDR units with the query that
     thousands of common-token competitors outvote them at any budget.
     The adaptive budget buys back part of the loss (+4pp at 30k) for
     ~+10–16 ms p50; full recovery needs phase-1 remedies: IDF-style
     unit-frequency weighting of votes, or an embedding cache with exact-scan
     merge (candidates = top-votes ∪ top-exact).
- Non-monotonicity between 30k and 50k runs is within sampling noise of 100
  queries plus eviction randomness.

A quantitative comparison against LLM-backed memory services (mem0/Zep/Letta)
is deliberately deferred until after dogfooding: they sit on a different axis
(API call per write, non-deterministic) and a fair harness needs both quality
and cost/latency columns.

### 7.2 Real text (bench_real: fastembed MiniLM dim=384, RU/EN)

103 facts from adjacent domains, 54 paraphrase queries, 15 exact-token queries,
20 noise queries; 14 duplicate-paraphrase facts probing the write gate.

| Metric | before calibration | after calibration |
|---|---|---|
| paraphrase hits@10 / MRR | 0.741 / 0.611 | **0.870 / 0.698** |
| exact tokens hits@10 / MRR | 0.667 / 0.633 | **1.000 / 0.956** |
| abstention on noise | 0.00 | 0.30 |
| false gate merges (base facts) | 85 of 89 | **0** (88 create) |
| duplicates recognized by the gate | partial | 14 / 14 |

Calibration distributions (the basis of the MiniLM threshold profile):
the null "fact—nearest neighbor" p50=0.478/p95=0.578/max=0.653 against the
duplicate signal min=0.633/p50=0.751 → theta_reinforce=0.70, theta_link=0.58
(false merges excluded constructively; a borderline pair degrades to a safe
LINK). Mean-centering did not help; multilingual-e5-large also failed to
separate this corpus (noise p95=0.798 vs target p50=0.856) at ~20× cost —
a negative result we record.

Tests: **122 passed** (unit contracts of all modules + end-to-end scenarios:
novelty gate, supersede, abstention, positive/negative feedback, forgetting,
associative multi-hop after sleep, save/reload identity, namespace and scope
isolation, embedder identity and geometry guards, multi-process state
integrity, v0.3 legacy migration, stdio-e2e MCP, operational guarantees).

## 8. Limitations of v0.4 (honestly)

- The default embedder is feature-hashing (lexical similarity, no semantics).
  The collision floor on unrelated texts is ~0.12 at dim=2048 —
  `cos_min_recall` is calibrated per embedder. A production embedder plugs in
  through the `EmbeddingProvider` contract without core changes.
- MiniLM anisotropy on homogeneous corpora caps absolute abstention: noise with
  top-1 cosine above `cos_min_strong_recall` (~30% of bench_real noise) passes.
  The flat-noise rule removes part of it; full elimination requires a
  contrastive retrieval embedder (multilingual-e5-large tested — does not
  separate this corpus better at ~20× cost).
- Associative links surface in recall only after the first consolidation; the
  edge CSR cache refreshes by `edges_rev` versioning — other processes'
  consolidations become visible at the next recall, within-sleep staleness is
  acceptable.
- Automatic contradiction detection — phase 2 (currently explicit `update_fact`).
- L1 performance is Python dict/deque: graceful to 50k traces on latency and
  rebuild (§7.3), but a recall-quality cliff appears between 10k and 30k on the
  synthetic setup (post-fetch stage under hashing-collision density; open
  phase-1 issue). Beyond 10⁵–10⁶ traces the hot path moves to numpy/CUDA
  (phase 3).
- Recall quality at scale: the write-gate misfire is fixed (opt-in abstention),
  but vote-based candidate generation caps at ~98% coverage on weak-overlap
  queries regardless of budget (§7.3); closing the last gap needs IDF-weighted
  voting or an embedding-cache exact merge (phase 1).
- No quantitative comparison against LLM-backed memory services yet
  (mem0/Zep/Letta): different trade-off axis — local, deterministic, zero-cost
  writes vs richer semantics through LLM extraction. A quality+cost harness is
  planned after the dogfooding period.
- FTS5 missing in exotic SQLite builds → the core works, the keyword channel
  disables itself (`fts_enabled=False`).
