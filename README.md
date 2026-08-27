# realMemory

[![ci](https://github.com/re-obscura/realMemory/actions/workflows/ci.yml/badge.svg)](https://github.com/re-obscura/realMemory/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/re-obscura/realMemory/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A persistent memory layer for LLM agents with continuous learning: a local
"hippocampal" memory that writes without re-indexing, forgets via trace
dynamics, and consolidates episodes into semantics during "sleep".

**Status: v0.5 — retrieval defaults to an exact cosine scan over an in-process
embedding cache (complete recall, no candidate-generation ceiling), one SQLite
store shared by all processes, global/project memory scopes, hybrid FTS5
search, thresholds calibrated on real text.**

## The idea in a nutshell

The LLM stays frozen ("cortex"). realMemory is a separate mutable module
("hippocampus"):

- **Novelty-gated writes**: a known fact gets potentiated, a related one is
  linked, a fresh one allocates a new trace. Reformulations never pile up.
- **Shared + per-project memory**: every trace carries a scope (`global` or a
  project name); recall sees the current project plus `global` and never mixes
  contexts.
- **Forgetting from trace dynamics**: each trace's retention decays
  exponentially, reinforcements extend its life, sufficiently reinforced
  episodes promote to semantic traces (slow decay). The forgetting curve is a
  property of the synapse, not a cron job.
- **An associative graph for free**: whatever was recalled together gets bound
  by plasticity (an STDP-like rule) — multi-hop traversal emerges from usage
  statistics, not from LLM entity extraction.
- **Hybrid retrieval**: exact-token search (FTS5) complements embeddings —
  error IDs, package names and codes are found even when cosine similarity is low.
- **Sleep**: offline consolidation commits eligibility traces, decays/prunes weak
  links and promotes statuses. All state lives in one SQLite database: an MCP
  server and hooks run concurrently without losing data.

## Quick start

```bash
pip install -e ".[dev]"
pytest                # full core test suite
python -m realmemory.eval.bench_recall --facts 1500 --queries 200   # synthetic
python -m realmemory.eval.bench_real                                # real-text (fastembed)
```

```python
from realmemory import Hippocampus, MemoryConfig

hippo = Hippocampus.open("./rm_data", config=MemoryConfig.dev())
hippo.remember("The project uses PostgreSQL 16 with alembic migrations",
               scope="myproject")            # a project-scoped fact
hippo.remember("The user prefers concise answers") # global by default

packet = hippo.recall("which database does the project use?", scope="myproject")
for item in packet.items:
    print(f"[{item.confidence:.2f}] ({item.source}) {item.text}")
if packet.abstained:
    print("no trustworthy memories")     # abstention instead of hallucination

hippo.consolidate()   # "sleep": commit traces, decay weak links
```

## Local embedder

By default the core uses a deterministic `HashingEmbedder` (no models).
The production local semantic embedder is fastembed (ONNX Runtime, CPU):

```bash
pip install 'realmemory[local]'
```

- Model: `paraphrase-multilingual-MiniLM-L12-v2`, **dim=384**, Russian+English.
- Model cache: `~/.cache/realmemory/fastembed` (~240 MB), downloaded once.
- Measured load: ~580 MB process RAM; ~65–75 ms per text on CPU;
  a full recall ≈ 77 ms. Invisible to the agent.
- Asymmetry is handled: facts are encoded with `embed()`, queries with `embed_query()`.
- Gate thresholds are calibrated per model anisotropy: the threshold profile
  lives in `FastEmbedProvider.recommended_thresholds`, applied at server start,
  derived from the real-text benchmark (see below).

## Wiring into ZCode / Claude Code (MCP)

Register a user-scope stdio server in your client config:

```json
"realmemory": {
  "type": "stdio",
  "command": "/path/to/venv/Scripts/python.exe",
  "args": ["-m", "realmemory.api.mcp_server",
           "--path", "/path/to/rm_data",
           "--embedder", "local"]
}
```

Agent tools (named as cognitive actions): `recall(query,k,project)` ·
`memorize(text,kind,related_ids,project)` · `reflect(memory_ids,reward)` ·
`revise(old_id,new_text)` · `introspect()` · `dream_log()`.

**Shared + per-project memory**: every trace is tagged with a scope — `global`
(preferences, identity) or a project name. The project is detected
automatically (`REALMEMORY_PROJECT` → `ZCODE_PROJECT_DIR` → current directory
containing `.git`); it can also be passed explicitly via the `project`
argument or `--project`. `recall` searches the current project + global;
other projects never leak in.

Full namespace isolation between separate brains is available via
`Hippocampus.open(path, namespace=...)` / `--namespace`.

The database stores an embedder marker (`db_meta`) and refuses to open with a
different one — old and new vectors are not comparable by cosine.

## Automation: making agents actually use it

Three mechanisms, installed by default:

1. **Skill / instructions** describing when to recall / memorize / reflect,
   loaded into every session context.
2. **SessionStart hook** → `python -m realmemory.hook_cli brief` — injects a
   short memory state: semantic facts and durable episodic traces of the
   current project + global, ~600 character budget.
3. **Stop hook** → `python -m realmemory.hook_cli sleep` — consolidation after
   each answer; throttled by database state (skips when nothing changed since
   the last sleep). Takes ~0.3 s, does not load the embedder model.

Hooks and the MCP server safely run at the same time: all state is in SQLite,
concurrent "sleeps" are serialized by a transaction.

## Operations

- **Backups**: before every "sleep" the database is copied to
  `<store>/backups/` (consistent sqlite backup API), last 10 copies kept
  (`backups_keep`; 0 disables). Any schema migration takes an automatic
  safety copy first.
- **Schema version** recorded in `db_meta.schema_version`.
- **Hook failures are not silent**: a failing hook prints to the session's
  stderr and leaves a `hook_error` event in the journal, visible in the report.
- **Learning discipline**: the report shows reflect/recall — below ~0.1 the
  agent rarely grades recalled memories and decay/promotion run blind.
- **Forgetting becomes literal**: traces whose retention fell below the
  recall floor and stayed unreinforced longer than `gc_grace_below_floor_s`
  are deleted at consolidation (rows, FTS index, eligibility links, caches);
  negative feedback therefore shrinks the base instead of hoarding zombies.
  Superseded history is kept by design.
- **Project routing** is verified with one call — `introspect` shows the
  currently detected project.

## Team sharing (preview)

Personal memory stays fully local by default. On top of it, an explicit
sharing layer is growing: every publication is a deliberate act recorded in a
local registry (with tombstones for retractions), and `~/.realmemory/team.yaml`
declares *what may* leave the machine; never-rules work fail-closed even
against explicit requests without `--force`.

```bash
pip install 'realmemory[team]'
python -m realmemory.team status --path ./rm_data   # сводка по проектам
python -m realmemory.team ui     --path ./rm_data   # интерактивный выбор (Textual)
python -m realmemory.team policy                     # показать политику/путь
```

A close fact recorded under a DIFFERENT author never reinforces that
trace — it links instead, keeping both viewpoints attributable. Cross-process
writes propagate through a `memories_rev` revision counter (volatile caches
resync lazily at the next recall/remember). Network transport is the next
stage; see [`docs/TEAM.md`](docs/TEAM.md).

## Observability ("how the memory behaves over time")

Every event is appended to the journal inside the database: writes, recalls
(latency, abstention, confidence), feedback, consolidations with full metrics.
Full report any time:

```bash
python -m realmemory.report --path ./rm_data [--json report.json]
```

Shows: memory growth by type/scope/status, novelty-gate decision history,
abstention share and p50/p95 recall latency, what got reinforced, which
episodes fade, retention dynamics across sleeps, hook failures.

## Phase 0 results (real runs)

Real-text benchmark (`bench_real`, fastembed MiniLM dim=384, 103 RU/EN facts,
89 queries — paraphrases, exact tokens, noise):

| Metric | before calibration | after calibration | v0.5 exact engine |
|---|---|---|---|
| paraphrase hits@10 / MRR | 0.741 / 0.611 | 0.870 / 0.698 | **0.889 / 0.709** |
| exact-token hits@10 / MRR | 0.667 / 0.633 | **1.000 / 0.956** | 1.000 / 0.956 |
| abstention on noise | 0.00 | 0.30 | **0.55** |
| false merges by the write gate | 85 of 89 facts | **0** (88 create) | 0 |
| duplicate paraphrases recognized | partial | 14 / 14 | 14 / 14 |

Against the naive full-scan cosine baseline the pipeline now wins on
paraphrase ranking quality (MRR 0.709 vs 0.677) and decisively on exact
tokens (1.000 vs 0.800); pure-threshold abstention remains stronger (0.85
vs 0.55) — see [`docs/ARCHITECTURE.md` §7.2](docs/ARCHITECTURE.md).

Scale sweep (`bench_recall`, hashing embedder dim=2048, 200 subset queries;
the hits metric counts only facts that own their own trace — write-gate
merges are reported separately):

| Corpus | pipeline hits@10 | baseline | gate merges | abstention | recall p50/p95, ms | writes/sec |
|---|---|---|---|---|---|---|
| 10 000 | **1.000** | 1.000 | 3.7% | 1.00 | 23 / 64 | 105 |
| 30 000 | **1.000** | 1.000 | 8.9% | 1.00 | 81 / 185 | 74 |
| 50 000 | **1.000** | 1.000 | 13.2% | 1.00 | 98 / 276 | 58 |

The recall-quality cliff between 10k and 30k that motivated v0.5 no longer
exists: the exact-scan engine matches the all-traces baseline everywhere.
`gate merges` is a corpus property of the novelty gate (lexically close
object-token twins merge above θ_reinforce by birthday-paradox growth),
not a retrieval loss; latencies are for the artificial dim=2048 setup —
at production dim=384 the same work costs roughly an order less.
Details and the negative Hamming-SDM result in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §3 and §7.

Lesson kept from the synthetic benchmark history: it scored 1.000 while
default thresholds on real text merged almost everything into blobs — the
calibration still lives in per-embedder profiles, and the gate-merge share
is published rather than hidden inside the hit rate.

Tests: **157 passed**.

## Architecture

Retrieval by default is an **exact cosine scan over an in-process embedding
cache** (numpy gemv over active traces; early termination is provably lossless
for the ranking because direct confidence ≤ cosine). Past
`exact_scan_max_traces` the engine falls back to **L1** — `SDRVotingIndex`,
pointer voting over an inverted index of SDR units — trading completeness for
memory footprint. **L2**, an assembly network over the same units
(associations, completion, multi-hop) plus a keyword channel (FTS5), the
novelty gate, decay policies and the offline consolidator ("sleep") complete
the stack.

Module interfaces are fixed in [`docs/CONTRACTS.md`](docs/CONTRACTS.md);
research background and sources in [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Project layout

```
src/realmemory/
├── encoding/     # embedders, SDR encoding
├── core/         # L1 SDRVotingIndex, L2 AssemblyNetwork, plasticity
├── policies/     # novelty gate, trace decay/promotion
├── store/        # SQLite storage (traces, edges, eligibility, events)
├── api/          # MCP server
└── eval/         # benchmarks
```

## License

[MIT](LICENSE)
