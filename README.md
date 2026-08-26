# realMemory

[![ci](https://github.com/re-obscura/realMemory/actions/workflows/ci.yml/badge.svg)](https://github.com/re-obscura/realMemory/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://github.com/re-obscura/realMemory/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A persistent memory layer for LLM agents with continuous learning: a local
"hippocampal" memory that writes without re-indexing, forgets via trace
dynamics, and consolidates episodes into semantics during "sleep".

**Status: v0.4 — single SQLite store shared by all processes, global/project
memory scopes, hybrid FTS5 search, thresholds calibrated on real text.**

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
- **Project routing** is verified with one call — `introspect` shows the
  currently detected project.

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

Synthetic benchmark (`bench_recall`, hashing embedder, dim=2048):

| Metric | 1500 facts | 5000 facts |
|---|---|---|
| pipeline hits@10 | **1.000** | 0.997 |
| baseline hits@10 (exact cosine, same embedder) | 1.000 | 1.000 |
| abstention on noise queries | **1.00** | 0.95 |
| recall p50 / p95, ms | 2.5 / 3.1 | 3.8 / 5.0 |
| writes/sec | 419 | 321 |

Real-text benchmark (`bench_real`, fastembed MiniLM dim=384, 103 RU/EN facts,
89 queries — paraphrases, exact tokens, noise):

| Metric | before calibration | after calibration |
|---|---|---|
| paraphrase hits@10 / MRR | 0.741 / 0.611 | **0.870 / 0.698** |
| exact-token hits@10 / MRR | 0.667 / 0.633 | **1.000 / 0.956** |
| abstention on noise | 0.00 | 0.30 |
| false merges by the write gate | 85 of 89 facts | **0** (88 create) |
| duplicate paraphrases recognized | partial | 14 / 14 |

Lesson from the synthetic benchmark: it scored 1.000 while default thresholds
on real text merged almost everything into a few blobs — the calibration is now
derived from benchmark distributions and lives in the embedder profile.
The same real-text benchmark includes a naive full-scan cosine baseline:
the pipeline wins clearly on exact tokens (1.000 vs 0.800), is on par on
paraphrases, and currently abstains less aggressively than a pure threshold —
see [`docs/ARCHITECTURE.md` §7.2](docs/ARCHITECTURE.md).
Scale sweep (10k–50k traces) with honest findings about a recall-quality cliff
at 30k on synthetic data: [§7.3](docs/ARCHITECTURE.md).

Details and the negative Hamming-SDM result in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §3 and §7.

Tests: **122 passed**.

## Architecture

In short: **L1** — `SDRVotingIndex`, pointer voting over an inverted index of
SDR units (capacity + candidates), **L2** — an assembly network over the same
units (associations, completion, multi-hop), topped with an exact embedding
rerank, a novelty gate, decay policies and an offline consolidator ("sleep").

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
