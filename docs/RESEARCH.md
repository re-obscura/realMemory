# realMemory Research Foundations

A research summary (August 2026) the architecture is based on. Every claim
carries a source. Method: OpenAlex / Crossref / arXiv API / primary sources;
search engines were partially blocked, so everything was verified against
primary URLs.

## 1. The niche is unoccupied

There is no published system where a plastic SNN/associative network acts as a
persistent memory sidecar to an unmodified transformer. The whole SNN×LLM
direction is about replacing the LLM itself with a spiking version for energy:
[SpikeGPT](https://arxiv.org/abs/2302.13939),
[SpikingBERT](https://arxiv.org/abs/2308.10873),
[SpikeLLM](https://arxiv.org/abs/2407.04752),
[NSLLM, NSR 2025](https://doi.org/10.1093/nsr/nwaf551).
The nearest "proper" neighbors, without spikes:
[HeLa-Mem, ACL 2026](https://doi.org/10.18653/v1/2026.acl-long.625) (Hebbian
associative memory for agents) and
[SpikeHD](https://doi.org/10.1038/s41598-022-11073-3) (SNN + hyperdimensional memory).

## 2. Parametric memory in LLM weights does not scale

Model editing (ROME/MEMIT) works for single edits and degrades into two-phase
catastrophic forgetting at scale:
[Model Editing at Scale](https://arxiv.org/abs/2401.07453). The only things
that work continuously without forgetting are separate mutable structures next
to frozen weights: latent pools ([MemoryLLM](https://arxiv.org/abs/2402.04624)
— ~1M updates without degradation; [M+](https://arxiv.org/abs/2502.00592)),
fast weights ([Titans](https://arxiv.org/abs/2501.00663),
[TTT layers](https://arxiv.org/abs/2407.04620),
[MIRAS](https://arxiv.org/abs/2504.13173)), external graphs.

## 3. The 2024–2026 competitor landscape and unsolved axes

- [MemGPT/Letta](https://arxiv.org/abs/2310.08560),
  [sleep-time compute](https://arxiv.org/abs/2504.13171) — offline consolidation
  cuts test-time compute ~5×;
- [Mem0](https://arxiv.org/abs/2504.19413) — online extraction/consolidation;
- [Zep/Graphiti](https://arxiv.org/abs/2501.13956) — bi-temporal knowledge graph;
- [HippoRAG 2](https://arxiv.org/abs/2502.14802) — Personalized PageRank,
  "non-parametric continual learning";
- [A-MEM](https://arxiv.org/abs/2502.12110) — Zettel-style note evolution;
- [MemoryBank](https://arxiv.org/abs/2305.10250) — the only explicit forgetting
  curve (an Ebbinghaus approximation formula).

Axes everyone acknowledges as unsolved: contradiction resolution (OpenAI
publicly admitted contradictory memories in its help center), a principled
forgetting policy, temporal logic, consolidation timing, adversarial robustness
(questions about the unsaid). Fact-recall benchmarks are saturating
([LoCoMo](https://arxiv.org/abs/2402.17753) >94%); the live axes are knowledge
update and abstention from [LongMemEval](https://arxiv.org/abs/2410.10813).

Conclusion: four operations native to spiking dynamics (novelty-gated writes,
eligibility decay, associative completion, replay consolidation) are exactly
the center of the unsolved set. This is realMemory's positioning.

## 4. Scientific foundations of the chosen mechanisms

- **Two-store consolidation**: pattern separation (DG) → completion (CA3)
  → replay during sleep → semantization. [Rolls 2013](https://doi.org/10.3389/fnsys.2013.00074),
  [Buzsáki 2015](https://doi.org/10.1002/hipo.22488),
  [Rasch & Born 2013](https://doi.org/10.1152/physrev.00032.2012).
- **BCPNN and palimpsest**: incremental Bayesian-Hebbian rule with soft
  overwrite without catastrophe —
  [Sandberg et al. 2002](https://doi.org/10.1016/s0925-2312(00)00270-8);
  spiking version — [Tully et al. 2016](https://doi.org/10.1371/journal.pcbi.1004954);
  episode→semantic semantics under repeated exposures —
  [Chrysanthidis et al. 2022](https://doi.org/10.1523/ENEURO.0062-22.2022).
- **Third factor / eligibility traces**:
  [Frémaux & Gerstner 2016](https://doi.org/10.3389/fncir.2015.00085),
  [Gerstner et al. 2018](https://doi.org/10.3389/fncir.2018.00053),
  [e-prop](https://doi.org/10.1038/s41467-020-17236-y).
- **Associative memory on spikes** (theoretical basis of L1/L2):
  [Gerstner & van Hemmen 1992](https://doi.org/10.1088/0954-898X_3_2_004);
  WTA via lateral inhibition —
  [Eliasmith 2005](https://doi.org/10.1162/0899766053630332).
- **SDM and capacity**: [Kanerva 1988](https://doi.org/10.1109/18.32123);
  Hopfield ≡ attention — [Ramsauer et al. 2020](https://arxiv.org/abs/2008.02217).
- **Coding**: comparison of rate/TTFS/population codes —
  [Guo et al. 2021](https://doi.org/10.3389/fnins.2021.638474).

## 5. Honest skepticism accepted in the design

- Capacity is determined by bytes, not spikes; an SNN store does **not** beat a
  vector database on capacity. That is why the retrieval rerank remains exact
  cosine, and the core's value is write/forget/consolidation policies as one
  dynamics.
- Simulating a plastic network on GPU is slower than dense matmul; the energy
  win exists only on neuromorphic silicon
  ([neuromorphic hardware benchmarking](https://doi.org/10.3389/fnins.2022.873935)).
  The v0–v1 target scale is personal memory (10⁴–10⁶ traces), where this is moot.
- Kill criteria: if on the update/contradiction/adversarial axes the core does
  not beat "exact search + decay heuristics", the SNN layer is demoted to the
  write path and the negative result is documented here.

## 6. Hardware (reference)

Loihi 2 — research-only (INRC); Hala Point (2024): 1.15B neurons, 128B synapses,
2.6 kW; BrainChip Akida AKD1000 — dev-kit $499, 1.2M neurons, on-chip learning.
Porting is an optional phase 4, not a plan dependency.
