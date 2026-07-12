# Internship Roadmap — GPU Direct Storage for LLM Cold-Start Reduction

> **Project goal:** Reduce cold-start times for LLM inference (serverless
> deployments and general startup) using GDS / CRIU / cuda-checkpoint —
> saving cost for users and enabling resource clawback for enterprises.

Each phase has a verifiable deliverable. Don't start a phase until the previous
one's numbers are written down — the whole story is "baseline → optimization →
measured delta."

---

## Phase 1 — Baseline: where does cold-start time go? *(current, ~1-2 weeks)*

Run the Experiment 2 protocol (this repo) on all 4 GPUs × 4 model sizes.

**Questions to answer with data:**
1. What fraction of cold start is disk read vs H2D copy vs engine init vs CUDA-graph compile?
2. Does the default loader saturate the disk? (`cold − warm` vs `disk_cold`)
3. How does the breakdown shift with model size (3 GB → 65 GB)?
4. Does GPU generation matter for loading, or only storage/PCIe? (hypothesis: GPU barely matters — that itself is a finding)

**Deliverable:** stacked-bar chart per GPU (disk / H2D / init / compile) + a
one-page write-up naming the dominant stage at each model size.
**Verify:** every results.jsonl row reproducible within ~10% on a rerun.

## Phase 2 — Software loaders: how far can you get *without* GDS? *(~2 weeks)*

vLLM ships alternative load formats — benchmark them against the Phase 1 baseline
on one GPU (H100) across all sizes:

| Option | What it does |
|---|---|
| `--load-format runai_streamer` | streams weights with concurrent readers |
| `--load-format fastsafetensors` | optimized safetensors loading (GDS-capable) |
| `--load-format tensorizer` | serialized-tensor format, fast deserialization |
| `safetensors` default | the baseline |

Also: page-cache pre-warming (`vmtouch`) as a "poor man's fast load" datapoint.

**Deliverable:** load-time table (loader × model size), $/cold-start estimate.
**Verify:** identical model outputs (same first 20 tokens, temp=0) across loaders.

## Phase 3 — GPU Direct Storage *(core of the internship, ~3-4 weeks)*

GDS (cuFile / nvidia-fs) moves data NVMe → GPU directly, bypassing the CPU
bounce buffer. This is the headline experiment.

1. Verify instance support: local NVMe, `nvidia-fs` kernel module, compatible filesystem (ext4/xfs), `gdscheck -p`.
2. Micro-benchmark first: `gdsio` raw NVMe→GPU vs POSIX read + `cudaMemcpy` — establishes the hardware delta before any LLM is involved.
3. End-to-end: `fastsafetensors` with GDS enabled under vLLM vs Phase 2 numbers.
4. Sweep model sizes — GDS should matter more as bytes grow (110B is the showcase).

**Deliverable:** GDS vs non-GDS cold-start table + honest analysis of when GDS
does *not* help (page cache already hot, network-attached storage, small models).
**Verify:** gdsio confirms the direct path is active (not silent fallback to bounce buffer — check `cuFile` logs).

## Phase 4 — Snapshot/restore: skip loading entirely *(~2-3 weeks)*

Loading faster is good; not loading at all is better. A warmed-up vLLM process
can be checkpointed and restored:

| Approach | What it covers |
|---|---|
| `cuda-checkpoint` + CRIU | full process + GPU state snapshot/restore |
| vLLM sleep mode (`--enable-sleep-mode`) | weights offloaded/restored in-process |

Measure: restore time vs Phase 3's best cold start, snapshot size on disk,
restore correctness (server answers identically after restore).

**Deliverable:** cold boot vs GDS-load vs CRIU-restore comparison — the
"serverless cold-start menu" with time and $ per option.
**Verify:** restored server passes the same 5-prompt benchmark with identical outputs.

## Phase 5 — Synthesis & recommendation *(~1 week)*

- Cost model: instance $/hr × cold-start seconds × starts/day for a serverless fleet; show $ saved per option.
- Resource-clawback story: how fast restore enables scale-to-zero without SLA pain.
- Final report + charts; recommendation matrix (model size × storage type → best strategy).

**Deliverable:** final write-up / presentation for the mentor.

---

## Practical notes

- Keep using the exp-1 pattern: build image once, pull everywhere; shut instances down after each session.
- GDS needs **local NVMe** — when picking Brev instances for Phase 3, check the storage type, not just the GPU.
- Log *everything* to results.jsonl with labels; the final report is only as good as the baseline hygiene.
