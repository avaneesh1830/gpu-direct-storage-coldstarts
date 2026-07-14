# Phase 4 Plan — Killing the Engine-Init Floor with Snapshot/Restore

**Goal:** cut serverless LLM cold start from the current best of **41.8 s** to **single-digit
seconds**, by snapshotting a fully warmed vLLM process and restoring it — skipping weight
loading, `torch.compile`, and CUDA-graph capture entirely.

**Why this and not GDS:** every cold-start second that loaders and storage *can* remove has
already been removed (650 s → 42 s). What remains is a **~33 s engine-init/compile floor** that
is storage-independent and untouchable by GDS. Snapshot/restore is the only technique that
removes it. It also **works on ordinary VMs** — unlike GDS, which is blocked on all rentable
cloud GPUs (see experiments 03 and 04).

---

## Background: the tooling exists and is production-grade

| Tool | What it does |
|---|---|
| **`cuda-checkpoint`** | NVIDIA CLI that saves/restores a running process's **GPU state** (VRAM contents, CUDA context) |
| **CRIU** | Linux tool that freezes a process's **CPU state** to disk (memory, file descriptors, execution state) and restores it |
| **NVIDIA Dynamo Snapshot (v1.2, June 2026)** | Production system combining both for **vLLM workers on Kubernetes** |

Reported results (NVIDIA): restore in **2.25 s from S3**, **9 s from local NVMe**; KV-cache
unmap (`cuMemUnmap` / `cuMemRelease`) shrinks the checkpoint artifact dramatically
(~190 GiB → ~6 GiB for a small model on B200).

**Known limitations to design around:**
- vLLM workers only, limited preview
- Multi-GPU / tensor-parallel: limited validation *(consistent with our finding that TP hurts
  cold start anyway — so target single-GPU, which is also the right serverless unit)*
- Specialized workers (multimodal, embedding, diffusion) unsupported

---

## The baseline to beat (from experiments 03 & 04)

| 123B (Mistral-Large-AWQ), single GPU, local NVMe, `enforce_eager` + InstantTensor | Time |
|---|---|
| Weight load | 8.2 s |
| Engine init (compile + graphs + warmup) | ~33 s |
| **Cold start (total)** | **41.8 s** |
| **Target with snapshot/restore** | **< 10 s** |

---

## Experiment design

### Phase 4a — Prove the mechanism (small model, ~1 day)

Establish that snapshot/restore works at all, on a cheap single-GPU box.

1. Start vLLM with the 1B model, fully warmed (weights loaded, graphs captured, one inference done).
2. `cuda-checkpoint --toggle --pid <pid>` to checkpoint GPU state.
3. `criu dump` the process tree to disk.
4. Kill the process. Drop the page cache.
5. `criu restore` + `cuda-checkpoint` to bring it back.
6. **Measure:** restore wall-clock, checkpoint artifact size on disk.
7. **Verify correctness:** the restored server must return **identical outputs** (temp=0, same
   prompt, same first 20 tokens) as before the checkpoint. *A fast restore that produces wrong
   tokens is not a result.*

**Success criterion:** restore time < cold start time, with byte-identical outputs.

### Phase 4b — Scale across model sizes (~2–3 days)

Repeat for 1B / 10B / 30B / 110B, and record:

| Metric | Why |
|---|---|
| Restore time | the headline — vs the 41.8 s cold-start baseline |
| Checkpoint size on disk | determines storage cost and restore I/O |
| Restore time **cold cache** vs **warm cache** | serverless has an empty cache by definition |
| Restore from **local NVMe** vs **network storage** | is restore itself I/O-bound? |
| Correctness | identical outputs, every size |

**Key hypothesis to test:** restore time should scale with **checkpoint size / storage
bandwidth** — meaning our *earlier* findings still apply, just to a different artifact. If so,
**InstantTensor-style concurrent I/O and NVMe locality should accelerate restore too**, and the
whole project composes into one coherent stack.

### Phase 4c — KV-cache handling (the artifact-size problem)

vLLM pre-allocates a huge KV cache (in our runs, ~65 GiB on an 80 GB card). Naively, that lands
in the checkpoint and makes it enormous.

- Measure the naive checkpoint size.
- Apply the `cuMemUnmap`/`cuMemRelease` KV-unmap approach (as Dynamo Snapshot does) and measure
  the reduction.
- Trade-off to quantify: does releasing the KV cache re-introduce allocation cost on restore?

### Phase 4d — The full stack, and the cost model (~2 days)

Compose everything and produce the deliverable table:

| Strategy | 123B cold start | Requires |
|---|---|---|
| Naive (network disk, default loader, CUDA graphs) | 650 s | — |
| \+ local NVMe | 214 s | instance choice |
| \+ `enforce_eager` | ~110 s | one flag |
| \+ InstantTensor | **41.8 s** | `pip install` |
| **\+ snapshot/restore** | **target < 10 s** | CRIU + `cuda-checkpoint` |
| *(GDS)* | *blocked on all rentable cloud GPUs* | bare metal |

Then the serverless economics: **$/cold-start × starts/day**, and the scale-to-zero argument
(how short a cold start must be before keeping GPUs idle stops being worth it).

---

## What to run it on

**No special hardware needed** — this is the whole point. A single-GPU box with local NVMe is
enough (the same class of machine that failed the GDS test works fine here).

Recommended: single H100/A100 with local NVMe, ~$2–4/hr. Run `preflight_check.sh` first as
always (it also confirms storage speed, which matters for restore I/O).

## Risks and honest unknowns

- **`cuda-checkpoint` + CRIU is preview-grade for vLLM** — expect breakage; that is itself a
  reportable finding.
- **Restore may be I/O-bound on a huge artifact** — if the checkpoint is ~65 GB, restoring it
  from a 2 GB/s disk costs ~30 s and we have merely *moved* the bottleneck. **Phase 4c (KV
  unmap) is therefore load-bearing, not optional.**
- **Cross-node / cross-GPU restore** (restore onto a *different* physical GPU) is the real
  serverless requirement and is the least-validated part. Worth testing explicitly.
- Multi-GPU TP restore has limited upstream validation — deliberately out of scope, and our own
  data says TP is the wrong choice for cold starts anyway.

## Deliverable

A cold-start optimization stack with measured numbers at every layer, an honest account of what
each technique can and cannot fix, and a recommendation matrix (model size × storage type →
best strategy) — plus the negative-but-useful GDS result already established.
