# Experiment 03 — LLM Cold-Start Loading & GPU Direct Storage

**Question:** When you cold-start an LLM inference server, where does the time go —
and can GPU Direct Storage (GDS) make it faster?

Experiment 02 showed token *generation* is memory-bandwidth-bound. This experiment
asks the complementary question: what limits *getting the model serving in the first
place* — the cold-start latency that dominates serverless LLM cost, autoscaling, and
scale-to-zero.

---

## Headline results

**Cold start is storage-bound, not GPU-bound.**

| 110B model | H100 | H200 | B300 | RTX PRO 6000 |
|---|---|---|---|---|
| Storage bandwidth | 0.12 GB/s | 0.12 GB/s | 0.83 GB/s | 1.55 GB/s |
| **Cold start** | **650 s** | **650 s** | **225 s** | **213 s** |
| Warm start (RAM-cached) | 114 s | 112 s | 126 s | 164 s |
| Decode (tok/s) | 27 | 27 | 45 | 21 |

- **H100 ≡ H200** cold start within 1% — identical storage, different GPU → the GPU is
  irrelevant to cold start.
- Faster storage → proportionally faster cold start. On a local NVMe box (2.8 GB/s
  measured) the 110B cold start dropped to **214 s**, and with loader + eager tuning to **80 s**.
- **Warm starts converge at ~110–190 s regardless of GPU or storage** — a storage-independent
  engine-init/compile floor. This is what snapshot/restore (CRIU, `cuda-checkpoint`) targets.

**GDS could not be demonstrated on any available cloud instance** — see [GDS investigation](#gds-investigation).

**The InstantTensor loader cut cold-cache weight loading up to ~9×** (110B: 72 s → 8 s)
and the end-to-end cold start ~2.5×, *without* GDS — via direct I/O + concurrency that
bypasses the page cache and saturates the disk. See [ANALYSIS.md §8](./ANALYSIS.md).
This is the largest single loading lever found on standard (non-bare-metal) cloud hardware.
It also **beat `fastsafetensors` by ~4x while running on slower storage** — concurrency, not
raw disk speed, is what wins. Adding GDS on top is *estimated* to save only a few more
seconds ([ANALYSIS.md §9](./ANALYSIS.md)), because the engine-init floor then dominates.

---

## Method

### Models (identical artifacts on every GPU — byte-for-byte, for a fair comparison)

| Tier | Model | Precision | On disk |
|---|---|---|---|
| 1B | Qwen2.5-1.5B-Instruct | BF16 | 3.1 GB |
| 10B | Qwen2.5-7B-Instruct | BF16 | 15.2 GB |
| 30B | Qwen2.5-32B-Instruct-AWQ | AWQ INT4 | 19.3 GB |
| 110B | Mistral-Large-2407-AWQ (123B) | AWQ INT4 | 64.9 GB |

AWQ INT4 is used for 30B/110B on *every* GPU — even those with enough VRAM for BF16 —
so all GPUs load identical bytes. This is a *loading* experiment, not a quality benchmark.

### Protocol (`run.sh` → `benchmark.py`)

Each model is loaded three ways, plus a raw disk read, to decompose the cold start:

| Step | Page cache | What the timing contains |
|---|---|---|
| `first_download` | — | HF download + load (setup, not analyzed) |
| `cold` | **dropped** | disk read + host→device copy + engine init + compile — **the cold-start number** |
| `warm` | hot | same, minus disk (weights served from RAM) |
| `disk_cold` | dropped | raw sequential read of the weight files only |

"Cold" is manufactured deliberately: `sync; echo 3 > /proc/sys/vm/drop_caches` flushes
the OS page cache so the load genuinely hits disk — the true serverless cold-start
scenario. **Decomposition comes from differencing:** `cold − warm ≈ disk time`;
`disk_cold` = the disk's ceiling; `warm − eager_warm ≈ torch.compile + CUDA-graph time`.

Two configurations were run per GPU:
1. **Baseline** — default vLLM loader, CUDA graphs on (custom image, vLLM 0.23.0).
2. **`eager+fst`** — `--enforce-eager --load-format fastsafetensors` (NGC image, vLLM 0.22.1).

### Metrics (per row in `results/*.jsonl`)

`llm_init_s` (headline: wall-clock to engine-ready) · `ttft_first_ms` / `ttft_warm_ms`
· `decode_tps` · `disk_read_gbps` · `effective_load_gbps` (model bytes ÷ init time —
the number GDS aims to raise).

---

## Findings

**1. Cold start belongs to the disk.** Disk read is 38 % of the 1B cold start but **83 %**
of the 110B on slow storage. The bigger the model, the more brutally storage dominates.

**2. The loader is already efficient.** On slow storage vLLM reads at ~97 % of the raw
disk ceiling (`cold − warm ≈ disk_cold`). Software can't fix a slow disk — only faster
storage or a fundamentally different path (GDS) can.

**3. Fast storage relocates the bottleneck to the engine floor.** On the NVMe box the
110B cold start (214 s) sits just above its warm start (188 s): disk is now ~26 s, and
`torch.compile` + CUDA-graph capture + engine init dominate. That ~110–190 s floor is
storage-independent → the CRIU / `cuda-checkpoint` opportunity.

**4. `enforce_eager` is a free lunch for large models.** It skips CUDA-graph capture,
saving 26–67 s of startup, at a decode cost of −60 % (1B) but only **−3 %** (110B).
Large-model serverless should run eager.

**5. From page cache the path reaches ~5 GB/s** — a 110B *could* load in ~13 s if storage
kept up. On slow disk it takes 539 s. That ~40× gap is the entire motivation for GDS.

Full tables and honest caveats: [`ANALYSIS.md`](./ANALYSIS.md).

---

## GDS investigation

The goal was to run the loader over GPU Direct Storage (`nvidia_fs` / cuFile) and measure
the speedup. It never engaged — and *why* is the result.

| Instance | Storage | `nvidia_fs` | GDS result |
|---|---|---|---|
| Brev H100/H200/B300 | virtio (network) | not present | Impossible — no local disk to DMA from |
| AWS RTX PRO 6000 | **local NVMe** (2.8 GB/s) | **loaded** | `nogds=True` — `gdscheck` shows `NVMe : compat`, `use_pci_p2pdma : false` |

On the AWS box the NVMe was physically present and the driver loaded, yet cuFile still
disabled GDS — even with `--privileged` and all `/dev/nvidia-fs*` devices passed into the
container (load times were identical with and without, confirming no path). **Root cause:**
the AWS Nitro hypervisor does not expose PCIe peer-to-peer DMA (`use_pci_p2pdma: false`)
to the guest, so the GPU cannot read the NVMe's buffers directly. A local NVMe is
necessary but **not sufficient** for GDS — it also needs bare-metal or explicit
PCIe-P2P passthrough.

**Conclusion:** true GDS is not demonstrable on standard *virtualized* cloud GPU
instances. It requires bare-metal (e.g. AWS `.metal`, OCI bare-metal shapes, Spheron
bare-metal nodes) where `gdscheck -p` reports `NVMe : Supported` and `use_pci_p2pdma : true`.

### What GDS actually changes — the data path

**Normal loading (what every run in this experiment did):**

```
NVMe SSD ──► CPU RAM (bounce buffer) ──► GPU VRAM
```

The CPU stages every byte in a system-RAM buffer, then a second copy pushes it to the
GPU. Two hops, CPU-mediated. When cuFile logs `nogds=True`, this is the path it uses.

**Real GDS:**

```
NVMe SSD ──────── direct DMA over PCIe ────────► GPU VRAM
```

The GPU's DMA engine reads straight from the SSD; the CPU and its RAM are bypassed
entirely. One hop, zero-copy. This is the path `use_pci_p2pdma: true` would enable.

### How to get real GDS

1. Provision a **bare-metal** GPU instance with **local NVMe** (no hypervisor between the
   GPU and the drive): AWS `.metal`, OCI bare-metal GPU shapes, Spheron/CoreWeave bare-metal.
2. On boot, verify: `gdscheck -p` must show `NVMe : Supported` **and** `use_pci_p2pdma : true`.
   (If it shows `compat` / `false`, GDS will silently disable itself — as it did here.)
3. Run the **unchanged** `eager+fst` command below — with a real path, cuFile logs GDS
   *enabled* instead of `nogds=True`. No code change is needed; the loader auto-detects.

### What difference it would make — and the honest caveat

GDS accelerates **only the disk-read portion** of the cold start; it does not touch the
~110–190 s engine-init/compile floor. So its upside is bounded by how much of the cold
start is disk:

| 110B cold start | Disk portion | Engine floor | GDS upside |
|---|---|---|---|
| Slow virtio (0.12 GB/s) | ~537 s | ~113 s | Large — disk dominates |
| Local NVMe, no GDS | ~26 s | ~188 s | **Small — disk is already tiny** |

The nuance worth stating: **GDS helps most exactly where fast NVMe already helps** — and
once storage is fast, the disk portion is already small, so GDS's benefit to *single-stream*
cold-start loading is modest. Its real value appears at **scale** (many concurrent model
loads, where the CPU bounce buffer and RAM bandwidth become the bottleneck) and in
**freeing CPU/RAM** for other work during load. This experiment's data suggests that for
single-instance cold starts, **storage locality + eager mode capture most of the win, and
snapshot/restore (CRIU / `cuda-checkpoint`) — which erases the engine floor — is the
larger remaining lever than GDS.**

---

## Running it

```bash
# On a fresh GPU instance (Docker + NVIDIA Container Toolkit installed):
scp benchmark.py run.sh <user>@<instance>:~/
export HF_TOKEN=hf_...            # HuggingFace token for downloads
tmux new -s bench                 # long-running: survive disconnects

# Baseline, all four sizes:
for S in 1b 10b 30b 110b; do SIZE=$S ./run.sh 2>&1 | tee ~/results/log_${S}.txt; done

# eager + fastsafetensors config (NGC image), reusing the cache:
for S in 1b 10b 30b 110b; do
  CACHE=~/model-cache PREFIX=gdseager_ \
  EXTRA="--enforce-eager --load-format fastsafetensors" \
  IMAGE=nvcr.io/nvidia/vllm:26.06-py3 \
  SIZE=$S ./run.sh 2>&1 | tee ~/results/log_${S}_gdseager.txt
done
```

`run.sh` env vars: `SIZE` (1b/10b/30b/110b) · `IMAGE` · `CACHE` (weights dir — point at
local NVMe to test fast storage) · `PREFIX` (label prefix) · `EXTRA` (extra `benchmark.py` flags).

## Files

```
benchmark.py   # load / disk benchmark modes → one JSONL row per run
run.sh         # 4-step cold/warm/disk protocol for one model size
ROADMAP.md     # 5-phase internship plan (baseline → loaders → GDS → CRIU → synthesis)
ANALYSIS.md    # full result tables, decomposition, and caveats
results/
  *.jsonl      # per-GPU/config data (H100, H200, B300, RTX, RTX_NVME, RTX_NVMEGDS, RTX_REALGDS)
  logs/        # raw vLLM console logs for every run
```

### Result file guide

| File | GPU / config |
|---|---|
| `H100.jsonl` `H200.jsonl` `B300.jsonl` `RTX.jsonl` | baseline + `eager+fst`, per GPU |
| `RTX_NVME.jsonl` | RTX box, baseline loading from **local NVMe** |
| `RTX_NVMEGDS.jsonl` | RTX box, `eager+fst` from NVMe (GDS attempted, `nogds=True`) |
| `RTX_REALGDS.jsonl` | RTX box, GDS forced via `--privileged` + cuFile mount — still `nogds=True` |
