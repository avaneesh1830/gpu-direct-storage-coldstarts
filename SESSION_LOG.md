# Complete Session Chronicle — LLM Cold-Start, Loaders, TP8 & the GDS Verdict

**Project:** Reducing serverless-GPU cold starts (GDS / CRIU / cuda-checkpoint internship project)
**Dates:** July 6–15, 2026 · **Author:** Avaneesh Aroor (with Claude as driver/analyst)
**Repo:** [github.com/avaneesh1830/gpu-direct-storage-coldstarts](https://github.com/avaneesh1830/gpu-direct-storage-coldstarts)
**Every number in this document was recomputed from the raw `results/*.jsonl` and log files on 2026-07-15.**

---

## 0. TL;DR — the six findings

1. **Cold start is storage-bound, not GPU-bound.** H100 and H200 on identical 0.12 GB/s storage
   produced identical 110B cold starts (650.0 s vs 649.6 s). Faster storage (B300 0.83 GB/s,
   RTX 1.55 GB/s) cut it to 225 s / 213 s. The GPU model was irrelevant.
2. **GDS never engaged anywhere — including a full 8× A100 DGX-class node.** Every instance
   reported `use_pci_p2pdma: false` (cuFile compat mode). GDS is gated on **virtualization,
   not node size** — the "providers hold GDS back for full nodes" hypothesis is disproven.
   The viable cloud route to *real* GDS is RDMA network filesystems (AWS FSx for Lustre), or bare metal.
3. **GDS is also the wrong lever.** With a concurrent loader, weight load is ~8 s of a ~42 s
   cold start; a **~33 s engine-init/compile floor** dominates and GDS cannot touch it.
4. **InstantTensor (`--load-format instanttensor`) is the biggest working lever found**:
   110B cold-cache weight load **71.9 s → 8.2 s (8.8×)**; end-to-end cold start 105.9 s → 41.8 s
   on the same box. Direct I/O + concurrency; cache-independent; no special hardware.
   Externally replicated by a collaborator (2.9× on Nemotron-30B).
5. **TP8 (8-GPU tensor parallel) hurts cold start** for models that fit on one GPU
   (110B: 41.8 s single-GPU → 73.0 s TP8) and is **impossible** for models whose attention
   heads aren't divisible by 8 (1B: 12 heads, 10B: 28 heads).
6. **The cumulative stack:** 650 s (naive) → **41.8 s** (local NVMe + eager + InstantTensor) —
   ~15× with no exotic hardware. The proposed Phase 4 (CRIU + cuda-checkpoint snapshot/restore,
   as productized by NVIDIA Dynamo Snapshot) targets single-digit seconds.

---

## 1. Goal and context

Internship goal: *"Leverage GPU Direct Storage to improve model & inference runtime loading
performance for LLMs — reducing serverless cold-start times (GDS / CRIU / cuda-checkpoint),
for cost savings and resource clawback."*

Experiment 1 (June, separate repo folder) benchmarked *inference* and showed decode is
memory-bandwidth-bound. This session (experiments 2–4) pivoted to *loading*: where does
cold-start time actually go, and what removes it?

## 2. Method

### Models (identical artifacts on every GPU — byte-for-byte fair comparison)

| Tier | Model | Precision | On disk |
|---|---|---|---|
| 1B | Qwen/Qwen2.5-1.5B-Instruct | BF16 | 3.1 GB |
| 10B | Qwen/Qwen2.5-7B-Instruct | BF16 | 15.2 GB |
| 30B | Qwen/Qwen2.5-32B-Instruct-AWQ | AWQ INT4 | 19.3 GB |
| 110B | casperhansen/mistral-large-instruct-2407-awq (123B params) | AWQ INT4 | 64.9 GB |
| (405B) | Meta-Llama-3.1-405B-AWQ (~203 GB) | AWQ INT4 | added to harness; run cancelled for cost |

AWQ INT4 everywhere for 30B/110B — even on GPUs that fit BF16 — because this is a *loading*
study: same bytes on every GPU.

### Protocol (`run.sh` → `benchmark.py`), per model per config

| Step | Page cache | Measures |
|---|---|---|
| `first_download` | — | HF download + load (setup, not analyzed) |
| `cold` | **dropped** (`sync; echo 3 > drop_caches`) | disk + H2D + engine init + compile — **the cold start** |
| `warm` | hot | same minus disk (weights served from RAM) |
| `disk_cold` | dropped | raw sequential read of weight files (disk ceiling) |

Decomposition by differencing: `cold − warm ≈ disk time`; `warm − eager-warm ≈ compile/graphs`.
"Cold" is *manufactured* (page cache flushed), not "freshly downloaded" — cold means **not in RAM**.

### Key metrics per JSONL row
`llm_init_s` (wall-clock to engine-ready — the headline) · `ttft_first_ms` / `ttft_warm_ms`
(sequential, prefix-cache off) · `decode_tps` · `disk_read_gbps` · `effective_load_gbps` ·
plus vLLM's `Loading weights took Xs` log line (the loader's isolated job).

### Tools built
- **`benchmark.py`** — load/disk modes, `--enforce-eager`, `--load-format
  {auto,fastsafetensors,runai_streamer,tensorizer,instanttensor}`, `--tp-size N`
- **`run.sh`** — the 4-step protocol; env vars `SIZE / IMAGE / CACHE / PREFIX / EXTRA`
- **`preflight_check.sh`** — <1-minute instance vetting: GPU, RAM (≥ model size for warm runs),
  storage type + measured `dd` speed, **GDS readiness** (`gdscheck -p`, `use_pci_p2pdma`,
  `nvidia_fs`), Docker GPU passthrough. Predicted the GDS outcome correctly on every box.
- **`vllm-loaders` image** — `FROM nvcr.io/nvidia/vllm:26.06-py3` + `pip install instanttensor
  runai-model-streamer` (neither loader ships in the NGC image)

### Images
Baseline runs: custom `avaneesharoor/ml-experiments:latest` (vLLM 0.23.0, entrypoint = API
server → override with `--entrypoint python3`). Loader/eager runs: NGC `nvcr.io/nvidia/vllm:26.06-py3`
(vLLM 0.22.1) or the derived `vllm-loaders`. Small version skew noted as a caveat.

## 3. Infrastructure log (all instances now deleted)

| # | Instance | GPU | Backend / SSH user | Storage (measured) | Used for |
|---|---|---|---|---|---|
| 1 | Brev H100 | H100 80 GB | Hyperstack / `ubuntu` | virtio, **0.12 GB/s** | exp-03 baseline + eager/fst |
| 2 | Brev H200 | H200 141 GB | Hyperstack / `ubuntu` | virtio, **0.12 GB/s** | exp-03 baseline + eager/fst |
| 3 | Shadeform B300 | B300 SXM6 275 GB | Shadeform / `shadeform` | virtio, **0.83–0.99 GB/s** | exp-03 baseline + eager/fst (B200 unavailable → B300 substituted) |
| 4 | RTX (user-run) | RTX PRO 6000 96 GB | AWS-backed | NVMe-backed EBS, **~1.55 GB/s** | exp-03 4th-GPU baseline + eager/fst |
| 5 | AWS DLAMI "GDS box" | RTX PRO 6000 96 GB | AWS / `ubuntu` | EBS root **0.03 GB/s**; local NVMe `/opt/dlami/nvme` **2.8 GB/s (dd)** / ~1.5 GB/s (our sequential bench) | NVMe tier + 3 GDS attempts |
| 6 | Shadeform H100 #2 | H100 80 GB | Shadeform / `shadeform` | virtio, **2.0–2.1 GB/s** | InstantTensor controlled ablation |
| 7 | Azure DGX Cloud 8× | **8× A100-SXM4-80 GB** | Brev alias `h100-2` (port 8888 only; misleadingly *named* "h100-2") | 8× local NVMe RAID `/ephemeral` 7 TB, **2.5–2.7 GB/s dd** (3.1–4.3 GB/s in our bench) | exp-04: GDS-on-full-node test, TP8, 3-loader shoot-out · **$19.30/hr** |

RAM was 141–1771 GB across boxes — always ≥ the 80 GB needed to hold the 110B in page cache
for valid warm runs (instances under ~128 GB RAM were explicitly rejected).

## 4. Results

### 4.1 Baseline cold start (seconds) — the 4-GPU grid

| Model | H100 | H200 | B300 | RTX |
|---|---|---|---|---|
| 1B | 83.9 | 84.8 | 64.4 | 84.6 |
| 10B | 187.5 | 187.8 | 82.7 | 96.5 |
| 30B | 243.3 | 245.7 | 121.8 | 121.8 |
| **110B** | **650.0** | **649.6** | **225.1** | **212.8** |

Warm starts: 110B = 113.6 / 112.4 / 126.2 / 163.6 — a **storage-independent engine floor**.
Raw disk reads (110B): 523.4 s / 523.6 s / 78.5 s / 41.7 s → 0.12 / 0.12 / 0.83 / 1.55 GB/s.
Decode sanity (110B): 27.0 / 27.0 / 45.3 / 21.4 tok/s (B300's HBM3e bandwidth shows).

**Findings:** disk share of cold start grows 38 % (1B) → 83 % (110B) on slow storage; the
default loader is ~97 % disk-saturated when cold (`cold − warm ≈ disk_cold`) — software cannot
fix a slow disk; from page cache the pipeline ran at ~5 GB/s (a 110B *could* load in ~13 s).
Reproducibility: repeated 1B baselines matched within ~2 % (H100, B300).

### 4.2 `enforce_eager` + fastsafetensors ("gdseager" config, exp-03)

110B cold / warm: H100 568.9/48.8 · H200 564.2/45.1 · B300 121.3/38.8 · RTX 78.6/49.8.
Eager removes torch.compile + CUDA-graph capture (~57–87 s depending on box). Decode penalty:
−60 % (1B) → **−3 % (110B)** — *free lunch for large models, bad trade for small ones*.
`nogds=True` in every one of these runs — this config measured the fastsafetensors loader +
eager, **not** GDS.

### 4.3 Storage-tier ladder (110B, same protocol)

| Storage | Disk BW | Cold start | + eager+loader |
|---|---|---|---|
| Brev virtio | 0.12 GB/s | 650 s | 564 s |
| B300 virtio | 0.83 GB/s | 225 s | 121 s |
| RTX EBS | 1.55 GB/s | 213 s | 79 s |
| AWS local NVMe | ~1.5–2.8 GB/s | 214 s | 80–82 s |

### 4.4 Loader shoot-out — single GPU (controlled: eager constant, only loader varies)

**Cold-cache weight load** (vLLM `Loading weights took`), Shadeform H100 #2, ~2 GB/s storage:

| Model | default | InstantTensor | Speedup |
|---|---|---|---|
| 1B | 1.45 s | 1.52 s | 0.95× (fixed overhead dominates) |
| 10B | 17.98 s | 2.44 s | **7.4×** |
| 30B | 14.11 s | 3.81 s | **3.7×** |
| **110B** | **71.86 s** | **8.21 s** | **8.8×** |

End-to-end cold start 110B: 105.9 s → **41.8 s** (2.5×). Decode unchanged (26.3 vs 26.2 tok/s).

**Why InstantTensor wins:** direct I/O (O_DIRECT) + pipelined concurrent reads. It pulled
**~8 GB/s from a disk that single-stream `dd` measured at ~2 GB/s** — concurrency beats raw
disk rating. It is **cache-independent** (110B: 8.21 s cold vs 8.34 s warm — O_DIRECT bypasses
the page cache), which is exactly the serverless property (cold caches by definition). Its cold
load even beats the *default loader reading from warm RAM* (10.8 s).

**vs fastsafetensors:** fst on *faster* storage (RTX NVMe, 2.8 GB/s) loaded the 110B cold in
**32.7 s**; InstantTensor on *slower* storage (2 GB/s) did it in **8.2 s** — ~4× faster despite
the storage disadvantage, because fst reads near-single-stream (on 0.12 GB/s virtio it took
524 s = exactly disk rate).

**External replication (collaborator, A100 on Brev, Nemotron-3-Nano-30B-A3B, BF16→FP16 cast):**
safetensors 38.45 s → InstantTensor **13.23 s (2.9×)**, Run:ai **23.65 s (1.6×)** weight load.
Independently confirms the ranking: **InstantTensor > Run:ai > default** on weight load.

### 4.5 TP8 — 8× A100 SXM (exp-04)

| Model @ TP8 | Loader | Cold weight-load | Cold start | Warm |
|---|---|---|---|---|
| 30B | auto | 5.76 s | 75.1 s | 49.6 s |
| 30B | instanttensor | **3.66 s** | **72.9 s** | 51.9 s |
| 30B | runai_streamer | (own logging) | 85.6 s | 56.9 s |
| 110B | auto | 20.50 s | 89.7 s | 50.4 s |
| 110B | instanttensor | **5.83 s** | **73.0 s** | 57.0 s |
| 110B | runai_streamer | (own logging) | 111.4 s | 77.3 s |

- **TP8 vs single GPU (110B, InstantTensor): 73.0 s vs 41.8 s — TP8 made cold start *worse*.**
  Sharding sped the weight read (8.2 → 5.8 s) but NCCL + 8 worker processes added ~20–25 s of
  engine init (visible in warm: 57 s vs 40 s). *TP8 is for models too big for one GPU, not for latency.*
- **TP8 impossible for 1B (12 heads) and 10B (28 heads)** — attention heads must divide by TP size
  (`ValueError` at engine build). Hard architectural constraint.
- **Loader advantage shrinks with sharding:** 3.5× at 110B → 1.6× at 30B, because 19 GB / 8 GPUs
  = 2.4 GB per GPU — loading becomes trivial and the ~50 s engine floor dominates.
- A planned 405B run (the model that genuinely *needs* 8 GPUs) was cancelled before download to cap
  cost; partial 1B TP1 rows exist from a cancelled comparison deemed unnecessary.

### 4.6 The GDS investigation — three attempts, one verdict

| Attempt | Hardware | `nvidia_fs` | `gdscheck` | Outcome |
|---|---|---|---|---|
| Brev H100/H200/B300 | virtio network disk | absent | n/a | Impossible — no local device to DMA from |
| AWS DLAMI RTX | real local NVMe | **loaded**, `/dev/nvidia-fs0-15` | `NVMe: compat`, `use_pci_p2pdma: false` | `nogds=True`; `--privileged` + cufile.json + all devices passed → **identical load times** (proof passthrough changed nothing) |
| **Azure 8× A100 SXM DGX Cloud** | 8× local NVMe RAID | **loaded; GPU↔NVMe peer paths enumerated** in `/proc/driver/nvidia-fs/peer_distance` | `NVMe: compat`, `use_pci_p2pdma: false` | Blocked — furthest anyone got, same wall |

**Verdict:** GDS is gated on **virtualization**. All three clouds run VMs; the hypervisor does
not expose PCIe peer-to-peer DMA to guests regardless of GPU count, NVMe presence, or DGX
branding ("DGX Cloud" ≠ bare-metal DGX). *Having an NVMe listed on the instance page is
necessary but not sufficient* — only `gdscheck -p` showing `NVMe: Supported` + `use_pci_p2pdma:
true` (or `systemd-detect-virt` = `none`) proves GDS capability.

**Collaborator's compat-mode result reconciled:** their `use_compat_mode: true` and our
`use_pci_p2pdma: false` describe the *same state* — cuFile running its CPU-bounce compat path.
The 3×/1.6× speedups they saw are the loaders' concurrency, not GDS.

**Routes to real GDS (in practicality order):**
1. **RDMA network filesystems — AWS FSx for Lustre + GPUDirect** (per AWS's own blog) — GDS over
   GPU↔NIC↔Lustre RDMA, sidestepping local-NVMe P2P entirely. Most promising cloud path; untested by us.
2. Bare metal: OCI `BM.GPU.*`, AWS `*.metal`, Spheron/CoreWeave bare-metal, real DGX (via mentor).

**Even if GDS worked, the estimate says ~10 %:** it would shave the 8 s load toward ~3–5 s while
the ~33 s engine floor stands. GDS's real value is projected at fleet-scale concurrent loads
(shared CPU bounce-buffer contention), not single cold starts.

## 5. The serverless reframe & Phase 4

The contradiction "reduce cold start with GDS for serverless GPUs, but GDS doesn't work on
serverless GPUs" dissolves once roles are separated: *tenants* (us) can't enable GDS; *platform
operators* (Modal/RunPod/etc.) own the hypervisor and could. Our tenant-side data answers the
platform question — and says GDS isn't the right lever anyway.

**The remaining time is the engine-init floor → snapshot/restore (CRIU + `cuda-checkpoint`).**
NVIDIA productized exactly this: **Dynamo Snapshot v1.2 (June 2026)** — CRIU + cuda-checkpoint
for vLLM workers on K8s, reporting **2.25 s restores from S3 / 9 s from local NVMe**, with KV-cache
unmap shrinking artifacts ~190 GiB → ~6 GiB (their example). It works on ordinary VMs.
`PHASE4_PLAN.md` stages the experiment: prove mechanism (1B) → scale sizes → KV-artifact problem
→ full stack + cost model, with output-correctness gates throughout.
`MENTOR_SUMMARY.md` carries the two asks: bare-metal access to close GDS, and approval to pivot.

## 6. Mistakes made and how they were solved (complete, honest list)

1. **Wrong cache mount:** custom image sets `HF_HOME=/model-cache`; mounting `~/.cache/huggingface`
   silently re-downloaded every run. → Mount `/model-cache`.
2. **Entrypoint traps:** image entrypoint is the vLLM API server, and `python` doesn't exist →
   `--entrypoint python3` + script path.
3. **`--rm` wiping downloads** before a cache mount existed → persistent host cache dirs.
4. **Disk-full crashes (twice):** NGC image's newer HF hub re-downloads the 110B rather than
   reuse the old cache; 247 GB disks overflowed mid-run (`No space left on device`). → Pre-clean
   finished caches / `docker rmi` old images; container-written caches are **root-owned** → `sudo rm`.
5. **tmux misuse:** loops died with SSH disconnects (no tmux); one "session" contained a nested
   ssh-to-itself eating keystrokes. → `tmux new-session -d -s name "cmd"` fire-and-forget;
   verify with `tmux ls`; `export HF_TOKEN` *before* `tmux new`.
6. **Log clobbering:** pulling H200 logs with identical filenames overwrote local H100 baseline
   logs (jsonl data unaffected; key excerpts survive in analyses). → All pulls now GPU-prefixed.
7. **Missing `tee` on early runs** (H100/H200/B300 first 1B logs lost to scrollback) → tee everything.
8. **"InstantTensor" misassumed to be fastsafetensors** (my error; user caught it). It is a real,
   distinct loader (`scitix/InstantTensor`). Mislabeled partial rows deleted and re-run correctly.
9. **`benchmark.py` argparse rejected `instanttensor`** (not in `choices`) → every IT run died
   instantly with zero rows. → Added to choices.
10. **InstantTensor/Run:ai not in the NGC image** (`ModuleNotFoundError`) → built derived
    `vllm-loaders` image with both pip packages.
11. **`apt install nvidia-gds` broke the DGX host driver:** it pulled a whole new driver
    (nvidia-595-server) without its userspace — `nvidia-smi` and *all* Docker GPU containers died
    in a cascade of missing files. → Install matching userspace
    (`libnvidia-compute/-cfg1/-encode/-decode-595-server`, `nvidia-utils-595-server`) and stub
    `/run/nvidia-persistenced/socket`, `/usr/bin/nvidia-cuda-mps-*`. Fully recovered. *Lesson:
    `nvidia-gds` is not a safe install on managed GPU images.*
12. **`gdscheck` needs `libcuda.so.1`, absent on DGX Cloud hosts** (CUDA userspace lives in
    containers) → fixed by (11); interim workaround: read `/proc/driver/nvidia-fs/*` directly.
13. **No public SSH on some boxes** (only port 8888/Jupyter; "provider doesn't allow port
    modifications") → use the Brev CLI's alias (`brev ls`; `ssh h100-2`), never the raw IP.
    Also: SSH usernames vary — `ubuntu` vs `shadeform` vs `nvidia`; brute-check on first contact.
14. **TP8 attempted on 1B/10B** — failed on head-divisibility (12, 28 not ÷ 8). Not fixable;
    reclassified as a *finding*.
15. **Run:ai "loses" overclaim (my error, corrected):** exp-04 compared *API-ready* times at TP8
    — noisy, engine-init-dominated, and I never extracted Run:ai's own weight-load number (it
    logs differently). Collaborator's clean single-GPU data shows Run:ai **beats baseline 1.6×**
    on weight load (InstantTensor 2.9×). Correct statement: *InstantTensor > Run:ai > default on
    weight load; compare loaders on weight-load, not API-ready; Run:ai ran stock concurrency
    (tunable via `--model-loader-extra-config '{"concurrency":N}'`).*
16. **Cost discipline lapses and fixes:** $19.30/hr node idled during debugging; a 405B (~203 GB)
    run was cancelled pre-download once TP8's conclusion was already established; TP1 confirmation
    runs cancelled as unnecessary. → `preflight_check.sh` now front-loads the go/no-go to <1 min.

## 7. Corrections ledger (claims revised during the session)

| Earlier claim | Corrected to |
|---|---|
| "InstantTensor = fastsafetensors" | Distinct loader; 4× faster than fst in like-for-like tests |
| "Run:ai Streamer loses to the default loader" | On *weight load* Run:ai beats default 1.6× (external data); it trails InstantTensor; my TP8 comparison was confounded by engine-init noise |
| "GDS strictly needs local-NVMe P2P / bare metal" | Bare metal **or** RDMA parallel filesystems (FSx for Lustre + GPUDirect) — the latter works on cloud |
| "gdseager" label in exp-03 | Renamed conceptually to *eager+fst*: GDS was never active (`nogds=True` everywhere) |

## 8. Repository map

```
gpu-direct-storage-coldstarts/
├── README.md                                  # mission, headline findings, experiment index
├── MENTOR_SUMMARY.md                          # status + GDS verdict + pivot proposal + asks
├── PHASE4_PLAN.md                             # CRIU/cuda-checkpoint experiment plan
├── experiment-01-rowracer-cpu-vs-gpu/         # (pre-existing) cuDF vs pandas
├── experiment-02-llm-inference-bottleneck/    # exp-1: TTFT/TPOT across H100/H200/B200
├── experiment-03-coldstart-loading-gds/       # 4-GPU cold-start grid, storage ladder, eager,
│   ├── README.md · ANALYSIS.md · ROADMAP.md   #   fst runs, GDS attempts 1–2, InstantTensor
│   ├── benchmark.py · run.sh · preflight_check.sh
│   └── results/ (H100/H200/B300/RTX/RTX_NVME/RTX_NVMEGDS/RTX_REALGDS/H100_2 .jsonl + logs/)
└── experiment-04-tp8-multiGPU-loaders-gds/    # 8×A100: GDS-on-full-node disproof, TP8,
    ├── README.md · ANALYSIS.md                #   3-loader shoot-out, mistakes/solutions record
    ├── benchmark.py · run.sh · preflight_check.sh
    └── results/ (A100x8_TP8.jsonl + logs/)
```
Local extras (this folder): `results/` master copies, `ANALYSIS.md`, `MENTOR_SUMMARY.md`,
`PHASE4_PLAN.md`, `preflight_check.sh`, this file. GDS & EE copies in `~/Desktop/loading w: GDS & EE/`.

## 9. Glossary (questions asked and answered along the way)

- **Node:** one physical server (GPUs typically come 4–8 per box, NVLink-connected). *Single-GPU
  slice* = renting one GPU of a shared node; *full node* = the whole machine. We tested a full
  node; GDS was still blocked — the wall is the hypervisor, not sharing.
- **Cold vs warm start:** cold = weights only on disk, page cache empty (the true serverless
  case — we *manufacture* it with `drop_caches`); warm = weights already in RAM. `cold − warm`
  isolates disk time. A model downloaded days ago still cold-starts if RAM was flushed.
- **VRAM vs system RAM vs disk:** VRAM = fits-or-not (110B-AWQ ≈ 62 GB needs >62 GB VRAM + KV
  headroom); RAM = must hold the model for a *valid warm measurement* (≥ ~80 GB for the 110B);
  disk size = crash-avoidance (~100 GB models + ~40 GB images); **disk speed = the cold-start
  clock** (cold ≈ bytes ÷ GB/s), and its share grows with model size.
- **PCIe vs SXM (instance label):** GPU form factor/interconnect. Irrelevant for single-GPU work;
  NVLink/SXM matters for multi-GPU traffic. Not related to whether storage is NVMe.
- **virtio (`vda`) vs `nvme` devices:** `vda`/QEMU disk = virtualized (often network) storage;
  a real `nvme*` with vendor model = local flash. But **local NVMe ≠ GDS-capable** — the P2P DMA
  door must be open (`use_pci_p2pdma: true`), which VMs deny. Quick tells:
  `lsblk -d -o NAME,MODEL` · `systemd-detect-virt` (`none` = bare metal) · `gdscheck -p`.
- **Tensor parallelism (TP):** splits each layer's attention heads/matrices across GPUs; head
  count must divide by TP size; adds NCCL + per-GPU worker init to startup.
- **Why the loaders differ:** default reads through the page cache, mostly single-stream →
  disk-rate-bound when cold. fst similar (GDS-capable but compat-blocked). Run:ai = concurrent
  threads → CPU buffer (tunable). InstantTensor = O_DIRECT + pipelined prefetch + concurrency →
  saturates the device beyond its single-stream rating and ignores cache state.

## 10. Open items

1. **FSx for Lustre + GPUDirect** — the untested, most-promising route to *real* GDS numbers on cloud.
2. **Bare-metal GDS** (mentor ask: DGX / OCI BM / AWS metal) — 1-minute preflight decides it.
3. **8× H100 preflight** — mentor's remaining checklist item; predicted to fail identically (~$0.30 to verify).
4. **Run:ai with tuned concurrency** — fairness follow-up before any final loader ranking.
5. **405B at TP8** — the model that *requires* 8 GPUs; the honest test of TP8's purpose.
6. **Phase 4: CRIU + cuda-checkpoint / Dynamo Snapshot** — the main event; targets <10 s restores.

---

*Also discussed during the session but out of scope here: interview framing of the skill set
demonstrated (GPU infra benchmarking, experiment design, honest measurement); Brev CLI/tmux/scp
mechanics; per-provider SSH quirks. See chat history / MENTOR_SUMMARY.md for the distilled versions.*
