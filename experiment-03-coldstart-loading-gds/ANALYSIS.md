# Analysis — Cold-Start Loading & GDS

Full result tables for Experiment 03. Data in [`results/`](./results/); raw vLLM logs in
[`results/logs/`](./results/logs/). All times are `llm_init_s` (wall-clock to
engine-ready) unless noted.

## Instances

| GPU | VRAM | Backend | Storage | Measured disk BW |
|---|---|---|---|---|
| H100 | 80 GB | Brev / Hyperstack | virtio (network) | 0.12 GB/s |
| H200 | 141 GB | Brev / Hyperstack | virtio (network) | 0.12 GB/s |
| B300 | 275 GB | Shadeform | virtio (network) | 0.83 GB/s |
| RTX PRO 6000 | 96 GB | AWS DLAMI | NVMe-backed EBS / local NVMe | 1.55 / 2.8 GB/s |

## 1. Cold start (baseline, seconds) — the headline

| Model (disk) | H100 | H200 | B300 | RTX |
|---|---|---|---|---|
| 1B (3.1 GB) | 83.9 | 84.8 | 64.4 | 84.6 |
| 10B (15.2 GB) | 187.5 | 187.8 | 82.7 | 96.5 |
| 30B (19.3 GB) | 243.3 | 245.7 | 121.8 | 121.8 |
| **110B (64.9 GB)** | **650.0** | **649.6** | **225.1** | **212.8** |

H100 vs H200: identical within 1 % → GPU generation is irrelevant to cold start.
Faster storage (B300, RTX) → proportionally faster cold start.

## 2. Where the 110B cold start goes

| Component | H100 / H200 | B300 | How obtained |
|---|---|---|---|
| Disk read | ~537 s (83 %) | ~99 s (44 %) | cold − warm |
| Engine floor (H2D + init + compile + graphs) | ~113 s | ~126 s | warm |
| Disk share of cold start | 1B 38 % → 10B 71 % → 30B 68 % → 110B 83 % | | per size |

## 3. Storage utilization — is the loader wasting the disk?

| 110B | H100 / H200 | B300 |
|---|---|---|
| Raw disk ceiling (`disk_cold`) | 523 s @ 0.12 GB/s | 78.5 s @ 0.83 GB/s |
| vLLM cold weight load | ~539 s | ~99 s |
| Loader efficiency | ~97 % | ~80 % |

The default loader nearly saturates slow storage — a slow disk cannot be fixed in
software. End-to-end effective load rate (`effective_load_gbps`): 0.10 GB/s (Brev) vs
0.29 GB/s (B300) vs ~5 GB/s from page cache → the gap GDS + NVMe would need to close.

## 4. Warm start (RAM-cached) — the storage-independent floor

| Model | H100 | H200 | B300 | RTX |
|---|---|---|---|---|
| 1B | 51.6 | 53.0 | 53.6 | 81.7 |
| 10B | 55.0 | 55.8 | 54.6 | 83.8 |
| 30B | 78.7 | 80.0 | 90.4 | 106.9 |
| 110B | 113.6 | 112.4 | 126.2 | 163.6 |

~110–190 s floor for the 110B regardless of GPU or storage. This is `torch.compile` +
CUDA-graph capture + engine init — the target for snapshot/restore (CRIU, `cuda-checkpoint`).

## 5. `enforce_eager` trade-off (startup saved vs decode lost)

| Model | 110B cold: baseline → eager | Decode penalty (Hopper) |
|---|---|---|
| 1B | — | −60 % |
| 10B | — | −29 / −37 % |
| 30B | — | −35 / −37 % |
| 110B | H100 650 → 564 · B300 225 → 121 | **−3 / −2 %** |

Large models are memory-bandwidth-bound, so CUDA graphs barely help decode → eager is
near-free startup savings. For small models it is a bad trade.

## 6. Storage-tier ladder (110B, same protocol)

| Storage | Disk BW | Cold start | + eager+fst |
|---|---|---|---|
| Brev virtio | 0.12 GB/s | 650 s | 564 s |
| B300 virtio | 0.83 GB/s | 225 s | 121 s |
| RTX EBS | 1.55 GB/s | 213 s | 79 s |
| RTX local NVMe | 2.8 GB/s | 214 s | 80 s |

Cold start falls ~8× (650 → 80 s) from storage locality + loader/eager tuning alone —
no GDS required.

## 7. GDS investigation (RTX PRO 6000 with local NVMe)

Three configs on the NVMe box, all in `results/`:

| Config | File | Device passthrough | Log result |
|---|---|---|---|
| Baseline on NVMe | `RTX_NVME.jsonl` | — | n/a |
| `eager+fst` on NVMe | `RTX_NVMEGDS.jsonl` | none | `nogds=True` |
| GDS forced | `RTX_REALGDS.jsonl` | `--privileged` + `cufile.json` + all `/dev/nvidia-fs*` | `nogds=True` |

Load times were identical across all three (1B cold ~30 s, 110B cold ~80 s with eager),
proving device passthrough changed nothing. `gdscheck -p` reported `NVMe : compat` and
`use_pci_p2pdma : false`: the AWS Nitro hypervisor does not expose PCIe peer-to-peer DMA
to the guest, so cuFile disables GDS rather than run a pointless CPU-bounce compat path.

## 8. InstantTensor loader vs default loader (H100, fast virtio ~2 GB/s, no GDS)

A controlled ablation with **`enforce_eager` held constant in both configs** — only the
loader changes — so the difference isolates the loader alone. Data: `results/H100_2_eager_ablation.jsonl`;
InstantTensor 0.1.9 via `--load-format instanttensor` (a derived image = NGC vLLM +
`pip install instanttensor`, since the base image lacks the package).

**Cold-cache weight-load time** (from vLLM's `Loading weights took` log line — the loader's
actual job, page cache dropped first):

| Model | Default loader | InstantTensor | Speedup |
|---|---|---|---|
| 1B (3.1 GB) | 1.45 s | 1.52 s | 0.95× (fixed overhead dominates) |
| 10B (15.2 GB) | 17.98 s | 2.44 s | **7.4×** |
| 30B (19.3 GB) | 14.11 s | 3.81 s | **3.7×** |
| 110B (64.9 GB) | 71.86 s | **8.21 s** | **8.8×** |

**End-to-end cold start** (eager, no GDS):

| Model | Default | InstantTensor | Speedup |
|---|---|---|---|
| 10B | 41.2 s | 26.3 s | 1.6× |
| 110B | **105.9 s** | **41.8 s** | **2.5×** |

**Why it wins:** the default loader reads *through the OS page cache* — on a cold cache it
is disk-bound (~0.9 GB/s effective, 72 s for the 110B). InstantTensor uses **direct I/O +
pipelined prefetching + concurrency**, reading the same 65 GB in 8.2 s (~8 GB/s). Two
consequences:

1. **It extracts far more from the same disk.** Single-stream `dd` measured ~2 GB/s on this
   volume; InstantTensor's concurrent reads hit ~8 GB/s — so a single-stream disk benchmark
   *underestimates* what a parallel loader can achieve.
2. **Its load time is cache-independent** — cold ≈ warm (110B: 8.2 s cold vs 8.3 s warm),
   because O_DIRECT bypasses the page cache. Its *cold* load (8.2 s) is even faster than the
   default loader reading from *warm RAM* (10.8 s). For serverless cold starts (empty cache
   by definition) this is the ideal property.

Caveat: the win grows with model size; on the 1B it is a slight *loss* (InstantTensor's
distributed/prefetch machinery has fixed setup overhead not amortized by tiny weights).
This ran **without GDS** (box had none) — InstantTensor's direct-I/O path alone; with GDS
on bare-metal it could go further.

## Key takeaways

1. **Cold start is storage-bound** — identical GPUs+storage → identical cold starts; ~7× faster storage → ~3× faster cold start.
2. **The loader is not the problem on slow storage** (~97 % disk-saturated). GDS/NVMe attacks exactly this.
3. **~5 GB/s from page cache** shows the disk→VRAM path's ceiling: a 110B in ~13 s if storage keeps up.
4. **~113 s warm floor** (compile + graphs + init) is storage-independent → snapshot/restore is the complement to GDS.
5. **`enforce_eager` is a free lunch for large models** (−65 s startup, −3 % decode), a bad deal for small ones.
6. **GDS needs bare-metal / PCIe-P2P hardware** — a local NVMe is necessary but not sufficient; virtualized cloud instances disable it.
7. **InstantTensor's direct-I/O loader cuts cold-cache weight loading up to ~9×** (110B: 72 s → 8 s) and the end-to-end cold start ~2.5×, *without* GDS — by using concurrency + O_DIRECT to bypass the page cache and saturate the disk. This is the largest single lever found for cold-start loading on standard (non-bare-metal) cloud instances.

## 9. Three-loader comparison, and what GDS would add

All three loaders measured on the same task — **cold-cache weight load of the 110B (64.9 GB)**,
taken from vLLM's `Loading weights took` log line:

| Loader | Storage | Load time | Effective bandwidth |
|---|---|---|---|
| default (`safetensors`) | virtio 0.12 GB/s | ~524 s | 0.12 GB/s |
| `fastsafetensors` (`nogds`) | virtio 0.12 GB/s | 524 s | 0.12 GB/s |
| default (`safetensors`) | virtio 2.0 GB/s | 71.9 s | 0.9 GB/s |
| `fastsafetensors` (`nogds`) | local NVMe 2.8 GB/s | 32.7 s | 2.0 GB/s |
| **`instanttensor`** | virtio 2.0 GB/s | **8.2 s** | **7.9 GB/s** |

**The decisive result:** InstantTensor on *slower* storage (2.0 GB/s) loaded the 110B in
**8.2 s**, while `fastsafetensors` on *faster* storage (2.8 GB/s local NVMe) took **32.7 s** —
**~4× faster despite a storage disadvantage.**

The reason is *concurrency*, not raw disk speed. `fastsafetensors` (in `nogds` mode) reads
roughly single-stream and merely tracks the disk's sequential rate (2.8 GB/s disk → 2.0 GB/s
effective). InstantTensor's pipelined, concurrent direct I/O extracts **~8 GB/s from a volume
that single-stream `dd` clocked at only 2 GB/s** — i.e. a single-stream disk benchmark
*understates* what a parallel loader can achieve. This also explains why the earlier
`fastsafetensors` runs never impressed: they were pinned to the single-stream disk rate.

### Estimated impact of adding GDS

GDS changes the *path* (`NVMe → CPU RAM → GPU` becomes `NVMe → GPU` direct DMA); it does not
change the *read strategy*. InstantTensor already does direct I/O with concurrency, so GDS
would only remove the remaining CPU bounce-buffer copy on top of that.

| 110B, bare-metal fast NVMe | Est. weight load | Est. end-to-end cold start |
|---|---|---|
| InstantTensor (direct I/O, no GDS) — *measured* | 8.2 s | 41.8 s |
| InstantTensor **+ GDS** — *estimated* | ~3–5 s | ~37–39 s |

**GDS's incremental benefit is projected to be small (~10 % of cold start) for single-stream
loading**, for two reasons:

1. InstantTensor already reads at ~8 GB/s. GDS mainly pays off once the **CPU bounce buffer**
   becomes the bottleneck — which requires storage faster than InstantTensor is already extracting.
2. Once the load falls to single-digit seconds, the **~33 s engine-init/compile floor dominates**
   the cold start, and GDS cannot touch it. Halving an 8 s load only moves cold start ~42 s → ~38 s.

**Where GDS *would* clearly matter** (outside this experiment's single-stream scope):
**many concurrent model loads on one node**, where CPU-bounce-buffer and RAM bandwidth become
a *shared* bottleneck. GDS's CPU bypass relieves that contention, so at fleet scale it could be
a genuine multiplier. For a single model loading once, it is marginal on top of InstantTensor.

**Conclusion:** for single-instance cold starts, a `pip install` + one flag (InstantTensor)
delivered ~9× faster loading with no special hardware — more than GDS is projected to add on
top of it. The larger remaining lever is not storage at all, but the **engine-init/compile
floor** → CRIU / `cuda-checkpoint` (Phase 4).

## Caveats (honest reporting)

- The `eager+fst` config ran cuFile with **GDS inactive** (`nogds=True`) on every box — it
  measures the fastsafetensors loader + `enforce_eager`, not GPU Direct Storage.
- Baseline used custom image vLLM 0.23.0; `eager+fst` used NGC vLLM 0.22.1 — a small version skew.
- B300 stands in for the planned B200 (unavailable); it also had faster storage, so its
  GPU-vs-storage effects are disentangled via the `disk_cold` rows, not GPU identity.
- 110B on the 80 GB H100 leaves only ~3.5 GiB KV cache — fine for single-request benchmarking, not concurrency.
- Reproducibility: repeated 1B baselines matched within ~2 % (H100, B300).
- **The GDS figures in §9 are an ESTIMATE, not a measurement** — GDS never engaged on any
  instance obtained (see §7). They are projected from the measured InstantTensor direct-I/O
  numbers plus the known cost GDS removes (the CPU bounce-buffer copy). Treat as a
  hypothesis to be tested on bare-metal hardware, not as a result.
- The three-loader comparison in §9 spans different instances/storage (noted per row); the
  InstantTensor-vs-default rows are same-box and fully controlled, while the
  InstantTensor-vs-fastsafetensors comparison is cross-box (InstantTensor was on *slower*
  storage, which strengthens rather than weakens its result).
