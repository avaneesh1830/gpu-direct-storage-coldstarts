# Cold-Start Reduction for Serverless GPUs — Status, Findings, and Proposed Pivot

**Author:** Avaneesh Aroor · **Repo:** github.com/avaneesh1830/gpu-direct-storage-coldstarts

---

## TL;DR

1. **GDS cannot be tested from any rented cloud GPU** — I proved this across three clouds,
   including the full 8× A100 DGX-class node you suggested. All report
   `use_pci_p2pdma: false` because they are VMs; hypervisors do not expose PCIe P2P DMA to
   guests. This is not a configuration problem — it is architectural.
2. **More importantly: GDS would not help much even if it worked.** Once a concurrent loader
   is used, weight loading drops to ~8 s and is no longer the bottleneck. A **~33–50 s
   engine-init/compile floor** dominates the cold start, and **GDS cannot touch it.**
3. **I found a working alternative that needs no special hardware:** the **InstantTensor**
   loader cuts cold-cache weight loading **~9×** (72 s → 8 s on a 123B model).
4. **Proposed pivot:** attack the engine-init floor with **CRIU + `cuda-checkpoint`**
   (snapshot/restore). NVIDIA shipped exactly this as **Dynamo Snapshot v1.2** in June 2026,
   reporting **restores in 2.25–9 s**. That is the real serverless lever — and it works on
   ordinary VMs, unlike GDS.

---

## 1. What was tested, and what GDS did

| Instance | Storage | `nvidia_fs` | GDS result |
|---|---|---|---|
| Brev H100 / H200 / B300 | virtio (network) | absent | No local disk to DMA from |
| AWS RTX PRO 6000 | local NVMe | loaded | `use_pci_p2pdma: false` |
| **Azure 8× A100 SXM DGX** (your suggestion) | **8× local NVMe RAID** | **loaded, GPU↔NVMe peers enumerated** | **`use_pci_p2pdma: false`** |

Your hypothesis — *"they might be holding back GDS for a full node worth of machine"* — was
worth testing, and I tested it directly on an 8-GPU DGX-class node. It is **disproven**:
GDS is gated on **virtualization, not node size**. Even with `nvidia_fs` loaded and the driver
successfully enumerating GPU↔NVMe PCIe peer paths, the hypervisor still denies the DMA.

**To run true GDS I need a bare-metal machine** (DGX, OCI bare-metal, or AWS `.metal`).
*Do you have access to one?* If so, the benchmark runs unchanged — the loaders auto-detect GDS.

## 2. The more important finding: GDS optimizes the wrong thing

Decomposition of a 123B cold start (best configuration measured — local NVMe, `enforce_eager`,
InstantTensor):

| Component | Time | Can GDS help? |
|---|---|---|
| Weight load | **8 s** | Maybe → 3–5 s |
| **Engine init (torch.compile + CUDA graphs + NCCL)** | **~33 s** | ❌ **No** |
| **Total cold start** | **41.8 s** | ~10 % at best |

Even in the perfect case — bare metal, GDS fully enabled — the upside is a few seconds off an
8-second load, while a 33-second engine-init floor sits untouched. **GDS optimizes the part
that is already small.**

This is the central result: *GDS is the wrong tool for serverless cold starts.*

## 3. What I found that *does* work (no special hardware)

| Lever | Effect (123B) | Requires |
|---|---|---|
| Storage locality (local NVMe vs network disk) | 650 s → 214 s cold start | choosing the right instance |
| `enforce_eager` (skip CUDA-graph capture) | −65 s startup, only −3 % decode on large models | one flag |
| **InstantTensor loader** | **weight load 72 s → 8 s (~9×)** | `pip install` + one flag |
| *(combined)* | **650 s → 42 s** | no GDS, no special hardware |

**InstantTensor beat `fastsafetensors` ~4× while running on *slower* storage** (8.2 s vs 32.7 s).
Reason: `fastsafetensors` reads roughly single-stream and is pinned to the disk's sequential
rate; InstantTensor's concurrent direct I/O extracts **~8 GB/s from a volume that single-stream
`dd` measured at only 2 GB/s.** It is also cache-independent (O_DIRECT → cold ≈ warm), which is
exactly the property serverless needs, since a cold start has an empty page cache by definition.

**On your Run:ai suggestion:** your fallback instinct — *"if GDS isn't panning out, try a
software loader"* — was **correct**, and it is the reason this project has a positive result at
all. I benchmarked Run:ai Model Streamer head-to-head; with stock settings it came **last**
(slower than even the default loader), while InstantTensor won. *Caveat: Run:ai's concurrency is
tunable and I ran defaults, so that is not necessarily its ceiling.*

## 4. Multi-GPU (TP8) — tested, and it does not help cold starts

| 123B | Weight load | Cold start |
|---|---|---|
| Single GPU | 8.2 s | **41.8 s** |
| TP8 (8× A100) | 5.8 s (faster) | **73.0 s** (worse) |

Sharding speeds the weight read but adds NCCL + 8 worker processes to engine init — that
overhead exceeds the loading gain. **TP8 is for models too large for one GPU, not for cold-start
latency.** (Also: TP8 is *impossible* for small models — attention heads must be divisible by
the TP size; 1B has 12 heads and 10B has 28, neither divisible by 8.)

## 5. Proposed pivot: snapshot/restore (CRIU + `cuda-checkpoint`)

The remaining cold-start time is the **engine-init floor**, which no loader and no storage
technology can remove. The only way to remove it is to **not do it** — snapshot a fully warmed,
ready-to-serve process and restore it.

This is not speculative. **NVIDIA shipped Dynamo Snapshot v1.2 in June 2026** — CRIU +
`cuda-checkpoint` for vLLM workers on Kubernetes — reporting **restores in 2.25 s (from S3) and
9 s (from local NVMe)**, versus my measured best cold start of 41.8 s. Notably, NVIDIA solved
serverless cold starts with **snapshotting, not GDS.**

Crucially: **CRIU/`cuda-checkpoint` work on ordinary virtualized instances** — unlike GDS, this
pivot is not blocked by the hardware I can actually rent.

**Target:** cold start **41.8 s → single-digit seconds**, i.e. a further ~5–10× on top of the
loader win, and ~65× versus the original naive baseline (650 s).

See `PHASE4_PLAN.md` for the concrete experiment plan.

---

## Summary of the ask

1. **Do you have access to a bare-metal GPU box?** It is the only way to close out the GDS
   question with a positive measurement rather than an architectural argument.
2. **Approval to pivot to Phase 4 (CRIU + `cuda-checkpoint` / Dynamo Snapshot).** The data says
   this is where the remaining cold-start time actually is, and it is unblocked on the hardware
   I have.
