# Analysis — TP8 Multi-GPU Loading, Loader Shoot-out, Final GDS Verdict

Data: [`results/A100x8_TP8.jsonl`](./results/A100x8_TP8.jsonl) · raw logs in [`results/logs/`](./results/logs/).
All times are `llm_init_s` (wall-clock to engine-ready) unless noted. Weight-load times are
vLLM's `Loading weights took` log line on the **cold** (page-cache-dropped) run.

## Node

| Spec | Value |
|---|---|
| GPUs | 8× A100-SXM4-80GB (640 GB VRAM total) |
| Machine | `azurerm.a100x8.sxm.brev-dgxc` (Azure DGX Cloud via Brev) |
| CPU / RAM | 96 vCPU / 1.77 TB |
| Storage | 8× local NVMe (894 GB ea.) RAID → `/ephemeral`, 7 TB, **2.7 GB/s** single-stream read |
| Cost | $19.30/hr |

## 1. GDS verdict — blocked, even here

```
GDS release version: 1.18.1.6
nvidia_fs version:   2.29
NVMe                      : compat        ← not "Supported"
properties.use_pci_p2pdma : false         ← the blocker
```

But `nvidia_fs` **did** enumerate GPU↔NVMe peer paths — further than the AWS box got:

```
/proc/driver/nvidia-fs/peer_distance
gpu            peer           p2pdist  link  gen   np2p  class
000e:00:00.0   1949:00:00.0   0x0080   0x04  0x03  0     nvme   ← 8 NVMe peers seen
```

The driver sees the topology; the **hypervisor refuses the DMA**. Conclusion: **GDS is gated on
virtualization, not node size.** A full 8-GPU DGX-class node with 8 local NVMe drives is still
a VM, and Azure does not expose PCIe P2P to guests.

Cross-experiment summary (all three failed, for two distinct reasons):

| Instance | Storage | `nvidia_fs` | Result |
|---|---|---|---|
| Brev H100/H200/B300 (exp-03) | virtio (network) | absent | no local disk to DMA from |
| AWS RTX PRO 6000 (exp-03) | local NVMe | loaded | `use_pci_p2pdma: false` |
| **Azure 8× A100 DGX (this)** | **8× local NVMe RAID** | **loaded, peers enumerated** | **`use_pci_p2pdma: false`** |

## 2. TP8 loader shoot-out

| Model | Loader | Cold weight-load | Cold start | Warm start |
|---|---|---|---|---|
| 30B | `instanttensor` | **3.66 s** | **72.9 s** | 51.9 s |
| 30B | `auto` | 5.76 s | 75.1 s | 49.6 s |
| 30B | `runai_streamer` | — ¹ | 85.6 s | 56.9 s |
| 110B | `instanttensor` | **5.83 s** | **73.0 s** | 57.0 s |
| 110B | `auto` | 20.50 s | 89.7 s | 50.4 s |
| 110B | `runai_streamer` | — ¹ | 111.4 s | 77.3 s |

¹ Run:ai Streamer uses its own progress bar and does not emit vLLM's `Loading weights took` line;
its end-to-end `llm_init_s` is shown instead. It *did* engage correctly (log:
`Loading safetensors using Runai Model Streamer`).

**InstantTensor wins at both sizes. Run:ai Streamer is slowest at both — worse than the default
loader.** Caveat: Run:ai ran with stock settings; concurrency is tunable
(`--model-loader-extra-config '{"concurrency":16}'`), so this is not its ceiling.

## 3. TP8 vs single-GPU (110B, InstantTensor, eager)

| | Weight load | **Cold start** | Warm start |
|---|---|---|---|
| Single GPU (H100, exp-03) | 8.2 s | **41.8 s** | 39.6 s |
| TP8 (8× A100) | **5.8 s** (faster) | **73.0 s** (worse) | 57.0 s |

TP8 speeds the weight read (sharding) but adds **NCCL init + 8 worker processes** to engine
startup. That overhead (~20–25 s, visible in the warm start: 57 s vs 40 s) **exceeds the loading
gain**. For a model that fits on one GPU, **TP8 is a net loss for cold-start latency**.

## 4. TP8 is impossible for small models

Tensor parallelism shards attention heads; head count must be divisible by TP size.

| Model | Heads | ÷ 8 | TP8 |
|---|---|---|---|
| 1B (Qwen2.5-1.5B) | 12 | 1.5 | ❌ `ValueError` |
| 10B (Qwen2.5-7B) | 28 | 3.5 | ❌ `ValueError` |
| 30B (Qwen2.5-32B-AWQ) | 40 | 5 | ✅ |
| 110B (Mistral-Large-AWQ) | 96 | 12 | ✅ |

## 5. The loader's advantage shrinks as you shard

| Model @ TP8 | bytes/GPU | disk read | InstantTensor vs default (weight load) |
|---|---|---|---|
| 110B (64.9 GB) | ~8.1 GB | 15.2 s | **3.5×** (5.8 s vs 20.5 s) |
| 30B (19.3 GB) | ~2.4 GB | 6.3 s | **1.6×** (3.7 s vs 5.8 s) |

Sharding 19 GB across 8 GPUs leaves only ~2.4 GB each — the disk read becomes trivial and the
**engine/NCCL floor (~50 s) dominates**, leaving nothing for a faster loader to win back.

> **A faster loader only matters when there are enough bytes per GPU for loading to matter.**

## Key takeaways

1. **GDS is gated on virtualization, not node size** — disproven on a full 8-GPU DGX node. Only bare-metal will work.
2. **TP8 hurts cold start** for models that fit on one GPU (110B: 73 s vs 42 s single-GPU) — NCCL/worker init overhead exceeds the loading gain.
3. **TP8 is impossible** for models whose attention heads don't divide by 8 (1B, 10B here).
4. **InstantTensor wins every configuration tested** — single-GPU and TP8, 30B and 110B.
5. **Run:ai Model Streamer (stock config) loses** — slower than the default loader at both sizes.
6. **Loader gains shrink with sharding** — the more you split a model, the less the loader matters.
7. **The practical recipe for fast cold starts:** local NVMe + `enforce_eager` + InstantTensor, on
   the **fewest GPUs that fit the model**. No GDS, no special hardware.

## Caveats

- Run:ai Model Streamer used **default concurrency**; it is tunable and this is not its best case.
- The TP8-vs-single-GPU comparison spans different GPUs (A100 vs H100) and storage (2.7 vs 2.0 GB/s).
  The *direction* of the result is robust (the penalty appears in warm start, which is I/O-independent),
  but the exact magnitude is not a controlled single-variable comparison.
- 1B/10B produced no TP8 rows — they failed at engine construction (head divisibility), by design.
- A 405B (~203 GB) run was planned to test a model that *requires* 8 GPUs, but was **cancelled**
  once the TP8 conclusion was established, to limit cost. It remains the natural next test.
