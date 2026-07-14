# Experiment 04 — Multi-GPU (TP8) Loading, Loader Shoot-out, and the Final GDS Verdict

**Questions:**
1. Does GPU Direct Storage work on a **full 8-GPU DGX-class node**? (Hypothesis: cloud providers
   may only enable GDS on full-node machines, not single-GPU slices.)
2. Does **tensor parallelism across 8 GPUs (TP8)** improve cold-start loading?
3. Which weight loader actually wins: **default safetensors vs InstantTensor vs Run:ai Model Streamer**?

**Hardware:** `azurerm.a100x8.sxm.brev-dgxc` — 8× A100-SXM4-80GB (640 GB VRAM), 96 vCPU,
1.77 TB RAM, 8× local NVMe ("Microsoft NVMe Direct Disk", 894 GB each) striped as RAID at
`/ephemeral` (7 TB, 2.7 GB/s single-stream). $19.30/hr.

---

## Headline results

### 1. GDS is blocked even on a full 8-GPU DGX node — the hypothesis is **disproven**

```
$ gdscheck -p
 NVMe                      : compat
 properties.use_pci_p2pdma : false
```

This node had *everything* the theory required: 8 GPUs, SXM/NVLink, DGX Cloud, 8 local NVMe
drives. GDS still refused. Notably `nvidia_fs` **did** enumerate GPU↔NVMe PCIe peer paths
(`/proc/driver/nvidia-fs/peer_distance` lists `class=nvme` peers with real link/gen values) —
further than the AWS box ever got — but the hypervisor still denies P2P DMA.

> **GDS is gated on *virtualization*, not node size.** This is still a VM (`azurerm`, "VM Mode").
> Azure's hypervisor does not expose PCIe peer-to-peer DMA to a guest, regardless of how many
> GPUs that guest has. Chasing bigger nodes on virtualized clouds will not unlock GDS —
> only genuine bare-metal will.

Three clouds, three refusals, all for the same reason (see [experiment-03](../experiment-03-coldstart-loading-gds/)
for Brev virtio and AWS local-NVMe results).

### 2. TP8 makes loading faster but cold start **slower**

| 110B | Weight load | **Cold start** | Warm start |
|---|---|---|---|
| Single GPU (H100, InstantTensor) | 8.2 s | **41.8 s** | 39.6 s |
| **TP8** (8× A100, InstantTensor) | **5.8 s** | **73.0 s** ← *worse* | 57.0 s |

Sharding across 8 GPUs *does* speed up the weight read — but it adds NCCL setup and 8 worker
processes to engine init, and that overhead (~20–25 s) **exceeds the loading gain**. The warm
starts confirm it (57 s vs 40 s): the penalty is in engine init, not I/O.

> **For a model that fits on one GPU, TP8 hurts cold-start latency.** TP8 is for models too
> large for a single card, or for throughput — not for cold starts.

### 3. TP8 is **impossible** for small models (hard architectural constraint)

```
ValueError: Total number of attention heads (12) must be divisible by tensor parallel size (8)
```

Tensor parallelism splits attention heads across GPUs, so head count must be divisible by TP size:

| Model | Attention heads | ÷ 8 | TP8 possible? |
|---|---|---|---|
| 1B (Qwen2.5-1.5B) | 12 | 1.5 | ❌ impossible |
| 10B (Qwen2.5-7B) | 28 | 3.5 | ❌ impossible |
| 30B (Qwen2.5-32B-AWQ) | 40 | 5 | ✅ |
| 110B (Mistral-Large-AWQ) | 96 | 12 | ✅ |

You *cannot* use all 8 cards for these small models at any setting.

### 4. Loader shoot-out at TP8 — InstantTensor wins, Run:ai loses

| Model | Loader | Cold weight-load | Cold start | Warm start |
|---|---|---|---|---|
| **30B** | `instanttensor` | **3.66 s** | **72.9 s** | 51.9 s |
| 30B | `auto` (safetensors) | 5.76 s | 75.1 s | 49.6 s |
| 30B | `runai_streamer` | — | **85.6 s** ← slowest | 56.9 s |
| **110B** | `instanttensor` | **5.83 s** | **73.0 s** | 57.0 s |
| 110B | `auto` (safetensors) | 20.50 s | 89.7 s | 50.4 s |
| 110B | `runai_streamer` | — | **111.4 s** ← slowest | 77.3 s |

- **InstantTensor wins at both sizes**, consistent with the single-GPU findings in experiment-03.
- **Run:ai Model Streamer lost consistently** — slower than even the *default* loader, at both
  sizes. It did engage correctly (`Loading safetensors using Runai Model Streamer` in the logs).
  *Caveat: we ran its stock settings; its concurrency is tunable via
  `--model-loader-extra-config '{"concurrency":16}'`, so this is not its ceiling.*

### 5. The loader's advantage **shrinks as you shard**

InstantTensor's win over the default loader collapses from **3.5×** (110B: 5.8 s vs 20.5 s)
to **1.6×** (30B: 3.7 s vs 5.8 s) at TP8.

Why: sharding the 30B's 19 GB across 8 GPUs leaves only ~2.4 GB per GPU. The disk read becomes
trivial (6.3 s), so there is almost nothing left for a faster loader to win back — the
**engine/NCCL init floor (~50 s) dominates**.

> **A faster loader only matters when there are enough bytes per GPU for loading to matter.**
> Shard thin enough and the loader stops mattering entirely.

---

## Why InstantTensor beats the previous approaches

| Approach | Mechanism | 110B cold weight-load | Needs special HW? |
|---|---|---|---|
| default `safetensors` | sequential read *through page cache* | 71.9 s (single-GPU) | no |
| `fastsafetensors` | single-stream, GDS-capable but `nogds` | 32.7 s (on *faster* NVMe) | GDS needs bare-metal |
| `runai_streamer` | concurrent threads → CPU buffer (stock config) | slowest of all | no |
| **`instanttensor`** | **direct I/O (O_DIRECT) + pipelined prefetch + concurrency** | **8.2 s** | **no** |

1. **It beats `fastsafetensors` ~4× while running on *slower* storage** (8.2 s on a 2.0 GB/s
   volume vs 32.7 s on a 2.8 GB/s NVMe). `fastsafetensors` reads roughly single-stream and is
   pinned to the disk's sequential rate; InstantTensor's concurrency extracts **~8 GB/s from a
   disk that single-stream `dd` clocked at only 2 GB/s.** A single-stream disk benchmark
   *understates* what a parallel loader can achieve.
2. **It is cache-independent.** O_DIRECT bypasses the page cache, so cold ≈ warm (110B: 8.2 s
   cold vs 8.3 s warm). Its *cold* load is even faster than the default loader reading from
   *warm RAM*. For serverless cold starts — where the cache is empty by definition — this is
   the ideal property.
3. **It needs no special hardware.** This is the decisive advantage over GDS, which is blocked
   on every virtualized cloud we tested. InstantTensor is a `pip install` plus one flag.

---

## Mistakes made, and how they were solved

Documented honestly — several cost real time and money.

### 1. `apt install nvidia-gds` broke the host NVIDIA driver
**What happened:** installing the GDS package pulled in an entire new driver
(`nvidia-595-server`: firmware, kernel modules) but *not* its userspace libraries. This
destroyed the working driver install — `nvidia-smi` died ("couldn't find libnvidia-ml.so") and
**all Docker GPU containers failed**.

**Cascade of errors:** the NVIDIA container runtime then demanded files that didn't exist —
`/run/nvidia-persistenced/socket`, `/usr/bin/nvidia-cuda-mps-control`, then
`libcuda.so.595.71.05`.

**Solution:** install the *matching* driver userspace to make the system consistent again:
```bash
sudo apt-get install -y libnvidia-compute-595-server nvidia-utils-595-server \
                        libnvidia-cfg1-595-server libnvidia-encode-595-server \
                        libnvidia-decode-595-server
# plus stub the files the runtime bind-mounts but that don't exist:
sudo mkdir -p /run/nvidia-persistenced && sudo touch /run/nvidia-persistenced/socket
sudo touch /usr/bin/nvidia-cuda-mps-control /usr/bin/nvidia-cuda-mps-server
```
This restored `nvidia-smi`, Docker GPU access, **and** put `libcuda.so.1` on the host — which
was itself required for the next problem.

**Lesson:** on a managed GPU image (DGX Cloud / Brev), `apt install nvidia-gds` is *not* safe.
It can silently replace the driver. Snapshot or expect to repair.

### 2. `gdscheck` could not run — `libcuda.so.1` absent on the host
**What happened:** on this DGX Cloud image the CUDA userspace lives *inside containers*, not on
the host, so `gdscheck` failed with `error while loading shared libraries: libcuda.so.1`.

**Solution:** fixed as a side effect of (1) — installing the driver userspace put `libcuda` on
the host. (Alternative: run `gdscheck` inside a CUDA container with the host's
`/usr/local/cuda/gds` bind-mounted.)

**Workaround worth knowing:** even without `gdscheck`, `/proc/driver/nvidia-fs/peer_distance`
and `/proc/driver/nvidia-fs/stats` expose the GPU↔peer topology and GDS version directly.

### 3. `--load-format instanttensor` rejected by our own harness
**What happened:** `benchmark.py`'s argparse `choices=[...]` didn't include `instanttensor`, so
every run died instantly with `invalid choice: 'instanttensor'` — producing zero rows.

**Solution:** added it to the allowed choices (see `benchmark.py`).

### 4. InstantTensor is a separate pip package, absent from the NGC image
**What happened:** vLLM has the *integration hooks* for InstantTensor, but the package itself
isn't bundled: `ModuleNotFoundError: No module named 'instanttensor'`.

**Solution:** built a derived image once and reused it:
```dockerfile
FROM nvcr.io/nvidia/vllm:26.06-py3
RUN pip install --no-cache-dir instanttensor runai-model-streamer
```
(tagged `vllm-loaders` — also provides Run:ai Streamer for the shoot-out.)

### 5. Earlier assumption error: "InstantTensor" ≠ `fastsafetensors`
In experiment-03 the term "InstantTensor" was initially assumed to mean `fastsafetensors`.
**It is a real, distinct loader** (`scitix/InstantTensor`) with its own vLLM load format.
Correcting this changed the conclusion materially — InstantTensor turned out ~4× faster than
fastsafetensors. **Verify a tool exists before benchmarking a substitute for it.**

### 6. No public SSH — only port 8888 exposed
The provider exposed only Jupyter (8888); port 22 was unreachable and the panel said
*"This cloud provider doesn't allow the modifications of ports."*

**Solution:** use the Brev CLI's SSH tunnel — `brev ls` to list, then `ssh h100-2` (the alias
Brev writes into `~/.brev/ssh_config`), **not** the raw IP.

### 7. Cost discipline
At **$19.30/hr**, mistakes are expensive. The `preflight_check.sh` script (included here) exists
precisely for this: it answers "does this box have GDS / fast storage / enough RAM?" in **under
a minute**, *before* committing to a multi-hour benchmark. It correctly predicted the GDS
failure on two separate machines. A planned 405B run was cancelled before it downloaded 203 GB
once the TP8 conclusion was already established.

---

## Running it

```bash
# 0. ALWAYS run preflight first — it answers the GDS/storage question in <1 min
scp preflight_check.sh <host>:~/ && ssh <host> './preflight_check.sh'

# 1. build the loaders image (InstantTensor + Run:ai are not in the base image)
docker build -t vllm-loaders - <<EOF
FROM nvcr.io/nvidia/vllm:26.06-py3
RUN pip install --no-cache-dir instanttensor runai-model-streamer
EOF

# 2. TP8 loader shoot-out (only for models whose attention heads divide by 8)
for LF in auto instanttensor runai_streamer; do
  CACHE=/ephemeral/mc PREFIX=tp8_${LF}_ \
  EXTRA="--enforce-eager --tp-size 8 --load-format $LF" \
  IMAGE=vllm-loaders SIZE=110b ./run.sh 2>&1 | tee ~/results/log_110b_tp8_${LF}.txt
done
```

## Files

```
benchmark.py        # now supports --tp-size N and --load-format instanttensor
run.sh              # 4-step cold/warm/disk protocol (SIZE/IMAGE/CACHE/PREFIX/EXTRA env vars)
preflight_check.sh  # GPU / RAM / storage / GDS-readiness check — RUN THIS FIRST
ANALYSIS.md         # full tables
results/
  A100x8_TP8.jsonl  # all runs (30B + 110B at TP8, 3 loaders)
  logs/             # raw vLLM logs
```
