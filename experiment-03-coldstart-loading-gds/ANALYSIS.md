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

## Key takeaways

1. **Cold start is storage-bound** — identical GPUs+storage → identical cold starts; ~7× faster storage → ~3× faster cold start.
2. **The loader is not the problem on slow storage** (~97 % disk-saturated). GDS/NVMe attacks exactly this.
3. **~5 GB/s from page cache** shows the disk→VRAM path's ceiling: a 110B in ~13 s if storage keeps up.
4. **~113 s warm floor** (compile + graphs + init) is storage-independent → snapshot/restore is the complement to GDS.
5. **`enforce_eager` is a free lunch for large models** (−65 s startup, −3 % decode), a bad deal for small ones.
6. **GDS needs bare-metal / PCIe-P2P hardware** — a local NVMe is necessary but not sufficient; virtualized cloud instances disable it.

## Caveats (honest reporting)

- The `eager+fst` config ran cuFile with **GDS inactive** (`nogds=True`) on every box — it
  measures the fastsafetensors loader + `enforce_eager`, not GPU Direct Storage.
- Baseline used custom image vLLM 0.23.0; `eager+fst` used NGC vLLM 0.22.1 — a small version skew.
- B300 stands in for the planned B200 (unavailable); it also had faster storage, so its
  GPU-vs-storage effects are disentangled via the `disk_cold` rows, not GPU identity.
- 110B on the 80 GB H100 leaves only ~3.5 GiB KV cache — fine for single-request benchmarking, not concurrency.
- Reproducibility: repeated 1B baselines matched within ~2 % (H100, B300).
