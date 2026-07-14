# GPU Direct Storage & LLM Cold-Start Performance

Optimizing **LLM serverless cold starts and inference runtimes** using NVIDIA GPUDirect
Storage (GDS), CRIU container snapshots, and CUDA Checkpoint/Restore. The work delivers
benchmarks and implementation strategies aimed at faster model serving and improved
resource utilization — the time and cost sink behind serverless LLM deployments,
autoscaling, and scale-to-zero.

**Author:** Avaneesh Aroor

---

## TL;DR — headline finding

Across four GPU generations (H100, H200, B300, RTX PRO 6000) loading models from
1.5B to 123B parameters, **cold-start time is governed by storage speed, not the GPU.**

| 110B model, cold start | slow network disk (0.12 GB/s) | local NVMe (2.8 GB/s) | + loader/eager tuning |
|---|---|---|---|
| Time to serving-ready | **650 s** | **214 s** | **80 s** |

Two different GPUs on identical storage produced identical cold starts (within 1%);
one GPU on ~7× faster storage cut cold start ~3×. Once storage is fast, the remaining
cost is a storage-independent **engine-init/compile floor (~110–190 s)** — the target
for snapshot/restore techniques (CRIU / cuda-checkpoint). GDS was tested on a full **8-GPU
A100 DGX-class node** and *still* blocked (`use_pci_p2pdma: false`) — proving it is gated on
**virtualization, not node size**; only bare-metal will unlock it. Separately, the **InstantTensor
direct-I/O loader cut cold-cache weight loading up to ~9×** (110B: 72s → 8s) without GDS.
True GDS could not be
demonstrated on any *virtualized* cloud instance available (Experiment 03), which is
itself a reportable result: GDS requires bare-metal / PCIe-P2P-capable hardware.

---

## Experiments

| # | Experiment | Question | Headline result |
|---|------------|----------|-----------------|
| [01](./experiment-01-rowracer-cpu-vs-gpu/) | RowRacer — CPU vs GPU CSV benchmark | pandas (CPU) vs cuDF/RAPIDS (GPU) on 1M/10M rows | GPU 24× faster on 10M rows |
| [02](./experiment-02-llm-inference-bottleneck/) | LLM inference bottleneck | What limits token *generation* across GPU tiers? | Decode is **memory-bandwidth-bound** |
| [03](./experiment-03-coldstart-loading-gds/) | Cold-start loading & GDS | What limits getting a model *serving*, and can GDS fix it? | Cold start is **storage-bound**; GDS blocked on virtualized cloud HW |
| [04](./experiment-04-tp8-multiGPU-loaders-gds/) | TP8 multi-GPU, loader shoot-out, final GDS verdict | Does GDS work on a full 8-GPU DGX node? Does TP8 help cold start? Which loader wins? | GDS gated on **virtualization, not node size**; **TP8 hurts cold start**; **InstantTensor wins** |

The research arc: Experiment 02 established that decode throughput scales with memory
bandwidth, not compute. Experiment 03 pivots to the complementary question — what limits
*loading* — and finds it is storage-bound, directly motivating the GDS investigation.

---

## Research Roadmap (8 weeks)

- **Weeks 1–2:** Baselines — survey the NVIDIA GDS ecosystem; decompose cold-start and
  inference cost across models from 1.5B to 123B parameters. *(Experiments 02, 03)*
- **Weeks 3–5:** Evaluate streaming/GDS loaders across GPU architectures and PCIe
  generations; explore checkpoint/restore via CRIU and `cuda-checkpoint`.
- **Weeks 6–8:** Integrate with Dynamo snapshots and vLLM; explore accelerated data
  paths (cuML / cuDF). Deliver cost model and deployment recommendation.

See [experiment-03-coldstart-loading-gds/ROADMAP.md](./experiment-03-coldstart-loading-gds/ROADMAP.md)
for the detailed phase plan.

---

## Repository structure

```
.
├── experiment-01-rowracer-cpu-vs-gpu/       # CPU vs GPU throughput baseline (cuDF/RAPIDS)
├── experiment-02-llm-inference-bottleneck/  # vLLM inference: TTFT/TPOT/throughput, H100/H200/B200
│   ├── benchmark.py  Dockerfile  run_model.sh  SETUP_GUIDE.md
│   └── results/                             # H100 / H200 / B200 result write-ups
├── experiment-03-coldstart-loading-gds/     # cold-start decomposition + GDS investigation
│   ├── benchmark.py  run.sh  ROADMAP.md
│   ├── README.md                            # full methodology, results, and analysis
│   └── results/                             # per-GPU JSONL data + raw vLLM logs
└── experiment-04-tp8-multiGPU-loaders-gds/  # 8xA100 TP8, loader shoot-out, final GDS verdict
    ├── benchmark.py  run.sh  preflight_check.sh
    ├── README.md                            # findings + mistakes/solutions record
    ├── ANALYSIS.md                          # full tables
    └── results/                             # TP8 JSONL data + raw vLLM logs
```

## Stack

vLLM · FlashAttention 3 · PagedAttention · Docker + NVIDIA Container Toolkit ·
AWQ INT4 quantization · Brev / Shadeform / AWS GPU instances ·
NVIDIA Magnum IO GPUDirect Storage (`cuFile` / `nvidia_fs`)
