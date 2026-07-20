# Experiment 05 — CRIU Checkpoint/Restore (CPU → GPU)

**Phase 4 of the cold-start project:** attack the storage-independent **engine-init
floor** (the ~110–190 s that remains after storage and loaders are optimized in
experiments 03–04) by *snapshotting an already-initialized process and restoring it*
instead of re-initializing. This is the mechanism behind NVIDIA Dynamo Snapshot and
Doubleword Cloudburst.

**Author:** Avaneesh Aroor

---

## TL;DR — what this experiment establishes

Both stages **passed**. CRIU checkpoint/restore was proven end-to-end, from a toy CPU
counter up to a 7B LLM resident on an H100 GPU — including the GPU's VRAM and device
state, via NVIDIA's `cuda-checkpoint`.

| Stage | Workload | Result | Image | Dump | Restore (cold) |
|---|---|---|---|---|---|
| 1  | CPU counter | resumed at N+1 after kill | 4 MB | 0.02 s | 0.08 s |
| 1b | 0.5B model in RAM (CPU) | resumed; inference intact | 2.2 GB | 12.8 s | 18.6 s |
| 2  | **7B model on H100 GPU** | **VRAM 15.3 GB → 0 → 15.3 GB; inference intact** | 17 GB | 15.4 s | 143 s |

**The mechanism works.** A process — including 15 GB of GPU weights — was frozen to
disk, killed, and restored to the exact instruction it died on, with weights
byte-correct (the model still answered "Paris").

## The key finding (why this doesn't *yet* beat a cold start)

Restore is a **sequential read of a memory-snapshot image**, so it is **disk-bound**:
in every stage, `image_size ÷ restore_time` equalled the measured disk speed
(~0.12 GB/s virtio on these boxes). That yields a precise decision rule:

> **snapshot restore beats a cold start  ⟺  (engine-init time skipped) > (image size ÷ disk speed)**

Stage 2 lost on **both** terms, and that is the useful result:
1. **Plain HuggingFace transformers has no large init floor** (warm load = 4.4 s) —
   the ~33 s floor worth skipping lives in **vLLM**, not HF.
2. **The naive image contains all the weights** → it is ~VRAM-sized → disk-bound.

The productized systems (Dynamo Snapshot: 2.25 s restores; Cloudburst: 12 min → 10 s)
win by **image shrinking** — memory-mapping weights so they are *excluded* from the
image, and unmapping the KV cache (NVIDIA example: 190 GiB → 6 GiB). Combined with a
real engine floor to skip, that flips both terms of the rule. That is **Stage 3**
([STAGE3_PLAN.md](STAGE3_PLAN.md)).

---

## What's here

```
experiment-05-criu-checkpoint-restore/
├── README.md              # this file
├── LEARNINGS_CPU.md       # full teaching walkthrough — CPU stage (how CRIU works, every flag, issues)
├── LEARNINGS_GPU.md       # full teaching walkthrough — GPU stage (cuda-checkpoint, the honest analysis)
├── STAGE3_PLAN.md         # next step: CRIU on vLLM + image shrinking (the Doubleword blog)
├── Dockerfile             # reproducible environment (criu + cuda-checkpoint + torch)
├── scripts/
│   ├── cpu/               # counter.py, start/checkpoint/restore.sh, cpu_counter.py, start_model.sh
│   └── gpu/               # gpu_counter.py, start_gpu.sh
└── results/               # RESULTS_cpu.md, RESULTS_gpu.md (raw numbers + interpretation)
```

## Environment

- **OS:** Ubuntu 24.04. **Note:** `criu` was dropped from Noble's repos (build failure,
  Launchpad #2066148) — install from the CRIU team PPA: `sudo add-apt-repository ppa:criu/ppa`.
- **GPU stage:** NVIDIA driver **r550+** (r570+ ideal) for `cuda-checkpoint`; criu 4.2
  ships `/usr/lib/criu/cuda_plugin.so` which auto-invokes it during dump/restore.
- **cuda-checkpoint binary:** `github.com/NVIDIA/cuda-checkpoint` →
  `bin/x86_64_Linux/cuda-checkpoint` (note the `_Linux` suffix) → put on `PATH`.
- CRIU is Linux-only but works in ordinary VMs (unlike GDS — no bare metal needed).
- The `Dockerfile` captures this whole toolchain. A pre-built image is published:
  ```bash
  docker pull avaneesh1830/criu-coldstart:latest
  ```
  (Rebuild locally with `docker build --platform linux/amd64 -t avaneesh1830/criu-coldstart:latest .`)

## How to run

### Stage 1 — CPU counter (any Linux box, no GPU)
```bash
cd scripts/cpu
sudo apt-get install -y criu          # from ppa:criu/ppa on Ubuntu 24.04
./start.sh                            # note the PID
# let it count to ~30
sudo ./checkpoint.sh <PID>            # dumps + kills it
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches   # true cold restore
sudo ./restore.sh
tail -f counter.log                   # PASS: resumes at N+1
```

### Stage 2 — model on GPU (needs a GPU box, driver r550+)
```bash
cd scripts/gpu
# install criu + cuda-checkpoint + torch (see Dockerfile), pre-download the model
MODEL_ID=Qwen/Qwen2.5-7B-Instruct ./start_gpu.sh    # loads model into VRAM
PID=$(pgrep -f gpu_counter.py)
sudo criu dump    -t $PID -D ckpt_gpu -v4 -o dump.log   # cuda plugin drains VRAM→host
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo criu restore -D ckpt_gpu -d -v4 -o restore.log     # re-uploads VRAM
tail -f gpu.log                       # PASS: counter resumes, inference intact
nvidia-smi                            # VRAM back to ~15 GB
```

Full step-by-step reasoning, every flag, and all issues hit are in the two
`LEARNINGS_*.md` files.

## Reproducibility notes / gotchas
- Launch the target **detached** (`setsid … < /dev/null &> log`) or CRIU refuses to
  dump a terminal-attached process.
- Load models with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` — `huggingface_hub`
  otherwise keeps a live TCP socket open, which CRIU refuses to dump.
- Drop the page cache before restore for the honest **cold** (disk-bound) number.
- On a big-RAM box, dump wall-time is cache-buffered (not the disk cost); the
  restore-after-drop_caches number is the honest one.
- `chown` the images directory between `sudo` runs (root-owned-file trap).

See the root [SESSION_LOG.md](../SESSION_LOG.md) and [PHASE4_PLAN.md](../PHASE4_PLAN.md)
for how this experiment fits the overall cold-start arc: 650 s (naive) → 41.8 s
(loaders, exp 03–04) → single-digit seconds (snapshot/restore — Stage 3 target).
