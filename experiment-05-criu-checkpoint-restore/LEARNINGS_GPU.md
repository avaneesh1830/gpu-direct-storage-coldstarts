# Experiment 5, Stage 2 — GPU Checkpoint/Restore: Complete Learnings

**What this is:** the full teaching walkthrough of Stage 2 (GPU) — how GPU
checkpointing works, every command and flag, what we proved, the honest analysis
of why it doesn't yet beat a cold start, and the on-ramp to Stage 3.
Companion to [LEARNINGS.md](LEARNINGS.md) (CPU stage) and [RESULTS_GPU.md](RESULTS_GPU.md).

**Did it work? YES.** GPU device state (15 GB of VRAM + the `/dev/nvidia*` handles)
was checkpointed to disk, the process was killed, and it was restored with the
weights byte-correct and the model still answering "Paris." The *mechanism* is
proven. Whether it *beats a cold start* is a separate question (§5) — and the
answer here is "not yet, and the data tells us exactly why."

---

## 0. The one-sentence version

A GPU process can't be checkpointed by CRIU alone because its state lives in VRAM
and behind `/dev/nvidia*` device handles CRIU can't see; NVIDIA's `cuda-checkpoint`
drains that GPU state into ordinary host RAM (and releases the GPU) so CRIU can
snapshot it like any other process, then re-uploads it on restore.

---

## 1. Why the CPU recipe doesn't "just work" on a GPU

In Stage 1, CRIU could snapshot everything because a normal process's entire state
lives in things CRIU can read: its RAM (`/proc/<pid>/`), registers, and open files.

A GPU process breaks that assumption in two ways:

1. **The weights live in VRAM**, which is memory on the GPU board — *not* in the
   process's system-RAM maps. CRIU walking `/proc/<pid>/maps` simply cannot see it.
2. **The process holds open GPU device handles** — file descriptors on
   `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, etc. These are owned by the
   NVIDIA driver. If CRIU tries to dump a process holding them, it chokes — the same
   category of problem as the live TCP socket from Stage 1 (state CRIU doesn't
   control).

So we need something to (a) move VRAM into host RAM where CRIU *can* see it, and
(b) release the device handles so the process looks "normal" to CRIU. That
something is `cuda-checkpoint`.

---

## 2. How `cuda-checkpoint` works (the state machine)

`cuda-checkpoint` is a small NVIDIA tool that drives a CUDA process through four
states via `--action`:

```
running  --lock-->  locked  --checkpoint-->  checkpointed
checkpointed  --restore-->  locked  --unlock-->  running
```

- **lock** — stop the process from issuing new CUDA calls and wait for in-flight
  ones to finish. (Has a `--timeout`; it waits for the GPU to reach a quiescent
  point. Our loop sleeps 1 s between GPU ops, so there's always a safe moment.)
- **checkpoint** — copy all of the process's **VRAM down to host RAM**, then
  **release the GPU** (close the `/dev/nvidia*` handles, free the device memory).
  After this the process holds *zero* GPU resources — `nvidia-smi` shows 0 MiB for
  it. Now it's an ordinary CRIU-dumpable process whose host RAM happens to contain
  what used to be on the GPU.
- **restore** — re-acquire the GPU and copy the saved memory **back up to VRAM**.
- **unlock** — let the process resume issuing CUDA calls.

`--toggle` is a convenience that does lock+checkpoint (if running) or
restore+unlock (if checkpointed). `--get-state` prints the current state.

**Who calls it?** You can call it by hand, but CRIU 4.2 ships a **CUDA plugin**
(`/usr/lib/criu/cuda_plugin.so`). When present, `criu dump` automatically invokes
`cuda-checkpoint` at the right moment (lock+checkpoint before dumping), and
`criu restore` invokes it after (restore+unlock). So our actual dump/restore
commands were *identical to the CPU stage* — the plugin did the GPU work
transparently. That's why the run "just worked": criu + plugin + cuda-checkpoint
formed a complete chain.

---

## 3. The setup, step by step (and the gotchas in each)

### Recon first — the driver-version gate
```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```
- Got `H100 80GB, 580.159.04`. **`cuda-checkpoint` needs driver r550+, ideally
  r570+.** r580 = full support. This is the GPU equivalent of "is criu even
  installable" — check it before anything else. An old driver here would have
  been a hard stop.

### Storage reality check
```bash
dd if=/dev/zero of=/tmp/ddtest bs=1M count=2048 oflag=direct
```
- `131 MB/s = 0.13 GB/s`. Same slow **virtio** disk as the earlier boxes; `lsblk`
  showed **no NVMe**, and `/ephemeral` turned out to be just a directory on the
  same `vda1` (also 129 MB/s). `oflag=direct` bypasses the page cache to measure
  the *true* disk speed, not RAM-buffered writes. This number sets the restore
  floor: restore ≈ image_size ÷ disk_speed.

### Install CRIU + the CUDA plugin
```bash
sudo add-apt-repository -y ppa:criu/ppa && sudo apt-get install -y criu
```
- Same PPA as Stage 1 (criu isn't in Ubuntu 24.04). criu 4.2 **includes**
  `/usr/lib/criu/cuda_plugin.so` — confirmed with `find /usr -name '*cuda*plugin*'`.
  If the plugin were missing we'd have had to drive `cuda-checkpoint` by hand.

### Fetch the `cuda-checkpoint` binary
```bash
curl -fsSL -o cuda-checkpoint \
  https://github.com/NVIDIA/cuda-checkpoint/raw/main/bin/x86_64_Linux/cuda-checkpoint
sudo cp cuda-checkpoint /usr/local/bin/
```
- **Gotcha:** the path is `bin/x86_64_Linux/` (with the `_Linux` suffix), *not*
  `bin/x86_64/` — the obvious guess 404s. Found the real path via the GitHub API
  (`git/trees/main?recursive=1`). It must be on `PATH` (we used `/usr/local/bin`)
  so the criu plugin can exec it.
- It's an 8 KB binary; `--version` reported `580.159.04`, matching the driver.

### Install CUDA torch + transformers in a venv
```bash
python3 -m venv ~/venv
~/venv/bin/pip install torch transformers accelerate safetensors
```
- venv because Ubuntu 24.04 blocks system pip (PEP-668). Default torch index
  pulls the CUDA build (`2.13.0+cu130`, `cuda True`).

### The workload — `gpu_counter.py`
Same three-in-one proof as the CPU model script, but on the GPU:
```python
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda()
...
gpu_c = torch.zeros(1, device="cuda")   # a GPU-resident counter (live GPU state)
...
    gpu_c += 1                          # a real GPU op every tick
    if i % 5 == 0:
        out = model.generate(ids, ...)  # weight-integrity check -> "Paris"
```
- `.cuda()` puts the 7B weights in VRAM (15.28 GB, confirmed by `nvidia-smi`).
- `torch.float16` on GPU (fast, half the memory of fp32; the opposite choice from
  the CPU stage, where fp16 is poorly supported).
- `gpu_c` on `device="cuda"` is the GPU analog of the CPU counter — it proves
  *live GPU compute state* (not just weights) survives.
- `MODEL_ID` is an env var so the same script scales to 7B/32B/etc.
- Launched with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (pre-downloaded the model
  first) to avoid the live-TCP-socket dump failure learned in Stage 1.

### Dump and restore — identical commands to the CPU stage
```bash
sudo criu dump    -t "$PID" -D ckpt_gpu -v4 -o dump.log
sync; echo 3 > /proc/sys/vm/drop_caches      # true cold restore
sudo criu restore -D ckpt_gpu -d  -v4 -o restore.log
```
- No GPU-specific flags — the `cuda_plugin.so` hooks in automatically. That's the
  elegant part: once the pieces are installed, GPU checkpointing looks exactly like
  CPU checkpointing.

---

## 4. What happened — the proof chain

| Stage | Observation | What it proves |
|---|---|---|
| before dump | `cuda-checkpoint --get-state` = `running`; VRAM 15,427 MiB | model live on GPU |
| after dump | `DUMP OK`; process dead; **VRAM 0 MiB**; image 17 GB | cuda-checkpoint drained VRAM→host, released GPU, criu captured it |
| after restore | `RESTORE OK`; **VRAM 15,427 MiB** again | GPU state re-uploaded |
| resume | counter froze ~44, resumed 47, 48, 49… | live GPU compute state survived |
| inference | still `Paris. Which` after restore | all 15 GB of weights byte-correct |
| identity | same PID 21404, still alive | genuine resurrection, not a restart |

**Timings:** dump 15.4 s, restore 143 s (cold), image 17 GB.

---

## 5. The honest analysis — mechanism ✓, net-win ✗ (and why that's the useful part)

Restore took **143 s**, and `17 GB ÷ 143 s = 0.12 GB/s` = the virtio disk speed
exactly. **Restore is disk-bound: it reads the entire memory image — including all
15 GB of weights — back from a slow disk.** On this setup, that loses to a fresh
start, for two independent reasons:

1. **Wrong engine.** Plain HuggingFace transformers has **no large engine-init
   floor**. A warm load was 4.4 s. There's almost nothing for restore to "skip," so
   spending 143 s to read an image back is a net loss. The ~33 s floor
   (torch.compile + CUDA-graph capture + NCCL + worker spawn) that makes
   snapshotting worthwhile lives in **vLLM**, not HF.
2. **Wrong storage.** 0.13 GB/s virtio. Since restore = image_size ÷ disk_speed, a
   17 GB image is inherently ~130 s here regardless of anything else.

### The refined decision rule (now with GPU data)
> **snapshot wins  ⟺  (engine-init time skipped)  >  (image size ÷ disk speed)**

- 7B plain HF, this box: ~0 s skipped **vs** 143 s read → **LOSS** (measured).
- Even 110B vLLM on fast NVMe: ~33 s skipped, but a *naive* image is ~62 GB+, so
  at 2.8 GB/s that's ~22 s to read — only a marginal win.

### Why the productized systems win: image shrinking
NVIDIA Dynamo Snapshot (2.25 s restores) and Doubleword Cloudburst (12 min → 10 s)
don't dump the weights into the image at all. Two tricks:
- **Memory-map the weights** from the model files on disk, so those pages are
  *file-backed*, and CRIU records only a reference to the file — **not** the
  gigabytes of content. The weights get re-mmap'd from disk (or page cache) on
  restore instead of being stored in and read from the image.
- **Unmap the KV cache** before checkpoint (it's regenerable), shrinking the image
  further (NVIDIA's example: 190 GiB → 6 GiB).

Result: the image collapses from ~VRAM-size to ~engine-state-size, so
`image ÷ disk_speed` becomes small and finally drops below the engine-init time
skipped. **That** is the win.

---

## 6. What was achieved & learned

**Achieved**
- End-to-end GPU checkpoint/restore: VRAM 15.3 GB → 0 → 15.3 GB, weights intact,
  live GPU counter resumed, same PID — all on the first real attempt.
- A clean, installable, reproducible GPU-CRIU toolchain (driver r580 + criu 4.2 +
  cuda_plugin.so + cuda-checkpoint 580).

**Learned**
1. **GPU state is capturable** — VRAM and device handles included — via
   `cuda-checkpoint` + CRIU's cuda plugin, with *no change* to the dump/restore
   commands.
2. **Naive GPU CRIU is disk-bound and doesn't beat a cold start**, because the
   image contains all the weights. Proven, not assumed: restore clocked the exact
   virtio disk rate.
3. **The two conditions for a real win are separable and both were absent here:**
   a big init floor (needs vLLM) and a small image (needs weight-mmap / KV-unmap).
   This decomposition is the roadmap for Stage 3.
4. **The driver version is the GPU gate** — check `nvidia-smi` before investing in
   anything else.

---

## 7. Setup gotchas (quick reference)
- `cuda-checkpoint` binary path is `bin/x86_64_Linux/` (the `_Linux` suffix), not
  `bin/x86_64/` — the naive URL 404s.
- Needs driver **r550+** (r570+ for full granular actions); verify with `nvidia-smi`.
- criu 4.2 (ppa:criu/ppa) already ships `/usr/lib/criu/cuda_plugin.so`; put
  `cuda-checkpoint` on `PATH` so the plugin can exec it.
- Same Stage-1 rules still apply: launch detached (`setsid`), load models with
  `HF_HUB_OFFLINE=1` (no live socket), drop caches before restore for the honest
  cold number, and `chown` the image dir between sudo runs.
- Dump wall-time is *not* the disk cost on a big-RAM box — writes buffer into RAM
  (196 GB here). Restore-after-drop_caches is the honest disk-bound measurement.

---

## 8. On-ramp to Stage 3 (the Doubleword blog)
The GPU result motivates the mentor's final step exactly. To make restore actually
beat a cold start we need both missing ingredients:
- **A real engine floor to skip** → checkpoint a **vLLM** worker, not plain HF.
- **A small image** → memory-map weights (exclude them from the image) and unmap
  the KV cache.
That is precisely what Doubleword's "Reverse-engineering cuda-checkpoint" and
"Cloudburst" posts, and NVIDIA Dynamo Snapshot, implement. Stage 3 = apply this
stack to vLLM and measure restore vs. our earlier 41.8 s cold-start baseline.
