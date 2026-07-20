# Experiment 5, Stage 3 — Plan: CRIU on vLLM + image shrinking (the Doubleword step)

**Goal:** make a checkpoint restore actually BEAT a cold start — i.e. turn the
"mechanism works but net-loss" result of Stage 2 into a net win. This is the
mentor's third ask ("follow through the Doubleword blog").

**Where Stages 1–2 left us:**
- CPU + GPU CRIU proven; GPU VRAM checkpoint/restore works via cuda-checkpoint.
- Refined rule:  snapshot wins  <=>  (engine-init skipped) > (image size / disk speed).
- Stage 2 lost on BOTH terms: plain HF has ~0 s init floor, and the naive image
  held all 15 GB of weights on a 0.13 GB/s disk. Stage 3 fixes both.

---

## The two levers to flip (both required)

### Lever A — a real engine-init floor to skip  ->  use vLLM, not plain HF
Plain HuggingFace `from_pretrained` has almost no startup cost beyond weight load.
vLLM's ~33 s floor (measured in exp 2–4) is torch.compile + CUDA-graph capture +
NCCL + worker-process spawn. Restoring a *ready vLLM worker* skips all of it.
=> Stage 3 must checkpoint a live vLLM engine, not an HF model.

### Lever B — a small checkpoint image  ->  keep weights OUT of the image
Naive CRIU dumps all memory, so image ~= VRAM size -> disk-bound restore.
Doubleword / NVIDIA Dynamo Snapshot shrink the image two ways:
1. **Memory-map weights** from the model files so those pages are file-backed;
   CRIU stores only a *reference* to the file, not the GB of content. On restore
   the weights are re-mmap'd from disk/page-cache, not read out of the image.
2. **Unmap the KV cache** before checkpoint (it's regenerable). NVIDIA example:
   190 GiB -> 6 GiB image.
=> image collapses from ~VRAM-size to ~engine-state-size, so image/disk_speed
   drops below the init time skipped.

---

## Read first (the actual blog the mentor pointed at)
1. Doubleword — "Reverse-engineering NVIDIA's cuda-checkpoint for faster cold starts"
   https://blog.doubleword.ai/what-happens-when-you-checkpoint-a-cuda-process
   (what the driver does; how they restore up to 4x faster)
2. Doubleword — "Cloudburst: 70x faster cold(ish) starts for SGLang"
   https://blog.doubleword.ai/fast-sglang-starts
   (CRIU + cuda-checkpoint on a real engine: 12 min -> 10 s on B200)
3. NVIDIA Dynamo Snapshot (the productized version, K8s):
   https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/

Extract from each: exactly which memory regions they exclude from the image, how
they trigger cuda-checkpoint, and how they measure restore vs cold start.

---

## Concrete mechanism (extracted from the Doubleword posts, 2026)

The Cloudburst post is SGLang-specific and gives the actual recipe. Key facts:

- **Image shrinking = `torch_memory_saver`.** Before checkpoint, SGLang *releases*
  the weights AND KV cache from GPU (HTTP `POST /release_memory_occupation`). So
  cuda-checkpoint + CRIU capture only the small CUDA/torch engine state, NOT the
  weights. Image collapses **192 GB -> 6.6 GB**. After restore, weights are reloaded
  by SGLang's own loader (`POST /restore_memory_occupation`). Weights are therefore
  NOT in the image — they come back separately and fast.
- **Their result (8x B200, NVMe Gen4): restore 9.6 s vs cold start 695 s** (warm 88 s).
  Breakdown of the 9.6 s: container 3.0 s + CRIU/cuda-checkpoint 3.5 s + weight
  reload 3.1 s (~38 GB/s). Restore is dominated by re-reading weights, not the image.
- **cuda-checkpoint is driven explicitly** (driver 580+ for device migration):
  ```
  cuda-checkpoint --action lock       --pid $P   # quiesce (~0.3 ms)
  cuda-checkpoint --action checkpoint --pid $P   # VRAM->host, close /dev/nvidia*, leaves nvidia-smi
  #   ... criu dump ...
  #   ... criu restore ...
  cuda-checkpoint --action restore    --pid $P   # rebuild CUDA ctx, host->VRAM
  cuda-checkpoint --action unlock     --pid $P   # resume kernels
  ```
  (CRIU's cuda_plugin.so can do lock+checkpoint / restore+unlock automatically, as in
  our Stage 2; the blog drives it explicitly for control.)
- **Environment prep to make an engine checkpointable (the real engineering):**
  - `HF_HUB_OFFLINE=1` — no live outbound socket (we already hit this in Stage 1/2).
  - Bind all TCP listeners to `0.0.0.0` or `localhost`; `GLOO_SOCKET_IFNAME=lo`.
  - `TORCHINDUCTOR_COMPILE_THREADS=1` — stray compile threads else hold CUDA contexts.
  - Disable io_uring: `sudo sysctl -w kernel.io_uring_disabled=2` (CRIU can't dump it).
  - Persist compile caches across restore: `~/.cache/flashinfer`, `TRITON_HOME`,
    `TORCHINDUCTOR_CACHE_DIR`, `~/.cache/tvm-ffi`.
  - After restore, a helper hits `/restore_memory_occupation` to reload weights.
- **Checkpoint-time speedups (optional):** Transparent Huge Pages (3.9 s -> 1.6 s for
  8.5 GB), pre-allocated staging buffer via LD_PRELOAD, async unmap. Bottleneck is
  zeroing anonymous pages in `mmap(MAP_POPULATE)`, not PCIe transfer.

**Implication for our plan:** use **SGLang** (not vLLM) to match the blog and get
`torch_memory_saver` + the `/release`+`/restore_memory_occupation` endpoints for free.
The image-shrink is an engine feature, not a CRIU trick.

---

## Suggested experiment ladder (smallest -> real)

**3a. (Optional, cheap) mmap-vs-anonymous image-shrink toy (~30 min, CPU box).**
Allocate N GB anonymous vs mmap an N GB file; checkpoint both; compare `du -sh ckpt/`.
Confirms file-backed pages are excluded from the image. Lower priority now that the
blog shows the real shrink is `torch_memory_saver` (offload weights), not plain mmap.

**3b. SGLang checkpoint WITHOUT shrinking (baseline, single GPU + NVMe).**
- Install SGLang; serve a small model (1.5B/7B); do one warm request (build graphs).
- Apply the env prep above (io_uring off, TCP localhost, offline, single compile thread).
- `cuda-checkpoint lock+checkpoint` -> `criu dump` -> `criu restore` ->
  `cuda-checkpoint restore+unlock`; verify the server answers correctly.
- Expect a large (~VRAM-sized) image here — this is the "before" for shrinking.

**3c. Add `torch_memory_saver` shrinking (the core result).**
- Before checkpoint: `POST /release_memory_occupation` (weights+KV leave GPU).
- Checkpoint (small image ~engine state) -> restore -> `POST /restore_memory_occupation`.
- Measure image size before/after (target: VRAM-sized -> single-digit GB).

**3d. The headline measurement.**
- restore time (cold) vs our exp-3/4 cold-start baseline (7B/30B equivalents; 41.8 s
  for 110B w/ InstantTensor). Report the head-to-head on FAST NVMe storage.

---

## Provisioning notes for Stage 3
- **Storage matters again.** This H100 box was 0.13 GB/s virtio with no NVMe. For a
  real restore-vs-cold-start number, get a box with genuine local NVMe (run the
  Stage-2 `dd ... oflag=direct` test; want >1 GB/s). Even with a shrunk image,
  slow disk caps the win. (Bare-metal or an NVMe-backed instance; same preflight
  discipline as exp 3–4.)
- Driver r550+ (r570+ ideal) — same gate as Stage 2.
- Reuse the whole Stage-2 toolchain: criu 4.2 (ppa) + cuda_plugin.so +
  cuda-checkpoint (bin/x86_64_Linux/), venv torch, HF_HUB_OFFLINE=1.

## Success criterion for Stage 3
A restored vLLM worker serves a correct response, and
   restore_time_cold  <  cold_start_time
for at least one model size — with the image-shrinking making the difference.
That closes the loop from the 650 s naive baseline -> 41.8 s (loaders) ->
single-digit-seconds (snapshot restore), matching Dynamo/Doubleword.

## Risks / open questions to watch
- vLLM multi-process + sockets may be much harder to checkpoint than a single HF
  process (the real reason this is "engineering", not one command).
- Whether vLLM's CUDA allocator / graphs restore cleanly via cuda-checkpoint.
- Whether weights actually stay out of the image (mmap vs. copied-to-GPU-then-
  drained-to-host — cuda-checkpoint drains VRAM to host anonymous memory, which
  WOULD land in the image; the shrinking trick may require keeping a host copy
  mmap'd and remapping, or vLLM/Dynamo-specific handling). Clarify from the blogs.
