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

## Suggested experiment ladder (smallest -> real)

**3a. Prove Lever B in isolation (cheap, high-value, ~30 min).**
Show that a memory-mapped file-backed region is excluded from a CRIU image.
- Write two tiny scripts: one allocates N GB in ANONYMOUS memory, one mmaps an
  N GB file. Checkpoint both; compare `du -sh ckpt/`.
- Expected: anonymous -> image ~= N GB; mmap -> image ~= tiny. This isolates the
  "image shrinking" mechanism before touching vLLM. Can even be done on a CPU box.

**3b. Checkpoint a live vLLM worker (the core of Stage 3).**
- Launch vLLM serving a small model (Qwen2.5-7B or the 1.5B) with the API up and
  one warm request done (so compile/graphs are built = the floor is "paid").
- Confirm it's checkpointable: vLLM spawns worker processes and opens sockets
  (API server, Ray/ZMQ). Expect the same class of issues as Stage 1's TCP socket —
  may need to checkpoint only the engine worker subtree, or bring the server to a
  quiescent state. This is the main engineering of Stage 3.
- `criu dump` the worker (cuda plugin drains VRAM), then restore, then verify the
  server still answers a request correctly.

**3c. Add image shrinking (Lever B) to the vLLM checkpoint.**
- Ensure weights are mmap'd/file-backed (vLLM + safetensors already mmap; verify
  they land in the image as references, not content). Unmap KV cache if feasible.
- Measure image size before/after shrinking.

**3d. The headline measurement.**
- restore time (cold)  vs  our exp-3/4 cold-start baseline (41.8 s for 110B w/
  InstantTensor; or the 7B/30B equivalents). Report the head-to-head.
- Do this on FAST storage (see below) so the storage term isn't the bottleneck.

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
