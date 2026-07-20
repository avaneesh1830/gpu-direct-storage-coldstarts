# Experiment 5, Stage 2 — GPU checkpoint/restore (cuda-checkpoint + CRIU)

Box: brev-pp5s8ux0o (Nebius), H100 80GB, driver 580.159.04, Ubuntu 24.04
Storage: /dev/vda1 virtio, 0.13 GB/s (dd O_DIRECT). NO local NVMe. 196 GB RAM.
Tools: criu 4.2 + /usr/lib/criu/cuda_plugin.so ; cuda-checkpoint 580.159.04
Model: Qwen2.5-7B-Instruct, fp16, plain HuggingFace transformers (NOT vLLM)

## Result: PASS (mechanism)
Frozen at gpu_counter=44, restored cold -> resumed at 47; inference still
returns "Paris"; VRAM 15.3 GB -> 0 (dump) -> 15.3 GB (restore). Same PID.
GPU device state fully captured and re-uploaded by cuda-checkpoint.

| Metric | Value |
|---|---|
| VRAM resident | 15.28 GB |
| Checkpoint image | 17 GB (VRAM drained to host + process RAM) |
| Dump time | 15.4 s (write buffered into 196 GB RAM; not disk-bound) |
| Restore time (cold) | 143 s (2:23) |

## Honest analysis — mechanism works, but net-win condition NOT met here
Restore = 143 s. 17 GB / 143 s = 0.12 GB/s = the virtio disk speed. Restore is
DISK-BOUND: it reads the whole memory image (incl. all 15 GB of weights) back
from a slow disk. Two reasons this doesn't beat a fresh start here:

1. WRONG ENGINE: plain HuggingFace transformers has NO large engine-init floor
   (no torch.compile / CUDA-graph / NCCL). Fresh warm load was 4.4 s. There is
   almost nothing for restore to "skip", so paying 143 s to read an image back
   is a net loss. The ~33 s floor that makes snapshotting worthwhile lives in
   vLLM, not HF.
2. WRONG STORAGE: 0.13 GB/s virtio. Restore time = image_size / disk_speed, so a
   17 GB image is inherently ~130 s here regardless.

## The decision rule, refined for GPU
    snapshot wins  <=>  (engine-init skipped)  >  (image_size / disk_speed)
- 7B plain HF, this box: ~0 s skipped  vs  143 s read  -> LOSS (as measured).
- 110B vLLM, fast NVMe: ~33 s skipped ... but naive image is ~62 GB+, so even at
  2.8 GB/s that's ~22 s read -> only marginal. The PRODUCTIZED win (NVIDIA Dynamo
  Snapshot 2.25 s, Doubleword Cloudburst) needs IMAGE SHRINKING: memory-map the
  weights from model files so they're NOT in the CRIU image, and unmap the KV
  cache. That collapses the image from ~VRAM-size to ~engine-state-size (their
  example: 190 GiB -> 6 GiB), which is what makes restore beat cold start.

## Takeaway
Stage 2 proves GPU device state (VRAM + /dev/nvidia handles) can be
checkpointed and restored intact via cuda-checkpoint + CRIU's cuda plugin.
The remaining work to a REAL cold-start win is Stage 3: apply this to vLLM
(to capture the engine-init floor) with image-shrinking (mmap weights / KV
unmap) so the image is small enough that restore < cold start. That is exactly
the Doubleword blog / Dynamo Snapshot approach.

## Setup notes (reproducible)
- cuda-checkpoint binary: github.com/NVIDIA/cuda-checkpoint bin/x86_64_Linux/
  (NOT bin/x86_64) -> /usr/local/bin ; needs driver r550+ (this box r580, full support)
- criu 4.2 from ppa:criu/ppa ships cuda_plugin.so; criu dump auto-invokes it
- HF_HUB_OFFLINE=1 to avoid the live-TCP-socket dump failure (same as CPU stage)
- dump wall-time is cache-buffered on a 196 GB box; restore-after-drop_caches is
  the honest disk-bound number
