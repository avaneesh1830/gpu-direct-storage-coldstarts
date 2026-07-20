# Experiment 5, Stage 1 — CRIU CPU counter checkpoint/restore

Box: brev-tg4rwc0bv (Nebius), Ubuntu 24.04 (noble), amd64, CPU-only
CRIU: 4.2-1ppa1.24.04 (from ppa:criu/ppa — not in noble universe; bug #2066148)

## Result: PASS
Process frozen at `count = 13`, killed (verified dead), page cache dropped,
restored cold → resumed at `count = 14, 15, 16, 17` with the same PID.

| Metric | Value |
|---|---|
| Dump time | 0.02 s |
| Restore time (cold cache) | 0.08 s (exit 0) |
| Checkpoint image size | 4.0 MB |

Interpretation: restore reconstructs an already-initialized process from a
4 MB on-disk image in 80 ms, skipping all program startup. For a real vLLM
worker this is the lever that bypasses the ~33 s engine-init floor measured
in experiments 2-4 (image would be tens of GB = model weights in RAM, so
restore time then scales with image size / disk speed — same storage-bound
physics as before, but reading back a ready process instead of re-running init).

## Gotchas hit
- criu absent from Ubuntu 24.04 universe (FTBFS, LP#2066148) -> install via ppa:criu/ppa
- `criu check` needs root -> `sudo criu check`
- ckpt/ and *_time.txt become root-owned after sudo runs -> chown or sudo rm before re-runs

---

# Stage 1b — real model resident in RAM (Qwen2.5-0.5B-Instruct, float32, CPU)

Same box. Model held in system RAM (~2.4 GB RSS), GPU not involved.
Process ticks a counter + runs inference ("The capital of France is" -> "Paris")
every 5 ticks. Checkpointed, killed (verified dead), cache dropped, restored cold.

## Result: PASS
Frozen at tick=49, resumed at tick=52; inference still returns "Paris. It"
after restore -> all 2.2 GB of weights came back correct, no reload.

| Metric | Toy counter | 0.5B model in RAM |
|---|---|---|
| Checkpoint image | 4 MB | 2.2 GB |
| Dump time | 0.02 s | 12.8 s |
| Restore (cold cache) | 0.08 s | 18.6 s |

## Key finding
Restore is DISK-BOUND: 2.2 GB / 18.6 s ~= 0.12 GB/s == the virtio disk speed
measured in exp 3-4. The checkpoint image is a memory snapshot; restoring =
sequential read of it. So:

    snapshot/restore wins iff (init time skipped) - (image read time) > 0

For 0.5B on slow virtio disk this is NEGATIVE (fresh warm load ~2-3 s beats an
18.6 s image read) -- init is too cheap to be worth snapshotting. The win
appears only when skipped-init >> image-read: i.e. the 110B vLLM case with a
~33 s engine-init floor, on fast NVMe. This run sharpens the thesis rather than
weakening it; the GPU 10B/30B test on an H100 (fast NVMe + real engine floor)
is where net savings should first go positive.

## Extra gotcha
- huggingface_hub keeps a live TCP socket in its connection pool after download;
  CRIU refuses to dump connected TCP sockets (sk-inet.c:200). Fix: load cached
  model with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 -> no socket -> clean dump.
  (Also the honest serverless config: weights pre-baked, load offline.)
