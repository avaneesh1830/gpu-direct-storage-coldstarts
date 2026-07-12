# 🟢 NVIDIA H100 80GB — Benchmark Results

> **Status:** ✅ Complete  
> **Date:** June 22, 2026  
> **Infrastructure:** NVIDIA H100 80GB · Hyperstack via Brev · Single GPU  
> **vLLM Version:** 0.23.0 · FlashAttention 3 · PagedAttention  
> **Docker Image:** avaneesharoor/ml-experiments:latest  

---

## 🖥️ Hardware Specs

| Component | Spec |
|---|---|
| GPU | NVIDIA H100 80GB SXM |
| VRAM | 79.7 GB HBM3 |
| Memory Bandwidth | 3.35 TB/s |
| CPU | 28 vCPUs |
| System RAM | 180 GiB |
| Storage | 850 GiB |
| Cloud | Hyperstack via Brev |

---

## 📊 Results Summary

| Model | Params | Quantization | TTFT | TPOT | Throughput | E2E Latency |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | BF16 | 368.0 ms | 3.1 ms | **160.2 tok/s** | 1,050 ms |
| Qwen2.5-7B-Instruct | 7B | BF16 | 375.5 ms | 9.1 ms | 65.7 tok/s | 2,419 ms |
| Qwen2.5-32B-Instruct-AWQ | 32B | AWQ INT4 | 473.3 ms | 16.8 ms | 37.8 tok/s | 4,228 ms |
| Llama-3.1-70B-Instruct-AWQ | 70B | AWQ INT4 | 473.4 ms | 32.6 ms | 20.0 tok/s | 8,010 ms |

> **TTFT** = Time To First Token · **TPOT** = Time Per Output Token · **E2E** = End-to-end latency

---

## 📈 Detailed Results

### 1️⃣ Qwen2.5-1.5B-Instruct — BF16

| Metric | Value |
|---|---|
| Server Startup | instant |
| GPU VRAM Used | 71.9 / 79.7 GB |
| CPU Usage | 0% |
| System RAM | 8.6 GB |
| Mean TTFT | 368.0 ms |
| P50 TTFT | 367.4 ms |
| P99 TTFT | 368.3 ms |
| Mean TPOT | 3.1 ms |
| Mean E2E | 1,049.9 ms |
| tok/s per request | 155.5 |
| Total tok/s | 160.2 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 369ms | 3.1ms | 185.7 |
| #2 | Python merge lists | 367ms | 3.1ms | 134.8 |
| #3 | ML differences | 368ms | 3.1ms | 173.7 |
| #4 | WW1 causes | 367ms | 3.0ms | 109.7 |
| #5 | Transformer architecture | 368ms | 3.1ms | 173.7 |

---

### 2️⃣ Qwen2.5-7B-Instruct — BF16

| Metric | Value |
|---|---|
| Server Startup | 53,139 ms (torch.compile + CUDA graph capture) |
| GPU VRAM Used | 72.0 / 79.7 GB |
| CPU Usage | 1% |
| System RAM | 9.9 GB |
| Mean TTFT | 375.5 ms |
| P50 TTFT | 375.6 ms |
| P99 TTFT | 375.9 ms |
| Mean TPOT | 9.1 ms |
| Mean E2E | 2,418.9 ms |
| tok/s per request | 65.6 |
| Total tok/s | 65.7 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 376ms | 9.1ms | 73.2 |
| #2 | Python merge lists | 376ms | 9.1ms | 50.6 |
| #3 | ML differences | 376ms | 9.1ms | 67.2 |
| #4 | WW1 causes | 374ms | 9.1ms | 64.6 |
| #5 | Transformer architecture | 376ms | 9.1ms | 72.4 |

---

### 3️⃣ Qwen2.5-32B-Instruct-AWQ — AWQ INT4 (awq_marlin)

| Metric | Value |
|---|---|
| Server Startup | instant (cached) |
| GPU VRAM Used | 72.5 / 79.7 GB |
| CPU Usage | 1% |
| System RAM | 9.4 GB |
| Mean TTFT | 473.3 ms |
| P50 TTFT | 472.6 ms |
| P99 TTFT | 474.3 ms |
| Mean TPOT | 16.8 ms |
| Mean E2E | 4,227.6 ms |
| tok/s per request | 37.5 |
| Total tok/s | 37.8 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 475ms | 16.8ms | 43.1 |
| #2 | Python merge lists | 473ms | 16.7ms | 27.0 |
| #3 | ML differences | 473ms | 16.7ms | 40.9 |
| #4 | WW1 causes | 472ms | 16.9ms | 35.2 |
| #5 | Transformer architecture | 474ms | 16.8ms | 41.6 |

---

### 4️⃣ Llama-3.1-70B-Instruct-AWQ — AWQ INT4 (awq_marlin)

| Metric | Value |
|---|---|
| Server Startup | 51,152 ms (torch.compile + CUDA graph capture) |
| GPU VRAM Used | 73.7 / 79.7 GB |
| CPU Usage | 1% |
| System RAM | 10.0 GB |
| Mean TTFT | 473.4 ms |
| P50 TTFT | 481.7 ms |
| P99 TTFT | 481.8 ms |
| Mean TPOT | 32.6 ms |
| Mean E2E | 8,010.0 ms |
| tok/s per request | 20.1 |
| Total tok/s | 20.0 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 482ms | 32.5ms | 22.6 |
| #2 | Python merge lists | 482ms | 32.5ms | 15.0 |
| #3 | ML differences | 440ms | 32.5ms | 21.2 |
| #4 | WW1 causes | 482ms | 32.7ms | 21.4 |
| #5 | Transformer architecture | 482ms | 32.5ms | 20.5 |

---

## 🔍 Analysis

### TPOT scales linearly with model size
3.1 → 9.1 → 16.8 → 32.6ms — proportional to parameter count. Decode is **memory-bandwidth-bound**: more weights = more bytes streamed per token from HBM. H100's 3.35 TB/s is the ceiling.

### TTFT stays flat across model sizes
368ms vs 473ms across 1.5B→70B. Benchmark sends 5 concurrent requests so queuing delay dominates over prefill compute. True single-request TTFT would differ significantly.

### VRAM reads ~72GB across all models
vLLM pre-allocates 90% of VRAM for KV-cache (PagedAttention). Actual weights: ~3GB (1.5B) · ~14GB (7B) · ~17GB (32B AWQ) · ~35GB (70B AWQ). Rest is KV-cache headroom — correct behaviour.

### Startup time 50s+ for large models
torch.compile + CUDA graph capture across batch sizes 1→512. One-time cost per instance — subsequent restarts reuse compiled cache.
