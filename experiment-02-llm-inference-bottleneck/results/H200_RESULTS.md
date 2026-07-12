# 🔵 NVIDIA H200 141GB — Benchmark Results

> **Status:** ✅ Complete  
> **Date:** June 22, 2026  
> **Infrastructure:** NVIDIA H200 141GB · Verda via Brev · Single GPU  
> **vLLM Version:** 0.23.0 · FlashAttention 3 · PagedAttention  
> **Docker Image:** avaneesharoor/ml-experiments:latest  

---

## 🖥️ Hardware Specs

| Component | Spec |
|---|---|
| GPU | NVIDIA H200 141GB SXM |
| VRAM | 140.4 GB HBM3e |
| Memory Bandwidth | 4.8 TB/s |
| CPU | 44 vCPUs |
| System RAM | 185 GiB |
| Cloud | Verda via Brev |
| Region | Helsinki, Finland |

> H200 has **43% more memory bandwidth** than H100 (4.8 vs 3.35 TB/s) and **76% more VRAM** (141 vs 80 GB). The extra VRAM lets 32B/72B models run at full BF16 precision (no quantization needed).

---

## 📊 Results Summary

| Model | Params | Quantization | TTFT | TPOT | Throughput | E2E Latency |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | BF16 | 636.9 ms | 2.1 ms | **153.2 tok/s** | 1,075 ms |
| Qwen2.5-7B-Instruct | 7B | BF16 | 521.3 ms | 4.8 ms | 100.2 tok/s | 1,590 ms |
| Qwen2.5-32B-Instruct | 32B | BF16 (full precision) | 537.0 ms | 17.9 ms | 35.6 tok/s | 4,535 ms |
| Mistral-Large-2407 | 123B | AWQ INT4 | 487.5 ms | 36.3 ms | 16.2 tok/s | 8,887 ms |

> **TTFT** = Time To First Token · **TPOT** = Time Per Output Token · **E2E** = End-to-end latency

---

## 📈 Detailed Results

### 1️⃣ Qwen2.5-1.5B-Instruct — BF16

| Metric | Value |
|---|---|
| Server Startup | instant |
| GPU VRAM Used | 126.6 / 140.4 GB |
| CPU Usage | 1% |
| System RAM | 7.2 GB |
| Mean TTFT | 636.9 ms |
| P50 TTFT | 635.8 ms |
| P99 TTFT | 637.6 ms |
| Mean TPOT | 2.1 ms |
| Mean E2E | 1,075.5 ms |
| tok/s per request | 149.5 |
| Total tok/s | 153.2 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 638ms | 2.1ms | 185.7 |
| #2 | Python merge lists | 638ms | 2.1ms | 133.5 |
| #3 | ML differences | 637ms | 2.1ms | 175.5 |
| #4 | WW1 causes | 636ms | 2.0ms | 87.2 |
| #5 | Transformer architecture | 636ms | 2.1ms | 165.3 |

---

### 2️⃣ Qwen2.5-7B-Instruct — BF16

| Metric | Value |
|---|---|
| Server Startup | 20,070 ms (torch.compile + CUDA graph capture) |
| GPU VRAM Used | 127.1 / 140.4 GB |
| CPU Usage | 0% |
| System RAM | 8.3 GB |
| Mean TTFT | 521.3 ms |
| P50 TTFT | 519.6 ms |
| P99 TTFT | 522.8 ms |
| Mean TPOT | 4.8 ms |
| Mean E2E | 1,589.5 ms |
| tok/s per request | 98.7 |
| Total tok/s | 100.2 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 523ms | 4.8ms | 116.6 |
| #2 | Python merge lists | 523ms | 4.8ms | 74.1 |
| #3 | ML differences | 520ms | 4.8ms | 104.9 |
| #4 | WW1 causes | 521ms | 4.8ms | 82.5 |
| #5 | Transformer architecture | 520ms | 4.8ms | 115.2 |

---

### 3️⃣ Qwen2.5-32B-Instruct — BF16 (full precision, no quantization)

| Metric | Value |
|---|---|
| Server Startup | 33,114 ms (torch.compile + CUDA graph capture) |
| GPU VRAM Used | 126.8 / 140.4 GB |
| CPU Usage | 0% |
| System RAM | 8.9 GB |
| Mean TTFT | 537.0 ms |
| P50 TTFT | 537.1 ms |
| P99 TTFT | 537.6 ms |
| Mean TPOT | 17.9 ms |
| Mean E2E | 4,535.0 ms |
| tok/s per request | 35.5 |
| Total tok/s | 35.6 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 538ms | 17.9ms | 38.4 |
| #2 | Python merge lists | 538ms | 17.9ms | 27.3 |
| #3 | ML differences | 538ms | 17.9ms | 37.7 |
| #4 | WW1 causes | 535ms | 18.0ms | 34.7 |
| #5 | Transformer architecture | 537ms | 17.9ms | 39.4 |

> **Note:** Run at full BF16 — H100 had to use AWQ INT4 for this model. H200's 141GB fits the full-precision weights, giving better output quality at nearly identical speed (17.9ms vs H100's 16.8ms AWQ).

---

### 4️⃣ Mistral-Large-2407 (123B) — AWQ INT4 (awq_marlin)

| Metric | Value |
|---|---|
| Server Startup | 34,121 ms (torch.compile + CUDA graph capture) |
| GPU VRAM Used | 128.9 / 140.4 GB |
| CPU Usage | 0% |
| System RAM | 8.3 GB |
| Mean TTFT | 487.5 ms |
| P50 TTFT | 471.0 ms |
| P99 TTFT | 498.4 ms |
| Mean TPOT | 36.3 ms |
| Mean E2E | 8,887.3 ms |
| tok/s per request | 16.4 |
| Total tok/s | 16.2 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 499ms | 36.3ms | 17.8 |
| #2 | Python merge lists | 498ms | 36.3ms | 11.8 |
| #3 | ML differences | 498ms | 36.3ms | 17.6 |
| #4 | WW1 causes | 471ms | 36.5ms | 17.9 |
| #5 | Transformer architecture | 471ms | 36.3ms | 16.9 |

> **Note:** True 123B dense model (Mistral-Large). At AWQ INT4 it occupies ~62GB weights; runs comfortably on the H200 with KV-cache headroom.

---

## 🔍 Analysis

### Memory bandwidth advantage over H100 is dramatic
The smaller models show H200's 4.8 TB/s bandwidth clearly:
- **1.5B:** 3.1ms (H100) → 2.1ms (H200) — 32% faster TPOT
- **7B:** 9.1ms (H100) → 4.8ms (H200) — 47% faster TPOT, throughput 66→100 tok/s

### Full precision without speed penalty
The 32B model ran at full BF16 on H200 (17.9ms TPOT) versus AWQ INT4 on H100 (16.8ms). The H200 delivers full-precision quality at essentially the same speed the H100 only achieved with lossy quantization — a direct benefit of the larger VRAM.

### TTFT is elevated and noisy
The flat ~500-640ms TTFT across all sizes reflects the 5 concurrent requests in the benchmark queuing against each other, not true prefill cost. The 1.5B's 637ms TTFT being higher than larger models confirms this is queuing noise, not compute.

### VRAM pre-allocation
vLLM reserved ~127GB (90% of 141GB) upfront for KV-cache across all runs — standard PagedAttention behaviour, independent of actual model size.
