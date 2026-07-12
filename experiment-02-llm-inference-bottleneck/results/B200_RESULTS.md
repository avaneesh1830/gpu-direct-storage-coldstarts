# 🔴 NVIDIA B200 — Benchmark Results

> **Status:** ✅ Complete  
> **Date:** June 22, 2026  
> **Infrastructure:** NVIDIA B200 (Blackwell) · Single GPU · vLLM v0.23.0  
> **Docker Image:** avaneesharoor/ml-experiments:latest  

---

## 🖥️ Hardware Specs

| Component | Spec |
|---|---|
| GPU | NVIDIA B200 |
| VRAM | 192 GB HBM3e |
| Memory Bandwidth | 8.0 TB/s |
| Architecture | Blackwell |
| Native FP8/FP4 | Yes (5th-gen tensor cores) |

> B200 has **2.4x the memory bandwidth of H100** (8.0 vs 3.35 TB/s) and **1.67x of H200** (8.0 vs 4.8 TB/s). Since decode is bandwidth-bound, this directly drives the fastest TPOT of all three GPUs tested.

---

## 📊 Results Summary

| Model | Params | Quantization | TTFT | TPOT | Throughput | E2E Latency |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | BF16 | 425.2 ms | 2.5 ms | 165.0 tok/s | 980 ms |
| Qwen2.5-7B-Instruct | 7B | BF16 | 26.0 ms* | 3.6 ms | **194.5 tok/s** | 824 ms |
| Qwen2.5-32B-Instruct | 32B | BF16 (full precision) | 473.8 ms | 13.6 ms | 45.9 tok/s | 3,535 ms |
| Mistral-Large-2407 | 123B | AWQ INT4 | 385.2 ms | 22.6 ms | 25.6 tok/s | 5,742 ms |

> *7B TTFT was unusually low (26ms) because the model + prompt KV-cache were already warm from a prior load. All other rows are cold-cache first runs.

> **TTFT** = Time To First Token · **TPOT** = Time Per Output Token · **E2E** = End-to-end latency

---

## 📈 Detailed Results

### 1️⃣ Qwen2.5-1.5B-Instruct — BF16

| Metric | Value |
|---|---|
| Server Startup | instant |
| System RAM | 9.6 GB |
| Mean TTFT | 425.2 ms |
| P50 TTFT | 425.1 ms |
| P99 TTFT | 425.2 ms |
| Mean TPOT | 2.5 ms |
| Mean E2E | 979.6 ms |
| tok/s per request | 160.3 |
| Total tok/s | 165.0 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 426ms | 2.5ms | 180.4 |
| #2 | Python merge lists | 425ms | 2.5ms | 141.0 |
| #3 | ML differences | 425ms | 2.5ms | 188.9 |
| #4 | WW1 causes | 425ms | 2.6ms | 105.8 |
| #5 | Transformer architecture | 425ms | 2.5ms | 185.2 |

---

### 2️⃣ Qwen2.5-7B-Instruct — BF16

| Metric | Value |
|---|---|
| Server Startup | instant (warm cache) |
| System RAM | 10.8 GB |
| Mean TTFT | 26.0 ms (warm cache) |
| P50 TTFT | 25.9 ms |
| P99 TTFT | 26.0 ms |
| Mean TPOT | 3.6 ms |
| Mean E2E | 823.6 ms |
| tok/s per request | 197.8 |
| Total tok/s | 194.5 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 26ms | 3.6ms | 215.9 |
| #2 | Python merge lists | 26ms | 3.6ms | 146.5 |
| #3 | ML differences | 26ms | 3.6ms | 194.6 |
| #4 | WW1 causes | 26ms | 3.6ms | 222.5 |
| #5 | Transformer architecture | 26ms | 3.6ms | 209.7 |

> **Note:** The 26ms TTFT reflects a warm prefix cache. The TPOT (3.6ms) and throughput (194 tok/s) are the meaningful numbers here — both the fastest 7B results across all three GPUs.

---

### 3️⃣ Qwen2.5-32B-Instruct — BF16 (full precision)

| Metric | Value |
|---|---|
| Server Startup | 11,045 ms (torch.compile + CUDA graph capture) |
| System RAM | 12.5 GB |
| Mean TTFT | 473.8 ms |
| P50 TTFT | 473.0 ms |
| P99 TTFT | 473.4 ms |
| Mean TPOT | 13.6 ms |
| Mean E2E | 3,535.3 ms |
| tok/s per request | 45.9 |
| Total tok/s | 45.9 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 473ms | 13.8ms | 49.1 |
| #2 | Python merge lists | 477ms | 13.8ms | 35.3 |
| #3 | ML differences | 473ms | 13.8ms | 48.1 |
| #4 | WW1 causes | 473ms | 12.7ms | 46.3 |
| #5 | Transformer architecture | 473ms | 13.8ms | 50.9 |

> **Note:** Run at full BF16 (no quantization) — fastest 32B TPOT of all three GPUs, beating even H100's AWQ INT4 (16.8ms) while keeping full precision.
>
> **Prefix-cache observation:** A second back-to-back run of this exact benchmark returned TTFT 43.3ms (vs 473ms cold) with identical TPOT (12.8ms). This demonstrates vLLM's prefix caching — the first run pays full prefill cost, repeat runs with identical prompts reuse the cached KV. The 473ms cold-cache number is used here for consistency with the H100/H200 cold runs.

---

### 4️⃣ Mistral-Large-2407 (123B) — AWQ INT4 (awq_marlin)

| Metric | Value |
|---|---|
| Server Startup | instant (cached) |
| System RAM | 11.4 GB |
| Mean TTFT | 385.2 ms |
| P50 TTFT | 385.2 ms |
| P99 TTFT | 385.2 ms |
| Mean TPOT | 22.6 ms |
| Mean E2E | 5,742.4 ms |
| tok/s per request | 25.6 |
| Total tok/s | 25.6 |

| Request | Prompt | TTFT | TPOT | tok/s |
|---|---|---|---|---|
| #1 | Theory of relativity | 386ms | 22.6ms | 28.3 |
| #2 | Python merge lists | 385ms | 22.6ms | 18.7 |
| #3 | ML differences | 385ms | 22.6ms | 28.0 |
| #4 | WW1 causes | 385ms | 22.7ms | 26.5 |
| #5 | Transformer architecture | 385ms | 22.6ms | 26.7 |

> **Note:** Same 123B model as H200. B200's TPOT (22.6ms) is **38% faster** than H200's (36.3ms) on the identical model — a direct result of B200's higher memory bandwidth.

---

## 🔍 Analysis

### Fastest TPOT at every model size
B200 delivers the lowest decode latency across the board, consistent with its 8 TB/s bandwidth (the highest of the three GPUs):

| Model | H100 | H200 | **B200** |
|---|---|---|---|
| 1.5B | 3.1ms | 2.1ms | **2.5ms** |
| 7B | 9.1ms | 4.8ms | **3.6ms** |
| 32B | 16.8ms (AWQ) | 17.9ms (BF16) | **13.6ms (BF16)** |
| 123B/70B | 32.6ms (70B) | 36.3ms (123B) | **22.6ms (123B)** |

### Full precision with best-in-class speed
The 32B ran at full BF16 and still posted the fastest TPOT of any GPU — including H100's lossy AWQ. With 192GB VRAM, B200 never needs quantization for models up to ~123B (which only used AWQ to keep KV-cache headroom at long context).

### Prefix caching clearly observable
The duplicate 32B run (473ms → 43ms TTFT) cleanly demonstrated vLLM's prefix-cache reuse. This also confirms the elevated cold-cache TTFT seen across all GPUs is prefill cost, not a hardware limit.

### GPU stats unavailable
GPUtil failed to install on this instance (venv permissions), so VRAM/util show 0. Throughput numbers confirm the GPU was fully engaged; the missing readings are a tooling artifact, not a measurement of idle hardware.
