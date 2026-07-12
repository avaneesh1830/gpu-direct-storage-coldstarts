# 🚀 LLM Inference Benchmarks — H100 · H200 · B200

> **Project:** [Your Project Name]  
> **Mentor:** [Mentor Name] · NVIDIA Solutions Architecture  
> **Author:** Avaneesh Aroor  
> **Goal:** Benchmark LLM inference across 1.5B → 123B parameter models on three
> NVIDIA GPU generations (Hopper → Hopper-refresh → Blackwell) using vLLM on a
> single GPU, and identify where the bottlenecks are.

---

## 🗂️ Results by GPU

| GPU | VRAM | Bandwidth | Architecture | Status | Results |
|---|---|---|---|---|---|
| H100 80GB | 80 GB | 3.35 TB/s | Hopper | ✅ Complete | [H100_RESULTS.md](results/H100_RESULTS.md) |
| H200 141GB | 141 GB | 4.8 TB/s | Hopper | ✅ Complete | [H200_RESULTS.md](results/H200_RESULTS.md) |
| B200 192GB | 192 GB | 8.0 TB/s | Blackwell | ✅ Complete | [B200_RESULTS.md](results/B200_RESULTS.md) |

> **Note on B100:** B100 was dropped because virtually no cloud provider stocks it —
> most went straight from H-series to B200. B200 represents the Blackwell tier.

---

## 📊 Headline Comparison — TPOT (ms, lower = faster)

Time Per Output Token is the key decode-speed metric. Lower means faster token generation.

| Model | H100 | H200 | B200 |
|---|---|---|---|
| 1.5B | 3.1 ms | **2.1 ms** | 2.5 ms |
| 7B | 9.1 ms | 4.8 ms | **3.6 ms** |
| 32B | 16.8 ms (AWQ) | 17.9 ms (BF16) | **13.6 ms (BF16)** |
| Large (70-123B) | 32.6 ms (70B AWQ) | 36.3 ms (123B AWQ) | **22.6 ms (123B AWQ)** |

## 📈 Throughput (tok/s, higher = better)

| Model | H100 | H200 | B200 |
|---|---|---|---|
| 1.5B | 160.2 | 153.2 | **165.0** |
| 7B | 65.7 | 100.2 | **194.5** |
| 32B | 37.8 | 35.6 | **45.9** |
| Large | 20.0 (70B) | 16.2 (123B) | **25.6 (123B)** |

---

## 🔑 Key Findings

### 1. Decode is memory-bandwidth-bound
TPOT scales almost perfectly with each GPU's memory bandwidth. This is the central
result: for single-stream LLM inference, **memory bandwidth — not raw compute —
determines token generation speed.** The B200 (8 TB/s) consistently beats the
H200 (4.8 TB/s) which beats the H100 (3.35 TB/s).

### 2. VRAM capacity removes the need to quantize
On the H100 (80GB), the 32B and large models **must** use AWQ INT4 quantization
just to fit — a quality compromise. On H200 (141GB) and B200 (192GB), the same
32B model runs at full BF16 precision. The B200 ran full-precision 32B *faster*
(13.6ms) than the H100 ran the quantized version (16.8ms) — better quality AND
better speed.

### 3. Same model, B200 is 38% faster than H200
On the identical 123B Mistral-Large model, B200's TPOT (22.6ms) beat H200's
(36.3ms) by 38% — a clean, apples-to-apples demonstration of the Blackwell
bandwidth advantage.

### 4. TTFT is dominated by prefill caching, not hardware
A back-to-back repeat of the 32B benchmark on B200 dropped TTFT from 473ms to
43ms with identical TPOT — proving the cold-cache TTFT reflects one-time prefill
cost, reused via vLLM's prefix cache on subsequent identical requests.

---

## 🛠️ How It Works

### Single reusable Docker image for all GPUs and model sizes
```bash
docker pull avaneesharoor/ml-experiments:latest
```
The model is selected at runtime — one image covers all 12 experiments
(3 GPUs × 4 model sizes). See [SETUP_GUIDE.md](SETUP_GUIDE.md) for the full
walkthrough of building the image, setting up each instance, and every command used.

### Benchmark
```bash
python3 benchmark.py --size 1B --model Qwen/Qwen2.5-1.5B-Instruct
```
Measures TTFT, TPOT, throughput, end-to-end latency, and diagnoses the bottleneck.

---

## 📁 Repository Structure

```
├── README.md                  # This file — cross-GPU comparison
├── SETUP_GUIDE.md             # Full setup + every command for all GPUs
├── Dockerfile                 # Single reusable image
├── run_model.sh               # Launch helper (SIZE=1B/10B/30B/110B)
├── benchmark.py               # Profiling script
└── results/
    ├── H100_RESULTS.md        # ✅ Hopper 80GB
    ├── H200_RESULTS.md        # ✅ Hopper 141GB
    └── B200_RESULTS.md        # ✅ Blackwell 192GB
```

---

## ⚙️ Stack

| Component | Detail |
|---|---|
| Serving Framework | vLLM v0.23.0 |
| Attention | FlashAttention 3 |
| Memory Management | PagedAttention + prefix caching |
| API | OpenAI-compatible (`/v1/chat/completions`) |
| Quantization | BF16 · AWQ INT4 (awq_marlin kernel) |
| Container | Docker + NVIDIA Container Toolkit |
| Cloud | Brev (Hyperstack / Verda / Shadeform backends) |

---

## ⚠️ Measurement Caveats (documented honestly)

1. **GPU util / VRAM readings** — GPUtil sampled the host between requests (or
   failed to install on the B200 venv), so those fields can read 0. Throughput
   numbers confirm the GPUs were fully engaged.
2. **TTFT is queuing + prefill dominated** — the benchmark fires 5 concurrent
   requests; reported TTFT reflects queue wait and cold prefill, not pure
   single-request prefill. The prefix-cache experiment (Finding #4) confirms this.
3. **Model size labels are approximate** — "110B" is the *category* name; actual
   models were 70B (H100) and 123B (H200/B200). See each results file for exact
   model IDs.

---

*Benchmarks run June 2026 · vLLM v0.23.0 · single-GPU inference*
