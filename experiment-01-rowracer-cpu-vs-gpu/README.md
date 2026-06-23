# Experiment 01 — RowRacer: CPU vs GPU CSV Benchmark

## Why I did this

This project is part of a broader research initiative on optimizing LLM inference and cold starts using NVIDIA GPU acceleration. Before diving into GPU-specific tools like GPUDirect Storage, CRIU snapshots, and CuML, I needed a clear baseline that answers a simple question:

**How much faster is a GPU at processing data compared to a CPU — and does it depend on data size?**

This experiment establishes that baseline. It uses real compute instances (8-core CPU vs NVIDIA L4 GPU) running identical data processing workloads, so every subsequent experiment in this series has a concrete reference point to compare against.

## What I built

A benchmark called **RowRacer** that runs 6 data operations on 1M and 10M row CSV files, comparing:
- **CPU**: 8-core instance, 32 GB RAM — pandas 3.0.3
- **GPU**: NVIDIA L4 (23 GB VRAM) — cuDF 26.06.00 / RAPIDS

Both instances used the same Docker base image (`avaneesharoor/ml-experiments`) and the same pre-generated CSV data to ensure a fair comparison.

## Results

### 1M rows

| Operation     | CPU (s) | GPU (s) | Speedup  |
|--------------|---------|---------|----------|
| read_csv     | 0.596   | 0.140   | 4.3x     |
| filter       | 0.042   | 0.014   | 3.0x     |
| groupby      | 0.040   | 0.038   | 1.1x     |
| sort         | 0.179   | 0.022   | 8.1x     |
| rolling_mean | 0.013   | 0.066   | 0.2x — CPU wins |
| multi_groupby| 0.062   | 0.005   | 12.7x    |
| **TOTAL**    | **0.932** | **0.284** | **3.3x** |

### 10M rows

| Operation     | CPU (s) | GPU (s) | Speedup  |
|--------------|---------|---------|----------|
| read_csv     | 5.768   | 0.260   | 22.2x    |
| filter       | 0.314   | 0.018   | 17.1x    |
| groupby      | 0.313   | 0.015   | 21.4x    |
| sort         | 3.346   | 0.099   | **33.7x**|
| rolling_mean | 0.128   | 0.018   | 7.3x     |
| multi_groupby| 0.607   | 0.026   | 23.3x    |
| **TOTAL**    | **10.477** | **0.436** | **24x** |

## Key findings

1. **GPU wins at scale** — 3.3x faster at 1M rows, 24x faster at 10M rows. The speedup is not constant; it grows with data size.
2. **Sort is where GPU dominates most** — 33.7x speedup. GPU parallel merge sort is extremely efficient on large arrays.
3. **Rolling mean is the exception** — at 1M rows, CPU is 5x faster. Rolling windows are sequential by nature (each value depends on the previous N rows), so GPU parallelism doesn't help much at small scale.
4. **VRAM transfer overhead is real** — at 1M rows some GPU times are slower because loading data into GPU memory costs more than the speedup gained. At 10M rows this cost becomes negligible.

## How to run

```bash
# Step 1 — generate CSV data (run once on CPU instance)
bash generate_data.sh

# Step 2 — copy CSVs to GPU instance
scp -r workspace/data ubuntu@<gpu-ip>:~/<project-folder>/workspace/

# Step 3 — run benchmarks (run in parallel on both instances)
bash run_cpu.sh    # on CPU instance
bash run_gpu.sh    # on GPU instance

# Step 4 — copy CPU results to GPU instance and compare
scp ubuntu@<cpu-ip>:~/<project-folder>/workspace/results_cpu.json \
    ubuntu@<gpu-ip>:~/<project-folder>/workspace/
bash compare.sh    # on GPU instance
```

## Infrastructure

- Cloud: Brev.nvidia (AWS)
- CPU instance: 8 cores, 32 GB RAM
- GPU instance: NVIDIA L4, 23 GB VRAM, 4 cores, 16 GB RAM
- Container: Docker with `avaneesharoor/ml-experiments:latest` as base image
