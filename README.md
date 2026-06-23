# gpu-direct-storage-coldstarts Project Overview

This project focuses on "Optimizing LLM serverless cold starts and inference runtimes using NVIDIA GPUDirect Storage (GDS), CRIU container snapshots, and CUDA Checkpoint/Restore."

## Key Initiative

The work delivers benchmarks and implementation strategies aimed at enabling swifter model serving alongside enhanced resource utilization.

## Research Roadmap

The 8-week tracker outlines progression through several technical domains:

- **Weeks 1-2:** Establishing baselines by surveying the NVIDIA technology ecosystem and measuring performance metrics across language models ranging from 8 billion to 120 billion parameters
- **Weeks 3-5:** Evaluating InstantTensor performance across different GPU architectures and PCIe versions, then exploring checkpoint/restore mechanisms via CRIU and CUDA tools
- **Weeks 6-8:** Advancing integration efforts with Dynamo snapshots, machine learning frameworks (vLLM), and exploring accelerated data processing through CuML/CuDF libraries

The initiative systematically addresses infrastructure optimization for resource-constrained deployment scenarios.

## Experiments

| # | Name | Description | Result |
|---|------|-------------|--------|
| 01 | [RowRacer — CPU vs GPU CSV Benchmark](./experiment-01-rowracer-cpu-vs-gpu/) | pandas (8-core CPU) vs cuDF/RAPIDS (NVIDIA L4) on 1M and 10M row datasets | GPU 24x faster on 10M rows |
