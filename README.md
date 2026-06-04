# gpu-direct-storage-coldstarts

Optimizing LLM serverless cold starts and inference runtimes using NVIDIA GPUDirect Storage (GDS), CRIU container snapshots, and CUDA Checkpoint/Restore.

Benchmarks and implementation for faster serving & improved utilization.

# Project Tracker

| Week | Topic                                                           | Status | Remarks |
| ---- | --------------------------------------------------------------- | ------ | ------- |
| 1    | NV Stack Overview                                               | 🚧     |         |
| 2    | Baselines for LLMs & Diffusion models[8b/30b/120b]              |        |         |
| 3    | InstantTensor benchmarking across GPU SKUs & PCIe generations   |        |         |
| 4    | Ecosystem for container checkpoint/restore                      |        |         |
| 5    | CRIU & CUDA-checkpointing                                       |        |         |
| 6    | Dynamo Snapshot                                                 |        |         |
| 7    | Integrating InstantTensor to SafeTensor loader/vLLM Omni        |        |         |
| 8    | Misc CuML/CuDF exploration for out-of-core execution & speedups |        |         |
