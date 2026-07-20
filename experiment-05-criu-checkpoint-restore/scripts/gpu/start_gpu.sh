#!/bin/bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
setsid ~/venv/bin/python gpu_counter.py < /dev/null &> gpu.log &
echo "gpu model PID: $!"
