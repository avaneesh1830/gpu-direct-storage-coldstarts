#!/bin/bash
# Usage: SIZE=1B ./run_model.sh
# Sizes: 1B, 10B, 30B, 110B

HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"
IMAGE="${IMAGE:-avaneesharoor/ml-experiments:latest}"
SIZE="${SIZE:-1B}"
MODEL_CACHE="${MODEL_CACHE:-/ephemeral/model-cache}"
mkdir -p "$MODEL_CACHE"

case "$SIZE" in
  1B)
    MODEL="Qwen/Qwen2.5-1.5B-Instruct"
    QUANT_FLAGS=""
    DTYPE="bfloat16"
    GPU_UTIL="0.90"
    MAX_LEN="8192"
    ;;
  10B)
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    QUANT_FLAGS=""
    DTYPE="bfloat16"
    GPU_UTIL="0.90"
    MAX_LEN="8192"
    ;;
  30B)
    MODEL="Qwen/Qwen2.5-32B-Instruct-AWQ"
    QUANT_FLAGS="--quantization awq_marlin"
    DTYPE="float16"
    GPU_UTIL="0.90"
    MAX_LEN="8192"
    ;;
  110B)
    MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
    QUANT_FLAGS="--quantization awq_marlin"
    DTYPE="float16"
    GPU_UTIL="0.92"
    MAX_LEN="4096"
    ;;
  *)
    echo "Unknown SIZE: $SIZE. Use 1B, 10B, 30B, or 110B."
    exit 1
    ;;
esac

echo "Starting vLLM: $MODEL (SIZE=$SIZE)"

docker run --rm \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -v "$MODEL_CACHE":/model-cache \
  -p 8000:8000 \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --entrypoint python3 \
  "$IMAGE" \
  -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype "$DTYPE" \
  $QUANT_FLAGS \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN"
