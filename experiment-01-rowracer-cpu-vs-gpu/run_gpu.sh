#!/usr/bin/env bash
# Run on your NVIDIA L4 GPU instance
set -e

IMAGE="rowracer:gpu"

echo "=== Building GPU image ==="
docker build -f Dockerfile.gpu -t "$IMAGE" .

echo ""
echo "=== Running GPU benchmark ==="
docker run --rm \
  --gpus all \
  -v "$(pwd)/workspace:/workspace" \
  "$IMAGE" python3 benchmark.py --mode gpu
