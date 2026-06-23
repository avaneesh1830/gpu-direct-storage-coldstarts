#!/usr/bin/env bash
# Run on your 8-core CPU instance
set -e

IMAGE="rowracer:cpu"

echo "=== Building CPU image ==="
docker build -f Dockerfile.cpu -t "$IMAGE" .

echo ""
echo "=== Running CPU benchmark ==="
docker run --rm \
  -v "$(pwd)/workspace:/workspace" \
  "$IMAGE" python3 benchmark.py --mode cpu

echo ""
echo "=== Done — copy workspace/results_cpu.json to the GPU instance ==="
