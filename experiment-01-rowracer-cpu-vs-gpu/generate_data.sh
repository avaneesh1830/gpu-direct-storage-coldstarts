#!/usr/bin/env bash
# Run ONCE on the CPU instance to generate both CSV files
# Then copy workspace/data/ to the GPU instance before benchmarking
set -e

IMAGE="rowracer:cpu"

echo "=== Building image ==="
docker build -f Dockerfile.cpu -t "$IMAGE" .

echo ""
echo "=== Generating 1M and 10M row CSV files ==="
docker run --rm \
  -v "$(pwd)/workspace:/workspace" \
  "$IMAGE" python3 generate_data.py

echo ""
echo "=== Done ==="
echo "Now copy workspace/data/ to the GPU instance:"
echo "  scp -r workspace/data ubuntu@<gpu-ip>:~/bench/workspace/"
