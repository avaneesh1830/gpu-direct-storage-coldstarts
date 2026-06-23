#!/usr/bin/env bash
# Run on the GPU instance AFTER copying results_cpu.json from the CPU instance
# Usage: scp user@<cpu-ip>:~/bench/workspace/results_cpu.json ~/bench/workspace/
#        bash compare.sh

set -e

IMAGE="rowracer:gpu"

echo "=== RowRacer — Final Comparison ==="
docker run --rm \
  -v "$(pwd)/workspace:/workspace" \
  "$IMAGE" python3 benchmark.py --compare
