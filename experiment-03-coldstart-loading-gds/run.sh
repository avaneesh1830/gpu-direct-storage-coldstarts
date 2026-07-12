#!/usr/bin/env bash
# Cold-start protocol for one model size on the current GPU instance.
#
# Baseline:      SIZE=1b ./run.sh
# GDS + eager:   CACHE=~/model-cache-gds PREFIX=gdseager_ \
#                EXTRA="--enforce-eager --load-format fastsafetensors" \
#                IMAGE=nvcr.io/nvidia/vllm:<tag> SIZE=1b ./run.sh
#
#   CACHE   - weights dir; point at an EMPTY dir to force a fresh download
#   PREFIX  - prepended to every label so configs don't collide in results.jsonl
#   EXTRA   - extra benchmark.py flags applied to ALL load steps
#   IMAGE   - docker image (default: our custom image)
# Needs: docker + nvidia toolkit, sudo (page-cache drops), HF_TOKEN exported.
set -e

SIZE="${SIZE:-1b}"
IMAGE="${IMAGE:-avaneesharoor/ml-experiments:latest}"
CACHE="${CACHE:-$HOME/model-cache}"
PREFIX="${PREFIX:-}"
EXTRA="${EXTRA:-}"
mkdir -p "$CACHE" "$HOME/results"

# Mount the cache at BOTH known HF locations so the same host dir works with
# our image (HF_HOME=/model-cache) and the NGC image (~/.cache/huggingface).
bench() {
  docker run --rm --runtime nvidia --gpus all --ipc=host \
    --entrypoint python3 \
    -e HF_TOKEN \
    -v "$CACHE":/model-cache \
    -v "$CACHE":/root/.cache/huggingface \
    -v "$HOME/benchmark.py":/tmp/benchmark.py \
    -v "$HOME/results":/tmp/results \
    "$IMAGE" /tmp/benchmark.py --model "$SIZE" "$@"
}

drop_caches() {
  sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  echo "[run.sh] page cache dropped"
}

echo "=== [$SIZE] step 1: first load (downloads if not cached) ==="
bench --mode load --label "${PREFIX}first_download" $EXTRA

echo "=== [$SIZE] step 2: COLD load (page cache dropped) ==="
drop_caches
bench --mode load --label "${PREFIX}cold" $EXTRA

echo "=== [$SIZE] step 3: WARM load (weights in page cache) ==="
bench --mode load --label "${PREFIX}warm" $EXTRA

echo "=== [$SIZE] step 4: raw disk read (cold) ==="
drop_caches
bench --mode disk --label "${PREFIX}disk_cold"

echo "=== [$SIZE] done. results in ~/results/results.jsonl ==="
