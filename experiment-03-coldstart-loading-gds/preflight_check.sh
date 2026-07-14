#!/usr/bin/env bash
# Preflight check for a fresh GPU instance before running the cold-start /
# InstantTensor(fastsafetensors) / GDS experiment.
# Usage: ./preflight_check.sh
set -uo pipefail

pass(){ echo "  [PASS] $1"; }
warn(){ echo "  [WARN] $1"; }
fail(){ echo "  [FAIL] $1"; }

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current \
  --format=csv,noheader 2>/dev/null && pass "nvidia-smi OK" || fail "nvidia-smi not found / GPU not visible"

echo
echo "=== System RAM (110B needs ~65GB in page cache for the warm run) ==="
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo "  Total RAM: ${RAM_GB} GB"
if   [ "$RAM_GB" -ge 128 ]; then pass "RAM ${RAM_GB}GB — safe for 110B warm-cache"
elif [ "$RAM_GB" -ge 80  ]; then warn "RAM ${RAM_GB}GB — 110B warm run may be tight"
else fail "RAM ${RAM_GB}GB — 110B (65GB) will not reliably stay cached"
fi

echo
echo "=== Root disk device / storage type ==="
lsblk -d -o NAME,SIZE,ROTA,MODEL 2>/dev/null
ROOT_DEV=$(df / | tail -1 | awk '{print $1}' | sed -E 's#/dev/##; s/p?[0-9]+$//')
echo "  Root device: $ROOT_DEV"
case "$ROOT_DEV" in
  vd*)   warn "Root disk is virtio (vd*) — almost certainly network storage, expect slow reads (~0.1 GB/s)" ;;
  nvme*) pass "Root disk is nvme* — could be EBS-over-NVMe (network) or local; verify with dd below" ;;
  sd*)   warn "Root disk is sd*/SCSI-emulated — often virtualized, verify speed with dd" ;;
  *)     warn "Unrecognized root device type: $ROOT_DEV" ;;
esac

echo
echo "=== Disk free space (need ~200GB+ for all 4 models + 2 docker images) ==="
df -h / | tail -1
FREE_GB=$(df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "$FREE_GB" ] && [ "$FREE_GB" -ge 200 ]; then pass "Free space ${FREE_GB}GB"
else warn "Free space ${FREE_GB:-unknown}GB — may run out mid-experiment"
fi

echo
echo "=== Disk speed: 1GB direct write + cold read ==="
TESTFILE=~/.preflight_disktest
dd if=/dev/zero of="$TESTFILE" bs=1M count=1024 oflag=direct 2>&1 | tail -1
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || warn "could not drop page cache (no sudo?) — read test below may be from cache"
dd if="$TESTFILE" of=/dev/null bs=1M iflag=direct 2>&1 | tail -1
rm -f "$TESTFILE"

echo
echo "=== Local NVMe / instance-store check ==="
lsblk -o NAME,SIZE,MOUNTPOINT 2>/dev/null | grep -i nvme || echo "  (no nvme* block devices found)"
mount | grep -iE "nvme|ephemeral|instance-store|local-ssd" || echo "  (no local-storage mount points detected)"

echo
echo "=== GPU Direct Storage (GDS) readiness ==="
GDSCHECK=$(command -v gdscheck 2>/dev/null || echo /usr/local/cuda/gds/tools/gdscheck)
if [ -x "$GDSCHECK" ]; then
  "$GDSCHECK" -p 2>&1 | grep -E "NVMe|use_pci_p2pdma|nvidia_fs version"
  if "$GDSCHECK" -p 2>&1 | grep -q "use_pci_p2pdma : true"; then
    pass "PCIe P2P DMA enabled — real GDS should work here"
  else
    fail "use_pci_p2pdma : false — GDS will silently disable (nogds=True). Needs bare-metal / P2P-passthrough instance."
  fi
else
  warn "gdscheck not found on this image — GDS tools not installed (install cuda-gds or use the NGC vLLM image)"
fi
lsmod 2>/dev/null | grep -q nvidia_fs && pass "nvidia_fs kernel module loaded" || warn "nvidia_fs kernel module NOT loaded"

echo
echo "=== Docker + NVIDIA Container Toolkit ==="
docker run --rm --gpus all ubuntu nvidia-smi >/dev/null 2>&1 \
  && pass "GPU visible inside containers" || fail "Docker GPU passthrough not working — install nvidia-container-toolkit"

echo
echo "=== SUMMARY ==="
echo "Review any [WARN]/[FAIL] lines above before starting the full benchmark loop."
echo "Go/no-go for GDS specifically: need [PASS] on 'PCIe P2P DMA enabled' above."
