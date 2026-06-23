#!/usr/bin/env python3
"""
RowRacer — CPU vs GPU Benchmark: CSV Data Processing
pandas (CPU, 8-core) vs cuDF/RAPIDS (L4 GPU)

Usage:
  python benchmark.py --mode cpu      # run on CPU instance  → results_cpu.json
  python benchmark.py --mode gpu      # run on GPU instance  → results_gpu.json
  python benchmark.py --compare       # merge both JSONs and print comparison table

NOTE: Run generate_data.py ONCE before benchmarking to create the CSV files.
      Copy the CSV files to both instances so neither benchmark wastes CPU on generation.
"""

import argparse
import time
import os
import sys
import json
import platform
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

DATA_DIR      = Path("/workspace/data")
CPU_RESULTS   = Path("/workspace/results_cpu.json")
GPU_RESULTS   = Path("/workspace/results_gpu.json")

CATEGORIES = ["Electronics", "Clothing", "Food", "Books", "Sports"]
REGIONS    = ["North", "South", "East", "West", "Central"]
OPS        = ["read_csv", "filter", "groupby", "sort", "rolling_mean", "multi_groupby", "TOTAL"]

# ── Data generation ───────────────────────────────────────────────────────────

def generate_csv(n_rows: int, path: Path) -> None:
    if path.exists():
        mb = path.stat().st_size / 1_048_576
        print(f"  [skip] {path.name} already exists ({mb:.0f} MB)")
        return

    print(f"  Generating {n_rows:,} rows → {path.name} ...")
    rng = np.random.default_rng(42)
    chunk_size = 1_000_000
    first = True

    for start in range(0, n_rows, chunk_size):
        end  = min(start + chunk_size, n_rows)
        size = end - start

        chunk = pd.DataFrame({
            "id":        np.arange(start, end, dtype=np.int32),
            "category":  np.random.choice(CATEGORIES, size),
            "region":    np.random.choice(REGIONS,    size),
            "value1":    rng.uniform(0,    1000, size).round(2),
            "value2":    rng.uniform(0,     500, size).round(2),
            "value3":    rng.uniform(-100,  100, size).round(4),
            "quantity":  rng.integers(1, 100, size),
            "revenue":   (rng.uniform(0, 1000, size) * rng.integers(1, 100, size)).round(2),
            "timestamp": 1_700_000_000 + rng.integers(0, 86_400 * 365, size),
        })
        chunk.to_csv(path, mode="a" if not first else "w", header=first, index=False)
        first = False
        print(f"    {end:>10,} / {n_rows:,} rows written", end="\r")

    mb = path.stat().st_size / 1_048_576
    print(f"\n  Done — {mb:.1f} MB")


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def _t() -> float:
    return time.perf_counter()


def warmup_gpu() -> None:
    dummy = cudf.DataFrame({"x": list(range(10_000))})
    _ = dummy["x"].mean()
    print("  GPU context warmed up")


# ── CPU benchmark (pandas) ────────────────────────────────────────────────────

def bench_cpu(path: Path, label: str) -> dict:
    res = {}
    print(f"\n  [CPU / pandas]  {label}")

    t = _t(); df = pd.read_csv(path)
    res["read_csv"] = _t() - t
    print(f"    read_csv       {res['read_csv']:>8.3f}s   rows={len(df):,}")

    t = _t(); filt = df[df["value1"] > 500.0]; _ = len(filt)
    res["filter"] = _t() - t
    print(f"    filter         {res['filter']:>8.3f}s   rows={len(filt):,}")

    t = _t()
    grp = df.groupby("category").agg(
        mean_v1=("value1",  "mean"),
        sum_rev=("revenue", "sum"),
        cnt    =("id",      "count"),
        max_v3 =("value3",  "max"),
    ).reset_index(); _ = len(grp)
    res["groupby"] = _t() - t
    print(f"    groupby        {res['groupby']:>8.3f}s")

    t = _t(); srt = df.sort_values("value1", ascending=False); _ = len(srt)
    res["sort"] = _t() - t
    print(f"    sort           {res['sort']:>8.3f}s")

    t = _t(); roll = df["value1"].rolling(100).mean(); _ = roll.iloc[-1]
    res["rolling_mean"] = _t() - t
    print(f"    rolling_mean   {res['rolling_mean']:>8.3f}s")

    t = _t()
    mg = df.groupby(["category", "region"]).agg(
        mean_v1=("value1",   "mean"),
        std_v1 =("value1",   "std"),
        sum_v2 =("value2",   "sum"),
        sum_qty=("quantity", "sum"),
    ); _ = len(mg)
    res["multi_groupby"] = _t() - t
    print(f"    multi_groupby  {res['multi_groupby']:>8.3f}s")

    res["TOTAL"] = sum(res.values())
    print(f"    {'─'*35}")
    print(f"    TOTAL          {res['TOTAL']:>8.3f}s")
    return res


# ── GPU benchmark (cuDF) ──────────────────────────────────────────────────────

def bench_gpu(path: Path, label: str) -> dict:
    res = {}
    print(f"\n  [GPU / cuDF]    {label}")

    t = _t(); df = cudf.read_csv(str(path))
    res["read_csv"] = _t() - t
    print(f"    read_csv       {res['read_csv']:>8.3f}s   rows={len(df):,}")

    t = _t(); filt = df[df["value1"] > 500.0]; _ = len(filt)
    res["filter"] = _t() - t
    print(f"    filter         {res['filter']:>8.3f}s   rows={len(filt):,}")

    t = _t()
    grp = df.groupby("category").agg(
        mean_v1=("value1",  "mean"),
        sum_rev=("revenue", "sum"),
        cnt    =("id",      "count"),
        max_v3 =("value3",  "max"),
    ).reset_index(); _ = len(grp)
    res["groupby"] = _t() - t
    print(f"    groupby        {res['groupby']:>8.3f}s")

    t = _t(); srt = df.sort_values("value1", ascending=False); _ = len(srt)
    res["sort"] = _t() - t
    print(f"    sort           {res['sort']:>8.3f}s")

    t = _t(); roll = df["value1"].rolling(100).mean(); _ = roll.iloc[-1]
    res["rolling_mean"] = _t() - t
    print(f"    rolling_mean   {res['rolling_mean']:>8.3f}s")

    t = _t()
    mg = df.groupby(["category", "region"]).agg(
        mean_v1=("value1",   "mean"),
        std_v1 =("value1",   "std"),
        sum_v2 =("value2",   "sum"),
        sum_qty=("quantity", "sum"),
    ); _ = len(mg)
    res["multi_groupby"] = _t() - t
    print(f"    multi_groupby  {res['multi_groupby']:>8.3f}s")

    res["TOTAL"] = sum(res.values())
    print(f"    {'─'*35}")
    print(f"    TOTAL          {res['TOTAL']:>8.3f}s")
    return res


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_single(results: dict, mode: str, label: str) -> None:
    tag = "CPU / pandas" if mode == "cpu" else "GPU / cuDF"
    w = 45
    print(f"\n{'═'*w}")
    print(f"  {tag} — {label}")
    print(f"{'═'*w}")
    print(f"  {'Operation':<18}  {'Time (s)':>9}")
    print(f"  {'─'*30}")
    for op in OPS:
        if op == "TOTAL":
            print(f"  {'─'*30}")
        print(f"  {op:<18}  {results.get(op, 0):>9.3f}")
    print(f"{'═'*w}")


def print_comparison(cpu: dict, gpu: dict, label: str) -> None:
    w = 65
    print(f"\n{'═'*w}")
    print(f"  RowRacer Results — {label}")
    print(f"  CPU: 8-core instance   GPU: NVIDIA L4")
    print(f"{'═'*w}")
    print(f"  {'Operation':<18}  {'CPU (s)':>9}  {'GPU (s)':>9}  {'Speedup':>9}")
    print(f"  {'─'*59}")
    for op in OPS:
        if op == "TOTAL":
            print(f"  {'─'*59}")
        c = cpu.get(op, 0)
        g = gpu.get(op, 0)
        spd = f"{c/g:.1f}x" if g > 0 else "N/A"
        tag = "  GPU wins" if g > 0 and c / g > 1.5 else ("  CPU wins" if g > 0 and g / c > 1.5 else "")
        print(f"  {op:<18}  {c:>9.3f}  {g:>9.3f}  {spd:>9}{tag}")
    print(f"{'═'*w}")


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_cpu_mode() -> None:
    w = 65
    print(f"\n{'═'*w}")
    print("  RowRacer — CPU Benchmark (pandas)")
    print(f"  Host   : {platform.node()}")
    print(f"  Python : {sys.version.split()[0]}   pandas : {pd.__version__}")
    cpu_info = subprocess.run(["nproc"], capture_output=True, text=True)
    print(f"  Cores  : {cpu_info.stdout.strip()}")
    print(f"{'═'*w}")

    datasets = {
        "1M rows":  (1_000_000,  DATA_DIR / "data_1M.csv"),
        "10M rows": (10_000_000, DATA_DIR / "data_10M.csv"),
    }

    # Verify data exists — run generate_data.py first if missing
    for label, (n, path) in datasets.items():
        if not path.exists():
            print(f"ERROR: {path} not found.")
            print("  Run generate_data.py first, then copy workspace/data/ to this instance.")
            sys.exit(1)
        mb = path.stat().st_size / 1_048_576
        print(f"  {label}: {path.name} ({mb:.0f} MB) ready")

    all_results = {}
    for label, (n, path) in datasets.items():
        print(f"\n{'─'*w}\n  DATASET: {label}  ({n:,} rows)\n{'─'*w}")
        r = bench_cpu(path, label)
        print_single(r, "cpu", label)
        all_results[label] = r

    with open(CPU_RESULTS, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {CPU_RESULTS}")
    print("  Copy this file to the GPU instance and run --compare\n")


def run_gpu_mode() -> None:
    if not GPU_AVAILABLE:
        print("ERROR: cuDF not found. Is this the GPU instance?")
        sys.exit(1)

    w = 65
    print(f"\n{'═'*w}")
    print("  RowRacer — GPU Benchmark (cuDF / RAPIDS)")
    print(f"  Host   : {platform.node()}")
    print(f"  Python : {sys.version.split()[0]}   cuDF : {cudf.__version__}")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        print(f"  GPU    : {out.stdout.strip()}")
    except Exception:
        pass
    print(f"{'═'*w}")

    datasets = {
        "1M rows":  (1_000_000,  DATA_DIR / "data_1M.csv"),
        "10M rows": (10_000_000, DATA_DIR / "data_10M.csv"),
    }

    # Verify data exists — copy workspace/data/ from CPU instance if missing
    for label, (n, path) in datasets.items():
        if not path.exists():
            print(f"ERROR: {path} not found.")
            print("  Copy workspace/data/ from the CPU instance to this machine first.")
            sys.exit(1)
        mb = path.stat().st_size / 1_048_576
        print(f"  {label}: {path.name} ({mb:.0f} MB) ready")

    print(f"\n{'─'*w}\n  Warming up GPU\n{'─'*w}")
    warmup_gpu()

    all_results = {}
    for label, (n, path) in datasets.items():
        print(f"\n{'─'*w}\n  DATASET: {label}  ({n:,} rows)\n{'─'*w}")
        r = bench_gpu(path, label)
        print_single(r, "gpu", label)
        all_results[label] = r

    with open(GPU_RESULTS, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {GPU_RESULTS}\n")


def run_compare_mode() -> None:
    if not CPU_RESULTS.exists():
        print(f"ERROR: {CPU_RESULTS} not found.")
        print("  Copy results_cpu.json from the CPU instance to /workspace/ on this machine.")
        sys.exit(1)
    if not GPU_RESULTS.exists():
        print(f"ERROR: {GPU_RESULTS} not found. Run --mode gpu first.")
        sys.exit(1)

    cpu_data = json.loads(CPU_RESULTS.read_text())
    gpu_data = json.loads(GPU_RESULTS.read_text())

    for label in cpu_data:
        if label in gpu_data:
            print_comparison(cpu_data[label], gpu_data[label], label)
        else:
            print(f"  WARNING: {label} missing from GPU results, skipping")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RowRacer — CPU vs GPU CSV Benchmark")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode", choices=["cpu", "gpu"],
                       help="Run benchmark on this instance (cpu or gpu)")
    group.add_argument("--compare", action="store_true",
                       help="Compare results_cpu.json vs results_gpu.json")
    args = parser.parse_args()

    if args.compare:
        run_compare_mode()
    elif args.mode == "cpu":
        run_cpu_mode()
    else:
        run_gpu_mode()


if __name__ == "__main__":
    main()
