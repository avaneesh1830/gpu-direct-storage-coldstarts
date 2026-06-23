#!/usr/bin/env python3
"""
RowRacer — Data Generator
Run this ONCE on the CPU instance to create the CSV files.
Then copy the CSVs to the GPU instance before benchmarking.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path("/workspace/data")
CATEGORIES = ["Electronics", "Clothing", "Food", "Books", "Sports"]
REGIONS    = ["North", "South", "East", "West", "Central"]


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


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== RowRacer — Generating Test Data ===\n")
    generate_csv(1_000_000,  DATA_DIR / "data_1M.csv")
    generate_csv(10_000_000, DATA_DIR / "data_10M.csv")
    print("\nDone. Copy workspace/data/ to the GPU instance before benchmarking.\n")
