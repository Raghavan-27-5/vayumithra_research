#!/usr/bin/env python3
"""
convert_to_parquet.py
─────────────────────
Converts the raw wind dataset CSV → compressed Parquet.

Usage (from repo root):
    python scripts/convert_to_parquet.py \
        --input  "data/raw/data (1).csv" \
        --output  data/processed/wind_data.parquet

The Parquet file (~66 MB vs 206 MB CSV) is gitignored.
Copy it to the remote desktop alongside the repo.

Output schema (all dtypes cast for memory efficiency):
    datetime         : datetime64[us]
    Index            : int16
    Latitude         : float32
    Longitude        : float32
    wind_speed       : float32
    wind_direction   : float32
    temperature      : float32
    humidity         : float32
    surface_pressure : float32
    YEAR             : int16
    MONTH            : int8
    DAY              : int8o
    HOUR             : int8
"""

import argparse
import time
from pathlib import Path

import pandas as pd

# ── Column dtypes ─────────────────────────────────────────────────────────────
DTYPES = {
    "YEAR":             "int16",
    "MONTH":            "int8",
    "DAY":              "int8",
    "HOUR":             "int8",
    "humidity":         "float32",
    "temperature":      "float32",
    "surface_pressure": "float32",
    "wind_speed":       "float32",
    "wind_direction":   "float32",
    "Longitude":        "float32",
    "Latitude":         "float32",
    "Index":            "int16",
}


def convert(input_path: Path, output_path: Path, chunksize: int = 500_000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading : {input_path}  ({input_path.stat().st_size / 1e6:.1f} MB)")
    t0 = time.time()

    # Read full CSV (206 MB fits in RAM easily even on the fragile machine)
    df = pd.read_csv(
        input_path,
        dtype=DTYPES,
        parse_dates=["datetime"],
    )

    print(f"  Shape : {df.shape}")
    print(f"  Date  : {df['datetime'].min()}  →  {df['datetime'].max()}")
    print(f"  Stations: {df['Index'].nunique()}")

    # ── Verify no duplicate (datetime, Index) pairs ───────────────────────────
    dupes = df.duplicated(subset=["datetime", "Index"]).sum()
    print(f"  Duplicate (datetime, Index) pairs: {dupes}")

    # ── Sort for downstream temporal ops ─────────────────────────────────────
    df = df.sort_values(["Index", "datetime"]).reset_index(drop=True)

    # ── Save to parquet ───────────────────────────────────────────────────────
    df.to_parquet(output_path, index=False, compression="snappy", engine="pyarrow")

    out_mb = output_path.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"\n✅ Saved  : {output_path}  ({out_mb:.1f} MB)  [{elapsed:.1f}s]")
    print(f"   Compression ratio: {input_path.stat().st_size / output_path.stat().st_size:.1f}×")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert wind CSV → Parquet")
    parser.add_argument("--input",  type=Path, default=Path('data/raw/data (1).csv'),
                        help="Path to the raw CSV file")
    parser.add_argument("--output", type=Path, default=Path("data/processed/wind_data.parquet"),
                        help="Output parquet path")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input not found: {args.input}\n"
            f"  Place the extracted CSV at: {args.input}"
        )

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
