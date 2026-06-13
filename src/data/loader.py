"""
src/data/loader.py
──────────────────
Thin wrapper to load the parquet file with correct dtypes.
All models (LGBM, DLinear, Mamba, KAN) use this as their data entry point.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PARQUET = Path("data/processed/wind_data.parquet")

RAW_DTYPES = {
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


def load_raw(path: Path = DEFAULT_PARQUET) -> pd.DataFrame:
    """
    Load the base parquet (raw meteorological columns only — no features).
    Returns a DataFrame sorted by [Index, datetime] with the datetime column parsed.

    Expected columns:
        datetime, Index, Latitude, Longitude,
        YEAR, MONTH, DAY, HOUR,
        wind_speed, wind_direction,
        temperature, humidity, surface_pressure
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Parquet not found at '{path}'.\n"
            "Run:  python scripts/convert_to_parquet.py  first, "
            "or pull from git (file is committed at data/processed/wind_data.parquet)."
        )

    df = pd.read_parquet(path)

    # Ensure datetime is parsed
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Enforce memory-efficient dtypes
    for col, dtype in RAW_DTYPES.items():
        if col in df.columns and df[col].dtype.name != dtype:
            df[col] = df[col].astype(dtype)

    df = df.sort_values(["Index", "datetime"]).reset_index(drop=True)

    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols | "
          f"{df['datetime'].min().date()} → {df['datetime'].max().date()} | "
          f"{df['Index'].nunique()} stations")
    return df
