#!/usr/bin/env python3
"""
scripts/compare_models.py
──────────────────────────
Aggregates all results CSVs (LGBM, DLinear, Mamba, hybrids) into
a unified benchmark table. Run after all training is complete.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --output results/metrics/benchmark_table.csv
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METRICS_DIR = Path("results/metrics")

RESULT_FILES = {
    "lgbm":    METRICS_DIR / "lgbm_results.csv",
    "dlinear": METRICS_DIR / "dlinear_results.csv",
    "mamba":   METRICS_DIR / "mamba_results.csv",
    "hybrid":  METRICS_DIR / "hybrid_results.csv",
}


def load_results() -> pd.DataFrame:
    frames = []
    for model_name, path in RESULT_FILES.items():
        if path.exists():
            df = pd.read_csv(path)
            if "model" not in df.columns:
                df["model"] = model_name
            frames.append(df)
        else:
            print(f"  [SKIP] {path} not found (model not yet trained)")

    if not frames:
        raise FileNotFoundError("No result CSVs found. Run training scripts first.")

    return pd.concat(frames, ignore_index=True)


def make_benchmark_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot: rows = (model, horizon), cols = (fold1_mae, fold2_mae, fold3_mae, mean_mae, std_mae, mean_r2)
    """
    summary = (
        df.groupby(["model", "horizon"])
        .agg(
            mae_fold1=("mae",  lambda x: x.iloc[0] if len(x) > 0 else np.nan),
            mae_fold2=("mae",  lambda x: x.iloc[1] if len(x) > 1 else np.nan),
            mae_fold3=("mae",  lambda x: x.iloc[2] if len(x) > 2 else np.nan),
            mae_mean=("mae",   "mean"),
            mae_std=("mae",    "std"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2",    "mean"),
            r2_std=("r2",     "std"),
        )
        .reset_index()
        .sort_values(["horizon", "mae_mean"])
    )
    return summary


def print_selection_memo(df: pd.DataFrame) -> None:
    """Print the final model selection memo."""
    print("\n" + "="*70)
    print("MODEL SELECTION MEMO")
    print("="*70)

    for h, hdf in df.groupby("horizon"):
        best = hdf.loc[hdf["mae_mean"].idxmin()]
        second = hdf.nsmallest(2, "mae_mean").iloc[-1]
        gain = second["mae_mean"] - best["mae_mean"]
        gain_pct = gain / second["mae_mean"] * 100

        robust = hdf["mae_std"].loc[hdf["mae_mean"].idxmin()]
        is_robust = hdf["mae_std"].max() < 0.5 * hdf["mae_mean"].min()  # loose rule

        print(f"\n  t+{h}h  →  BEST: {best['model']}")
        print(f"          MAE={best['mae_mean']:.4f} ± {best['mae_std']:.4f}")
        print(f"          Margin over runner-up: {gain:.4f} ({gain_pct:.1f}%)")

        if gain_pct < 1.0:
            print(f"          ⚠️  Gain is within noise (<1%) — models are equivalent.")

        if not is_robust:
            print(f"          ⚠️  High fold variance (std={robust:.4f}) — check stability.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=METRICS_DIR / "benchmark_table.csv")
    args = parser.parse_args()

    print("Loading results ...")
    df = load_results()
    print(f"Loaded {len(df)} rows from {df['model'].nunique()} models")

    benchmark = make_benchmark_table(df)

    print("\n" + "="*70)
    print("UNIFIED BENCHMARK TABLE — MAE (lower is better)")
    print("="*70)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(benchmark.to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(args.output, index=False)
    print(f"\n✅ Saved → {args.output}")

    print_selection_memo(benchmark)


if __name__ == "__main__":
    main()
