"""
src/pipeline/ts_dataset.py
──────────────────────────
PyTorch Dataset for DLinear, Mamba, and KAN.

FEATURE MODES
─────────────
You asked: "if we feed DLinear the features directly, does it perform better?"
YES — absolutely. DLinear is channel-independent: every input column gets its
own linear layer. Feeding it pre-engineered features (lags, pressure tendencies,
monsoon flags, etc.) gives it the same information advantage that LGBM has.

Two modes are available via the `variates` argument:

  SEQUENCE_VARIATES  (11 cols) — raw physics only
      Baseline DLinear per the paper. The 336h look-back window captures
      lag/rolling structure implicitly. Fastest to build.

  FEATURE_VARIATES   (50+ cols) — engineered features
      Feature-augmented DLinear. Each lag, rolling stat, and atmospheric
      tendency becomes its own channel. Expected to match or beat LGBM.
      Slightly more memory, no extra training time.

CAUSALITY GUARANTEE
───────────────────
Window: values[i-seq_len : i]   — covers past observations only
Target: raw_ws[i + h - 1]       — h steps ahead of the LAST point in window

Proof for h=1:
  Window ends at index i-1 (last known point)
  1-step ahead = index i
  raw_ws[i + 1 - 1] = raw_ws[i] ✅

Proof for h=48:
  48-step ahead = index i+47
  raw_ws[i + 48 - 1] = raw_ws[i+47] ✅

The pre-computed target_t_plus_h columns are NOT used for the target value
(avoids a double-shift bug). Raw wind_speed is used directly.

NORMALIZATION
─────────────
Stats are computed from the raw training-split DataFrame (O(rows) memory),
NOT from stacked windows (which would be O(rows × seq_len × C) — OOM risk).
Scaler fitted on train, applied identically to val.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── Mode 1: raw physics only (paper default) ─────────────────────────────────
SEQUENCE_VARIATES = [
    "wind_speed",
    "wind_direction",
    "temperature",
    "humidity",
    "surface_pressure",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "is_sw_monsoon", "is_ne_monsoon",
]

# ── Mode 2: engineered features (LGBM-comparable input richness) ─────────────
# Lags/rolling are explicitly included — DLinear treats each as a channel.
# Short-lag features (lag_1/2/3, accel_1) are excluded for long horizons;
# use horizon-specific subsets via get_feature_variates(horizon).
FEATURE_VARIATES_BASE = [
    # Raw physics (always included)
    "wind_speed", "wind_direction", "temperature", "humidity", "surface_pressure",
    # Wind dynamics
    "wind_speed_lag_6",  "wind_speed_lag_12", "wind_speed_lag_24",
    "wind_speed_lag_48", "wind_speed_lag_72", "wind_speed_lag_168",
    "wind_speed_roll_mean_6",  "wind_speed_roll_std_6",
    "wind_speed_roll_mean_12", "wind_speed_roll_std_12",
    "wind_speed_roll_mean_24", "wind_speed_roll_std_24",
    "dir_sin", "dir_cos", "wind_x", "wind_y",
    "wind_accel_3", "wind_accel_6",
    "wind_ewm_6", "wind_ewm_12", "wind_ewm_24",
    "wind_volatility_6", "wind_volatility_24",
    # Atmospheric
    "pressure_tendency_3h", "pressure_tendency_6h", "pressure_tendency_24h",
    "temp_tendency_3h", "temp_tendency_6h",
    "surface_pressure_lag_6",  "surface_pressure_lag_24",
    "temperature_lag_6",       "temperature_lag_24",
    "humidity_lag_6",
    # Time
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "hour_x_month_sin", "hour_x_month_cos",
    "is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
    # Spatial
    "regional_ws_mean", "ws_vs_region", "ws_anomaly",
    "regional_pressure_mean", "pressure_vs_region",
]

FEATURE_VARIATES_SHORT_EXTRA = [
    # Additional short-range features (only for h ≤ 6)
    "wind_speed_lag_1", "wind_speed_lag_2", "wind_speed_lag_3",
    "wind_accel_1",
]

N_VARIATES         = len(SEQUENCE_VARIATES)
N_FEATURE_VARIATES = len(FEATURE_VARIATES_BASE) + len(FEATURE_VARIATES_SHORT_EXTRA)


def get_feature_variates(horizon: int) -> list[str]:
    """Return the feature variate list appropriate for the given horizon."""
    if horizon <= 6:
        return FEATURE_VARIATES_SHORT_EXTRA + FEATURE_VARIATES_BASE
    return FEATURE_VARIATES_BASE


class WindWindowDataset(Dataset):
    """
    Causal sliding-window dataset for one fold split.

    Args:
        df          : fully-engineered DataFrame (run build_full_feature_matrix first)
        fold        : dict with fold/train_start/train_end/val_start/val_end
        split       : "train" or "val"
        seq_len     : look-back window in hours (default 336 = 14 days)
        horizons    : forecast horizons whose targets are packed into y
        variates    : input feature columns (SEQUENCE_VARIATES or FEATURE_VARIATES)
        scaler_mean : (C,) mean array; if None, computed from this split (train only)
        scaler_std  : (C,) std  array; if None, computed from this split (train only)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fold: dict,
        split: str,
        seq_len:     int        = 336,
        horizons:    list[int]  = None,
        variates:    list[str]  = None,
        scaler_mean: np.ndarray | None = None,
        scaler_std:  np.ndarray | None = None,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        horizons = horizons or [1, 6, 12, 24, 48]
        variates = variates or SEQUENCE_VARIATES

        # ── Date filter ───────────────────────────────────────────────────────
        if split == "train":
            mask = (df["datetime"] >= fold["train_start"]) & \
                   (df["datetime"] <  fold["train_end"])
        else:
            mask = (df["datetime"] >= fold["val_start"]) & \
                   (df["datetime"] <  fold["val_end"])

        split_df = df[mask].copy()

        missing = [v for v in variates if v not in split_df.columns]
        if missing:
            raise KeyError(
                f"Missing variates: {missing[:5]}{'...' if len(missing)>5 else ''}. "
                "Run build_full_feature_matrix() first."
            )

        # ── Normalization stats from DataFrame directly (O(rows), not O(rows×L×C)) ──
        if scaler_mean is None:
            self.mean = split_df[variates].mean().values.astype("float32")
            self.std  = (split_df[variates].std().values + 1e-8).astype("float32")
        else:
            self.mean = scaler_mean.astype("float32")
            self.std  = scaler_std.astype("float32")

        # ── Build windows per station (no inter-station mixing) ───────────────
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        max_h = max(horizons)

        for _, sdf in split_df.groupby("Index"):
            sdf    = sdf.sort_values("datetime").reset_index(drop=True)
            values = sdf[variates].values.astype("float32")        # (T, C)
            raw_ws = sdf["wind_speed"].values.astype("float32")    # (T,)  — target source
            n      = len(sdf)

            # i = index of first unknown step (forecast origin)
            # Window covers [i-seq_len, i) — last obs is at i-1
            # h-step target = wind_speed[i + h - 1]  (h steps after last obs)
            # Loop bound: i + max_h - 1 < n  →  i < n - max_h + 1
            for i in range(seq_len, n - max_h + 1):
                x = values[i - seq_len : i]                        # (seq_len, C)
                y = np.array(
                    [raw_ws[i + h - 1] for h in horizons],
                    dtype="float32",
                )                                                   # (n_horizons,)
                if not (np.any(np.isnan(x)) or np.any(np.isnan(y))):
                    self.windows.append((x, y))

        if len(self.windows) == 0:
            raise RuntimeError(
                f"No valid windows for fold {fold.get('fold','?')} split={split}. "
                "Check: (1) date ranges in fold dict, (2) seq_len < station series length."
            )

        self.horizons = horizons
        self.variates = variates
        self.seq_len  = seq_len
        print(f"  [{split}] {len(self.windows):,} windows | "
              f"{len(variates)} variates | seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.windows[idx]
        x = (x - self.mean) / self.std                            # (seq_len, C)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())


def build_fold_datasets(
    df: pd.DataFrame,
    fold: dict,
    seq_len:  int       = 336,
    horizons: list[int] = None,
    variates: list[str] = None,
) -> tuple[WindWindowDataset, WindWindowDataset]:
    """
    Build train + val datasets for a fold.
    Scaler fitted on train ONLY, applied to val — no leakage.
    """
    train_ds = WindWindowDataset(df, fold, "train", seq_len, horizons, variates)
    val_ds   = WindWindowDataset(
        df, fold, "val", seq_len, horizons, variates,
        scaler_mean=train_ds.mean,
        scaler_std=train_ds.std,
    )
    return train_ds, val_ds
