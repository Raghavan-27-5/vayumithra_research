"""
src/pipeline/ts_dataset.py
──────────────────────────
PyTorch Dataset for DLinear, Mamba, and KAN.

WHY raw sequences and not LGBM features:
  DLinear and Mamba are sequence-to-sequence models — they learn temporal
  patterns directly from the raw multivariate time series. They do NOT use
  hand-crafted lag/rolling features because:
    1. They look back over a full 336-hour window and learn those patterns themselves.
    2. Adding pre-computed lags would create redundant and correlated inputs.
  This is consistent with the DLinear paper (Zeng et al. 2022) and Mamba paper.

What IS included as model input (SEQUENCE_VARIATES):
  - Raw wind speed, direction, temperature, humidity, pressure (the physics)
  - Cyclical time features (hour_sin/cos, month_sin/cos) — these encode
    diurnal/seasonal cycles that are critical and also used by LGBM
  - Monsoon regime flags — critical for Indian coastal wind patterns

What is NOT included (and why):
  - Pre-computed lags (the 336h window gives the model the raw history)
  - Rolling stats (model learns rolling patterns from the window)
  - Spatial/regional features (aggregated across stations — not causal per-window)
  - Neighbor features (not available in a single-station window)

Causality guarantee:
  Window [i-seq_len : i] → target at timestamp i+h-1
  The window ends BEFORE the forecast origin. No future data is ever included.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Variates included in the raw time series input (same order always)
SEQUENCE_VARIATES = [
    "wind_speed",
    "wind_direction",
    "temperature",
    "humidity",
    "surface_pressure",
    # Cyclical time — already computed in build_full_feature_matrix, no leakage
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "is_sw_monsoon", "is_ne_monsoon",
]
N_VARIATES = len(SEQUENCE_VARIATES)


class WindWindowDataset(Dataset):
    """
    Causal sliding-window dataset for one fold split.

    Args:
        df          : fully-engineered DataFrame
        fold        : dict with train_start/train_end/val_start/val_end keys
        split       : "train" or "val"
        seq_len     : look-back length in hours (default 336 = 14 days)
        horizons    : list of forecast horizons — target vector has len(horizons) values
        variates    : columns to use as model input (default: SEQUENCE_VARIATES)
        scaler_mean : per-variate mean (computed from train; applied to val)
        scaler_std  : per-variate std  (computed from train; applied to val)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fold: dict,
        split: str,
        seq_len:  int       = 336,
        horizons: list[int] = None,
        variates: list[str] = None,
        scaler_mean: np.ndarray | None = None,
        scaler_std:  np.ndarray | None = None,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        horizons = horizons or [1, 6, 12, 24, 48]
        variates = variates or SEQUENCE_VARIATES

        # Date filter
        if split == "train":
            mask = (df["datetime"] >= fold["train_start"]) & \
                   (df["datetime"] <  fold["train_end"])
        else:
            mask = (df["datetime"] >= fold["val_start"]) & \
                   (df["datetime"] <  fold["val_end"])

        split_df = df[mask].copy()
        missing  = [v for v in variates if v not in split_df.columns]
        if missing:
            raise KeyError(f"Missing variates in df: {missing}. "
                           f"Run build_full_feature_matrix() first.")

        # Build windows per station (no inter-station mixing)
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        max_h = max(horizons)

        for station_id, sdf in split_df.groupby("Index"):
            sdf    = sdf.sort_values("datetime").reset_index(drop=True)
            values = sdf[variates].values.astype("float32")          # (T, C)
            tgt    = {h: sdf[f"target_t_plus_{h}"].values for h in horizons}
            n      = len(sdf)

            for i in range(seq_len, n - max_h + 1):
                x = values[i - seq_len : i]                           # (seq_len, C)
                y = np.array([tgt[h][i + h - 1] for h in horizons],
                             dtype="float32")                          # (n_horizons,)
                if not (np.any(np.isnan(x)) or np.any(np.isnan(y))):
                    self.windows.append((x, y))

        if len(self.windows) == 0:
            raise RuntimeError(
                f"No valid windows for fold {fold['fold']} split={split}. "
                "Check date ranges and seq_len vs data length per station."
            )

        # Normalization — fit ONLY on training split
        all_x = np.stack([w[0] for w in self.windows])       # (N, seq_len, C)
        if scaler_mean is None:
            self.mean = all_x.mean(axis=(0, 1))               # (C,)
            self.std  = all_x.std(axis=(0, 1)) + 1e-8
        else:
            self.mean = scaler_mean
            self.std  = scaler_std

        self.horizons = horizons
        self.variates = variates
        self.seq_len  = seq_len

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.windows[idx]
        x = (x - self.mean) / self.std                        # (seq_len, C)
        return torch.from_numpy(x), torch.from_numpy(y)


def build_fold_datasets(
    df: pd.DataFrame,
    fold: dict,
    seq_len:  int       = 336,
    horizons: list[int] = None,
    variates: list[str] = None,
) -> tuple[WindWindowDataset, WindWindowDataset]:
    """
    Build train + val datasets for a fold.
    Scaler is ALWAYS fitted on train and applied to val (no leakage).
    """
    train_ds = WindWindowDataset(df, fold, "train", seq_len, horizons, variates)
    val_ds   = WindWindowDataset(
        df, fold, "val", seq_len, horizons, variates,
        scaler_mean=train_ds.mean,
        scaler_std=train_ds.std,
    )
    return train_ds, val_ds
