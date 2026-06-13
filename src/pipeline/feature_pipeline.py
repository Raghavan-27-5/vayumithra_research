"""
src/pipeline/feature_pipeline.py
─────────────────────────────────
CANONICAL feature engineering pipeline — verbatim extraction from wind_forecast.ipynb.
This is the single source of truth for ALL feature groups across ALL models.

Cell-by-cell correspondence:
  Cell 9   → add_cyclical_time(), add_wind_vectors()
  Cell 10  → add_wind_lags_and_targets()
  Cell 22  → add_advanced_wind_features()
  Cell 24  → add_spatial_regional_features()
  Cell 29  → add_neighbor_propagation_features()
  Cell 30  → add_atmospheric_physics_features()
  Cell 31  → HORIZON_FEATURES dict (feature sets per horizon)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────────────────────────────────────
# Constants — match notebook exactly
# ─────────────────────────────────────────────────────────────────────────────
WIND_LAG_HOURS    = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS      = [6, 12, 24, 48]
EWM_SPANS         = [6, 12, 24]
VOL_WINDOWS       = [6, 24, 72]
ACCEL_DIFFS       = [1, 3, 6]
ATMOS_TEND_DIFFS  = [1, 3, 6, 12, 24]
ATMOS_LAG_HOURS   = [1, 6, 12, 24, 48]
FORECAST_HORIZONS = [1, 2, 3, 4, 5, 6, 12, 24, 48]
EPS = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# Canonical feature column groups  (Cell 31 / Cell 15 of notebook)
# ─────────────────────────────────────────────────────────────────────────────
WIND_BASE = [
    "wind_speed_lag_1", "wind_speed_lag_2", "wind_speed_lag_3",
    "wind_speed_lag_6", "wind_speed_lag_12", "wind_speed_lag_24",
    "wind_speed_lag_48", "wind_speed_lag_72", "wind_speed_lag_168",
    "wind_speed_roll_mean_6",  "wind_speed_roll_std_6",
    "wind_speed_roll_mean_12", "wind_speed_roll_std_12",
    "wind_speed_roll_mean_24", "wind_speed_roll_std_24",
    "wind_speed_roll_mean_48", "wind_speed_roll_std_48",
    "dir_sin", "dir_cos", "wind_x", "wind_y",
    "wind_accel_1", "wind_accel_3", "wind_accel_6",
    "dir_change_sin", "dir_change_cos",
    "wind_ewm_6", "wind_ewm_12", "wind_ewm_24",
    "wind_volatility_6", "wind_volatility_24", "wind_volatility_72",
]

TIME_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "hour_x_month_sin", "hour_x_month_cos",
    "hour_x_month_sin2", "hour_x_month_cos2",
    "is_sw_monsoon", "is_ne_monsoon", "is_dry_season",
]

ATMOS_FEATURES = [
    "pressure_tendency_1h",  "pressure_tendency_3h",  "pressure_tendency_6h",
    "pressure_tendency_12h", "pressure_tendency_24h",
    "temp_tendency_1h",  "temp_tendency_3h",  "temp_tendency_6h",
    "temp_tendency_12h", "temp_tendency_24h",
    "humidity_tendency_1h", "humidity_tendency_6h", "humidity_tendency_24h",
    "surface_pressure_lag_1",  "surface_pressure_lag_6",
    "surface_pressure_lag_12", "surface_pressure_lag_24", "surface_pressure_lag_48",
    "temperature_lag_1",  "temperature_lag_6",
    "temperature_lag_12", "temperature_lag_24", "temperature_lag_48",
    "humidity_lag_1", "humidity_lag_6", "humidity_lag_24",
    "surface_pressure_roll_mean_6",  "surface_pressure_roll_std_6",
    "surface_pressure_roll_mean_24", "surface_pressure_roll_std_24",
    "temperature_roll_mean_6",  "temperature_roll_std_6",
    "temperature_roll_mean_24", "temperature_roll_std_24",
    "ws_x_pressure_tend_6h", "ws_x_temp_tend_6h", "pressure_x_humidity",
]

SPATIAL_ATMOS = [
    "regional_ws_mean",       "regional_ws_std",
    "regional_pressure_mean", "regional_pressure_std",
    "regional_humidity_mean", "regional_humidity_std",
    "ws_vs_region", "pressure_vs_region", "humidity_vs_region",
    "ws_anomaly",   "pressure_anomaly",
]

NEIGHBOR_FEATURES = [
    "neighbor_1_lag1", "neighbor_1_lag6",
    "neighbor_2_lag1", "neighbor_2_lag6",
]

META = ["Longitude", "Latitude", "Index"]

# Long-horizon feature set — drops short-lag leakers (Cell 31)
_WIND_LONG = [f for f in WIND_BASE if f not in
              ["wind_speed_lag_1", "wind_speed_lag_2",
               "wind_speed_lag_3", "wind_accel_1"]]

# Per-horizon feature sets — canonical (mirrors Cell 31 exactly)
HORIZON_FEATURES: dict[int, list[str]] = {
    1:  WIND_BASE + TIME_FEATURES + ATMOS_FEATURES + META,
    2:  WIND_BASE + TIME_FEATURES + ATMOS_FEATURES + META,
    3:  WIND_BASE + TIME_FEATURES + ATMOS_FEATURES + META,
    4:  WIND_BASE + TIME_FEATURES + ATMOS_FEATURES + META,
    5:  WIND_BASE + TIME_FEATURES + ATMOS_FEATURES + META,
    6:  WIND_BASE + TIME_FEATURES + ATMOS_FEATURES + SPATIAL_ATMOS + META,
    12: WIND_BASE + TIME_FEATURES + ATMOS_FEATURES + SPATIAL_ATMOS + META,
    24: _WIND_LONG + TIME_FEATURES + ATMOS_FEATURES + SPATIAL_ATMOS + META,
    48: _WIND_LONG + TIME_FEATURES + ATMOS_FEATURES + SPATIAL_ATMOS + META,
}

TARGET_COLS = [f"target_t_plus_{h}" for h in FORECAST_HORIZONS]

# Walk-forward folds — identical across ALL models (Cell 21)
WALK_FORWARD_FOLDS = [
    {"fold": 1, "train_start": "2013-01-01", "train_end": "2019-01-01",
     "val_start": "2019-01-01", "val_end": "2020-01-01"},
    {"fold": 2, "train_start": "2013-01-01", "train_end": "2020-01-01",
     "val_start": "2020-01-01", "val_end": "2021-01-01"},
    {"fold": 3, "train_start": "2013-01-01", "train_end": "2021-01-01",
     "val_start": "2021-01-01", "val_end": "2022-01-01"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering functions
# ─────────────────────────────────────────────────────────────────────────────

def add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 9 [2] — Cyclical time features + monsoon regimes."""
    df = df.copy()
    df["hour_sin"]  = np.sin(2 * np.pi * df["HOUR"] / 24).astype("float32")
    df["hour_cos"]  = np.cos(2 * np.pi * df["HOUR"] / 24).astype("float32")
    dow = df["datetime"].dt.dayofweek
    df["dow_sin"]   = np.sin(2 * np.pi * dow / 7).astype("float32")
    df["dow_cos"]   = np.cos(2 * np.pi * dow / 7).astype("float32")
    df["month_sin"] = np.sin(2 * np.pi * df["MONTH"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["MONTH"] / 12).astype("float32")
    # Cell 30 [4] — Diurnal × seasonal interactions
    df["hour_x_month_sin"]  = (df["hour_sin"] * df["month_sin"]).astype("float32")
    df["hour_x_month_cos"]  = (df["hour_cos"] * df["month_cos"]).astype("float32")
    df["hour_x_month_sin2"] = (df["hour_sin"] * df["month_cos"]).astype("float32")
    df["hour_x_month_cos2"] = (df["hour_cos"] * df["month_sin"]).astype("float32")
    # Cell 30 [5] — Monsoon regime encoding
    df["is_sw_monsoon"] = df["MONTH"].isin([6, 7, 8, 9]).astype("int8")
    df["is_ne_monsoon"] = df["MONTH"].isin([10, 11, 12]).astype("int8")
    df["is_dry_season"] = df["MONTH"].isin([1, 2, 3, 4, 5]).astype("int8")
    return df


def add_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 9 [3][4] — Wind direction encoding + velocity decomposition."""
    df = df.copy()
    theta         = np.deg2rad(df["wind_direction"])
    df["dir_sin"] = np.sin(theta).astype("float32")
    df["dir_cos"] = np.cos(theta).astype("float32")
    df["wind_x"]  = (df["wind_speed"] * df["dir_cos"]).astype("float32")
    df["wind_y"]  = (df["wind_speed"] * df["dir_sin"]).astype("float32")
    return df


def add_wind_lags_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 10 — lags, rolling stats, forecast targets."""
    df = df.copy()
    g = df.groupby("Index")["wind_speed"]
    for lag in WIND_LAG_HOURS:
        df[f"wind_speed_lag_{lag}"] = g.shift(lag).astype("float32")
    for w in ROLL_WINDOWS:
        r = g.rolling(w)
        df[f"wind_speed_roll_mean_{w}"] = r.mean().reset_index(level=0, drop=True).astype("float32")
        df[f"wind_speed_roll_std_{w}"]  = r.std().reset_index(level=0, drop=True).astype("float32")
    for h in FORECAST_HORIZONS:
        df[f"target_t_plus_{h}"] = g.shift(-h).astype("float32")
    return df


def add_advanced_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 22 — acceleration, directional dynamics, EWM, volatility, interactions."""
    df = df.copy()
    g = df.groupby("Index")
    for d in ACCEL_DIFFS:
        df[f"wind_accel_{d}"] = g["wind_speed"].diff(d).astype("float32")
    df["dir_change_sin"] = g["dir_sin"].diff(1).astype("float32")
    df["dir_change_cos"] = g["dir_cos"].diff(1).astype("float32")
    for span in EWM_SPANS:
        df[f"wind_ewm_{span}"] = (
            g["wind_speed"].transform(lambda x: x.ewm(span=span, adjust=False).mean())
            .astype("float32")
        )
    for w in VOL_WINDOWS:
        df[f"wind_volatility_{w}"] = (
            g["wind_speed"].rolling(w).std()
            .reset_index(level=0, drop=True).astype("float32")
        )
    # Cell 22 [5] — interaction features
    df["speed_x_hour_sin"] = (df["wind_speed"] * df["hour_sin"]).astype("float32")
    df["speed_x_hour_cos"] = (df["wind_speed"] * df["hour_cos"]).astype("float32")
    df["speed_x_dir_sin"]  = (df["wind_speed"] * df["dir_sin"]).astype("float32")
    df["speed_x_dir_cos"]  = (df["wind_speed"] * df["dir_cos"]).astype("float32")
    return df


def add_atmospheric_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 30 — pressure/temp/humidity tendencies, lags, rolling stats, cross-interactions."""
    df = df.copy()
    g = df.groupby("Index")
    for d in ATMOS_TEND_DIFFS:
        df[f"pressure_tendency_{d}h"] = g["surface_pressure"].diff(d).astype("float32")
        df[f"temp_tendency_{d}h"]     = g["temperature"].diff(d).astype("float32")
    for d in [1, 6, 24]:
        df[f"humidity_tendency_{d}h"] = g["humidity"].diff(d).astype("float32")
    for col, lags in [("surface_pressure", ATMOS_LAG_HOURS),
                      ("temperature",       ATMOS_LAG_HOURS),
                      ("humidity",          [1, 6, 24])]:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = g[col].shift(lag).astype("float32")
    for col in ["surface_pressure", "temperature"]:
        for w in [6, 24]:
            r = g[col].rolling(w)
            df[f"{col}_roll_mean_{w}"] = r.mean().reset_index(level=0, drop=True).astype("float32")
            df[f"{col}_roll_std_{w}"]  = r.std().reset_index(level=0, drop=True).astype("float32")
    df["ws_x_pressure_tend_6h"] = (df["wind_speed"] * df["pressure_tendency_6h"].fillna(0)).astype("float32")
    df["ws_x_temp_tend_6h"]     = (df["wind_speed"] * df["temp_tendency_6h"].fillna(0)).astype("float32")
    df["pressure_x_humidity"]   = (df["surface_pressure"] * df["humidity"]).astype("float32")
    return df


def add_spatial_regional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 24 — regional aggregates, deviations, anomalies."""
    df = df.copy()
    regional = (
        df.groupby("datetime")
        .agg({"wind_speed":       ["mean", "std"],
              "surface_pressure": ["mean", "std"],
              "humidity":         ["mean", "std"]})
    )
    regional.columns = [
        "regional_ws_mean",       "regional_ws_std",
        "regional_pressure_mean", "regional_pressure_std",
        "regional_humidity_mean", "regional_humidity_std",
    ]
    df = df.merge(regional.reset_index(), on="datetime", how="left")
    df["ws_vs_region"]       = (df["wind_speed"]       - df["regional_ws_mean"]).astype("float32")
    df["pressure_vs_region"] = (df["surface_pressure"] - df["regional_pressure_mean"]).astype("float32")
    df["humidity_vs_region"] = (df["humidity"]         - df["regional_humidity_mean"]).astype("float32")
    df["ws_anomaly"]         = ((df["wind_speed"] - df["regional_ws_mean"]) /
                                (df["regional_ws_std"] + EPS)).astype("float32")
    df["pressure_anomaly"]   = ((df["surface_pressure"] - df["regional_pressure_mean"]) /
                                (df["regional_pressure_std"] + EPS)).astype("float32")
    return df


def add_neighbor_propagation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 29 — vectorized O(M·T·logT) nearest-neighbor lag propagation."""
    df = df.copy()
    stations = (
        df[["Index", "Latitude", "Longitude"]]
        .drop_duplicates(subset=["Latitude", "Longitude"])
        .sort_values("Index").reset_index(drop=True)
    )
    coords = stations[["Latitude", "Longitude"]].values
    nbrs = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(coords)
    _, indices = nbrs.kneighbors(coords)

    neighbor_map: dict[int, list[int]] = {}
    for i, row in stations.iterrows():
        sid = int(row["Index"])
        neighbor_map[sid] = stations.iloc[indices[i][1:]]["Index"].tolist()

    pivot_lag1 = df.pivot_table(index="datetime", columns="Index", values="wind_speed_lag_1")
    pivot_lag6 = df.pivot_table(index="datetime", columns="Index", values="wind_speed_lag_6")

    for rank in range(2):
        mapping = {sid: ns[rank] for sid, ns in neighbor_map.items() if rank < len(ns)}
        vals1 = np.full(len(df), np.nan, dtype="float32")
        vals6 = np.full(len(df), np.nan, dtype="float32")
        for sid, nid in mapping.items():
            if nid not in pivot_lag1.columns:
                continue
            mask = df["Index"].values == sid
            dts  = df.loc[mask, "datetime"]
            vals1[mask] = pivot_lag1[nid].reindex(dts).values.astype("float32")
            vals6[mask] = pivot_lag6[nid].reindex(dts).values.astype("float32")
        df[f"neighbor_{rank+1}_lag1"] = vals1
        df[f"neighbor_{rank+1}_lag6"] = vals6

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_full_feature_matrix(df: pd.DataFrame, add_neighbors: bool = True,
                               verbose: bool = True) -> pd.DataFrame:
    """Run the full canonical feature engineering pipeline in order."""
    def log(msg):
        if verbose: print(msg)

    df = df.sort_values(["Index", "datetime"]).reset_index(drop=True)
    log("[1/7] Cyclical time + monsoon regimes ...")
    df = add_cyclical_time(df)
    log("[2/7] Wind vectors ...")
    df = add_wind_vectors(df)
    log("[3/7] Lags, rolling stats, targets ...")
    df = add_wind_lags_and_targets(df)
    log("[4/7] Advanced wind (accel, EWM, volatility) ...")
    df = add_advanced_wind_features(df)
    log("[5/7] Atmospheric physics ...")
    df = add_atmospheric_physics_features(df)
    log("[6/7] Spatial regional aggregates ...")
    df = add_spatial_regional_features(df)
    if add_neighbors:
        log("[7/7] Neighbor propagation ...")
        df = add_neighbor_propagation_features(df)
    log(f"✅ Feature matrix: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def build_lgbm_arrays(df: pd.DataFrame, horizon: int, fold: dict):
    """
    Return (X_train, y_train, X_val, y_val) for LGBM using canonical feature set.
    Drops NaNs. No scaling needed (LGBM is scale-invariant).
    """
    feats      = HORIZON_FEATURES[horizon]
    target_col = f"target_t_plus_{horizon}"
    available  = [f for f in feats if f in df.columns]

    train_mask = (df["datetime"] >= fold["train_start"]) & (df["datetime"] < fold["train_end"])
    val_mask   = (df["datetime"] >= fold["val_start"])   & (df["datetime"] < fold["val_end"])

    train_df = df[train_mask][available + [target_col, "datetime"]].dropna()
    val_df   = df[val_mask][available   + [target_col, "datetime"]].dropna()

    return (
        train_df[available], train_df[target_col],
        val_df[available],   val_df[target_col],
    )
