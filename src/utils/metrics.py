"""src/utils/metrics.py — Shared evaluation metrics for all models."""
import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def skill_score(y_true, y_pred, y_persist) -> float:
    """
    Persistence skill score: fraction improvement of model over naive persistence.
    skill = 1 - MAE(model) / MAE(persistence)
    Range: (-∞, 1]. Positive = better than persistence. 0 = same. Negative = worse.
    """
    mae_model   = mae(y_true, y_pred)
    mae_persist = mae(y_true, y_persist)
    return float(1.0 - mae_model / mae_persist) if mae_persist > 0 else 0.0


def evaluate(y_true, y_pred, y_persist=None) -> dict:
    """Full metrics dict. Pass y_persist to get skill score."""
    result = {
        "mae":  mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
    }
    if y_persist is not None:
        result["persist_mae"] = mae(y_true, y_persist)
        result["skill"]       = skill_score(y_true, y_pred, y_persist)
    return result


def summarize_folds(fold_results: list[dict]) -> pd.DataFrame:
    """Aggregate fold-level result dicts into mean ± std summary table."""
    df = pd.DataFrame(fold_results)
    agg: dict = {
        "mae_mean":   ("mae",  "mean"), "mae_std":  ("mae",  "std"),
        "rmse_mean":  ("rmse", "mean"),
        "r2_mean":    ("r2",   "mean"), "r2_std":   ("r2",   "std"),
    }
    if "skill" in df.columns:
        agg["skill_mean"] = ("skill", "mean")
    return df.groupby("horizon").agg(**agg).reset_index()
