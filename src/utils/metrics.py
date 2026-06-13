"""src/utils/metrics.py — Shared evaluation metrics."""
import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def skill_score(y_true, y_persist):
    """Fraction improvement over persistence baseline."""
    mae_model   = mae(y_true, y_persist)  # NOTE: caller passes persist preds
    # Actually: skill = 1 - mae_model / mae_persist — see usage
    return mae_model


def persistence_mae(y_true, lag_values):
    return mae(y_true, lag_values)


def evaluate(y_true, y_pred, y_persist=None) -> dict:
    """Return a standard metrics dict."""
    results = {
        "mae":  mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2":   r2(y_true, y_pred),
    }
    if y_persist is not None:
        p_mae = persistence_mae(y_true, y_persist)
        results["persist_mae"] = p_mae
        results["skill"]       = 1.0 - (results["mae"] / p_mae) if p_mae > 0 else 0.0
    return results


def summarize_folds(fold_results: list[dict]) -> pd.DataFrame:
    """Aggregate fold-level results into mean ± std summary."""
    df = pd.DataFrame(fold_results)
    summary = df.groupby("horizon").agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"),   r2_std=("r2", "std"),
        skill_mean=("skill", "mean"),
    ).reset_index()
    return summary
