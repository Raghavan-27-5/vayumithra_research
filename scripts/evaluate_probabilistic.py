"""
evaluate_probabilistic.py
─────────────────────────
Evaluate a trained iTransformerNHiTS_Probabilistic model and compute all
probabilistic metrics: pinball loss per quantile, P50 MAE/RMSE, coverage,
interval width, Winkler score.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.probabilistic_model import (
    QUANTILES,
    QUANTILE_LABELS,
    iTransformerNHiTS_Probabilistic,
    PinballLoss,
)

DATA_PATH = r"C:\Projects\raghavan\iTransformer\dataset\vayumithra\vayumithra_10st_uv.csv"
TARGET_STATION = "s5"
ALL_HORIZONS = [1, 2, 3, 4, 5, 6, 12, 24, 48]

FOLDS = [
    {"fold": 1, "train_end": "2019-01-01", "val_start": "2019-01-01",
     "val_end": "2020-01-01", "test_start": "2020-01-01", "test_end": "2021-01-01"},
    {"fold": 2, "train_end": "2020-01-01", "val_start": "2020-01-01",
     "val_end": "2021-01-01", "test_start": "2021-01-01", "test_end": "2022-01-01"},
]


# ── Metrics ──────────────────────────────────────────────────────────────────

def pinball_score(y_true, y_pred, q):
    errors = y_true - y_pred
    return float(np.mean(np.maximum(q * errors, (q - 1) * errors)))

def interval_coverage(y_true, p_low, p_high):
    covered = np.sum((y_true >= p_low) & (y_true <= p_high))
    return float(covered / len(y_true))

def interval_width(p_low, p_high):
    return float(np.mean(p_high - p_low))

def winkler_score(y_true, p_low, p_high, alpha=0.20):
    width = p_high - p_low
    penalty_low = (2 / alpha) * np.maximum(p_low - y_true, 0)
    penalty_high = (2 / alpha) * np.maximum(y_true - p_high, 0)
    return float(np.mean(width + penalty_low + penalty_high))


# ── Data ─────────────────────────────────────────────────────────────────────

def _load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    ws_col = f"ws_{TARGET_STATION}"
    df[ws_col] = np.sqrt(
        df[f"u_{TARGET_STATION}"].values ** 2 + df[f"v_{TARGET_STATION}"].values ** 2
    )
    return df

def _make_windows(df, feature_cols, ws_col, start, end, mean, std, seq_len, pred_len):
    mask = (df["datetime"] >= start) & (df["datetime"] < end)
    sub = df[mask]
    vals = (sub[feature_cols].values.astype(np.float32) - mean) / std
    ws_raw = sub[ws_col].values.astype(np.float32)
    xs, ys = [], []
    n = len(sub)
    for i in range(0, n - seq_len - pred_len + 1):
        x_win = vals[i: i + seq_len]
        if np.isnan(x_win).any(): continue
        y_win = ws_raw[i + seq_len + pred_len - 1]
        if np.isnan(y_win): continue
        xs.append(x_win)
        ys.append([y_win])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── Main evaluation ──────────────────────────────────────────────────────────

class Config:
    def __init__(self, **kwargs):
        self.seq_len = 336; self.pred_len = 1; self.enc_in = 50
        self.d_model = 256; self.n_heads = 4; self.e_layers = 2
        self.d_ff = 1024; self.dropout = 0.1; self.activation = "gelu"
        self.embed = "timeF"; self.freq = "h"; self.factor = 1
        self.class_strategy = "projection"; self.use_norm = True
        self.output_attention = False
        self.__dict__.update(kwargs)


def evaluate_one(fold_num: int, horizon: int, ckpt_dir: str, device: torch.device):
    df = _load_data()
    ws_col = f"ws_{TARGET_STATION}"
    feature_cols = [ws_col] + [c for c in df.columns if c != "datetime"]

    fold = [f for f in FOLDS if f["fold"] == fold_num][0]

    # Compute train stats for normalization
    train_mask = df["datetime"] < fold["train_end"]
    train_df = df[train_mask]
    train_mean = train_df[feature_cols].mean().values.astype(np.float32)
    train_std = (train_df[feature_cols].std().values + 1e-8).astype(np.float32)

    # Build test windows
    _, vx, vy = None, None, None  # val not needed for final eval
    tex, tey = _make_windows(
        df, feature_cols, ws_col,
        fold["test_start"], fold["test_end"],
        train_mean, train_std, 336, horizon,
    )

    # Load model
    ckpt_path = Path(ckpt_dir) / f"probabilistic_fold{fold_num}_h{horizon}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = Config(pred_len=horizon, enc_in=tex.shape[2])
    model = iTransformerNHiTS_Probabilistic(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Evaluate
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(tex), torch.from_numpy(tey)
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=0
    )

    preds_list, ys_list = [], []
    criterion = PinballLoss(quantiles=QUANTILES)
    total_loss = 0.0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(x_enc=bx)
            total_loss += criterion(pred, by).item()
            preds_list.append(pred.cpu().numpy())
            ys_list.append(by.cpu().numpy())

    test_pinball = total_loss / len(test_loader)
    preds = np.concatenate(preds_list, axis=0).squeeze(1)  # (N, 4)
    y_true = np.concatenate(ys_list, axis=0).squeeze(1)    # (N,)

    # Compute metrics
    results = {
        "horizon": horizon,
        "fold": fold_num,
        "test_pinball_avg": test_pinball,
        "P50_MAE": float(mean_absolute_error(y_true, preds[:, 1])),
        "P50_RMSE": float(np.sqrt(mean_squared_error(y_true, preds[:, 1]))),
    }

    for qi, (q, label) in enumerate(zip(QUANTILES, QUANTILE_LABELS)):
        results[f"pinball_{label}"] = pinball_score(y_true, preds[:, qi], q)

    p10, p90 = preds[:, 0], preds[:, 2]
    results["coverage_P10_P90"] = interval_coverage(y_true, p10, p90)
    results["width_P10_P90"] = interval_width(p10, p90)
    results["winkler_P10_P90"] = winkler_score(y_true, p10, p90)

    return results, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--horizon", nargs="+", type=int, default=[1])
    parser.add_argument("--ckpt_dir", type=str, default="results/models/probabilistic")
    parser.add_argument("--output", type=str, default="results/metrics/probabilistic_results.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    all_results = {}
    for h in args.horizon:
        print(f"\nEvaluating fold={args.fold} horizon={h}h ...")
        results, _ = evaluate_one(args.fold, h, args.ckpt_dir, device)
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        all_results[f"H{h}"] = results

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
