"""
train_probabilistic.py
─────────────────────
Train the iTransformerNHiTS_Probabilistic model on VayuMithra data
with walk-forward CV. Uses PinballLoss and produces 4 quantile outputs.

Usage:
    python scripts/train_probabilistic.py --fold 1 --horizon 1 6 24
"""
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.probabilistic_model import (
    QUANTILES,
    iTransformerNHiTS_Probabilistic,
    PinballLoss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Projects\raghavan\iTransformer\dataset\vayumithra\vayumithra_10st_uv.csv"
TARGET_STATION = "s5"  # predict wind speed for this station

WALK_FORWARD_FOLDS = [
    {"fold": 1, "train_end": "2019-01-01", "val_start": "2019-01-01",
     "val_end": "2020-01-01", "test_start": "2020-01-01", "test_end": "2021-01-01"},
    {"fold": 2, "train_end": "2020-01-01", "val_start": "2020-01-01",
     "val_end": "2021-01-01", "test_start": "2021-01-01", "test_end": "2022-01-01"},
    {"fold": 3, "train_end": "2021-01-01", "val_start": "2021-01-01",
     "val_end": "2022-01-01", "test_start": "2022-01-01", "test_end": "2022-10-01"},
]

ALL_HORIZONS = [1, 2, 3, 4, 5, 6, 12, 24, 48]


# ── Simple namespace for config ──────────────────────────────────────────────
class Config:
    def __init__(self, **kwargs):
        self.seq_len = 336
        self.pred_len = 1           # overridden per horizon
        self.enc_in = 50
        self.d_model = 256
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 1024
        self.dropout = 0.1
        self.activation = "gelu"
        self.embed = "timeF"
        self.freq = "h"
        self.factor = 1
        self.class_strategy = "projection"
        self.use_norm = True
        self.output_attention = False
        self.__dict__.update(kwargs)


# ── Data loading ─────────────────────────────────────────────────────────────

def _ws_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.sqrt(u ** 2 + v ** 2).astype(np.float32)


def load_vayumithra_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    ws_key = f"ws_{TARGET_STATION}"
    df[ws_key] = _ws_from_uv(
        df[f"u_{TARGET_STATION}"].values,
        df[f"v_{TARGET_STATION}"].values,
    )
    return df


def build_datasets(
    df: pd.DataFrame,
    fold: dict,
    seq_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_x, train_y), (val_x, val_y), (test_x, test_y)."""
    # Input columns: all except datetime
    feature_cols = [c for c in df.columns if c != "datetime"]
    ws_col = f"ws_{TARGET_STATION}"

    # Ensure wind speed is included in features
    if ws_col not in feature_cols:
        feature_cols = [ws_col] + feature_cols

    # Pre-normalize features (fit on train)
    train_mask = df["datetime"] < fold["train_end"]
    train_df = df[train_mask]
    train_mean = train_df[feature_cols].mean().values.astype(np.float32)
    train_std = (train_df[feature_cols].std().values + 1e-8).astype(np.float32)

    def _make_array(start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
        mask = (df["datetime"] >= start) & (df["datetime"] < end)
        sub = df[mask]
        # Normalise
        vals = (sub[feature_cols].values.astype(np.float32) - train_mean) / train_std
        ws_raw = sub[ws_col].values.astype(np.float32)
        n = len(sub)
        xs, ys = [], []
        for i in range(seq_len, n - pred_len + 1):
            x_win = vals[i - seq_len : i]
            if np.isnan(x_win).any():
                continue
            y_win = ws_raw[i + pred_len - 1]
            if np.isnan(y_win):
                continue
            xs.append(x_win)
            ys.append([y_win])
        if not xs:
            raise RuntimeError(
                f"No valid windows: {start} → {end} (seq_len={seq_len}, "
                f"pred_len={pred_len})"
            )
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    train_x, train_y = _make_array(df["datetime"].min(), fold["train_end"])
    val_x,   val_y   = _make_array(fold["val_start"],  fold["val_end"])
    test_x,  test_y  = _make_array(fold["test_start"], fold["test_end"])

    log.info(
        f"  train={len(train_x):,}  val={len(val_x):,}  "
        f"test={len(test_x):,}  |  pred_len={pred_len}"
    )
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x_enc=x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, total = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x_enc=x)
        total += criterion(pred, y).item()
        preds.append(pred.cpu().numpy())
        ys.append(y.cpu().numpy())
    return total / max(len(loader), 1), np.concatenate(preds), np.concatenate(ys)


# ── Evaluation metrics for probabilistic forecasts ──────────────────────────

def pinball_score(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    errors = y_true - y_pred
    return float(np.mean(np.maximum(q * errors, (q - 1) * errors)))


def interval_coverage(
    y_true: np.ndarray, p_low: np.ndarray, p_high: np.ndarray
) -> float:
    covered = np.sum((y_true >= p_low) & (y_true <= p_high))
    return float(covered / len(y_true))


def interval_width(p_low: np.ndarray, p_high: np.ndarray) -> float:
    return float(np.mean(p_high - p_low))


def winkler_score(
    y_true: np.ndarray,
    p_low: np.ndarray,
    p_high: np.ndarray,
    alpha: float = 0.20,
) -> float:
    width = p_high - p_low
    penalty_low = (2 / alpha) * np.maximum(p_low - y_true, 0)
    penalty_high = (2 / alpha) * np.maximum(y_true - p_high, 0)
    return float(np.mean(width + penalty_low + penalty_high))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--horizon", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log.info(f"Device       : {device}")

    horizons = args.horizon or ALL_HORIZONS
    folds = WALK_FORWARD_FOLDS
    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]
        if not folds:
            raise ValueError(f"--fold {args.fold} not found")

    log.info("Loading data ...")
    df = load_vayumithra_data()
    log.info(f"Data: {len(df):,} rows, {len([c for c in df.columns if c != 'datetime'])} features")

    save_dir = Path("results/models/probabilistic")
    save_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        log.info(f"\n{'=' * 60}")
        log.info(f"FOLD {fold['fold']}")

        for h in horizons:
            log.info(f"\n  ── Horizon t+{h}h ──")

            (train_x, train_y), (val_x, val_y), (test_x, test_y) = build_datasets(
                df, fold, seq_len=336, pred_len=h,
            )

            train_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(train_x), torch.from_numpy(train_y)
            )
            val_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(val_x), torch.from_numpy(val_y)
            )
            test_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(test_x), torch.from_numpy(test_y)
            )

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=0, pin_memory=False,
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=args.batch_size * 2,
                shuffle=False, num_workers=0, pin_memory=False,
            )
            test_loader = torch.utils.data.DataLoader(
                test_ds, batch_size=args.batch_size * 2,
                shuffle=False, num_workers=0, pin_memory=False,
            )

            cfg = Config(pred_len=h, enc_in=train_x.shape[2])
            model = iTransformerNHiTS_Probabilistic(cfg)
            n_params = sum(p.numel() for p in model.parameters())
            log.info(f"  Parameters: {n_params:,}")

            model = model.to(device)
            criterion = PinballLoss(quantiles=QUANTILES).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )

            best_val_loss = float("inf")
            patience_ctr = 0
            t0 = time.time()

            for epoch in range(1, args.epochs + 1):
                tr_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device
                )
                va_loss, va_preds, va_y = evaluate(
                    model, val_loader, criterion, device
                )
                scheduler.step()

                if epoch % 5 == 0 or epoch == 1:
                    print(
                        f"  Ep {epoch:3d}/{args.epochs} | "
                        f"tr={tr_loss:.4f}  va={va_loss:.4f}  "
                        f"[{time.time()-t0:.0f}s]"
                    )

                if va_loss < best_val_loss:
                    best_val_loss = va_loss
                    patience_ctr = 0
                    ckpt_path = (
                        save_dir
                        / f"probabilistic_fold{fold['fold']}_h{h}.pt"
                    )
                    torch.save(
                        {"model_state": model.state_dict(), "epoch": epoch},
                        ckpt_path,
                    )
                else:
                    patience_ctr += 1
                    if patience_ctr >= args.patience:
                        print(f"  Early stop at epoch {epoch}")
                        break

            elapsed = time.time() - t0

            # ── Reload best and evaluate on test ─────────────────────────────
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])

            # Validation metrics
            va_loss, va_preds, va_y = evaluate(model, val_loader, criterion, device)

            # Test metrics
            te_loss, te_preds, te_y = evaluate(model, test_loader, criterion, device)

            # Pred shapes: (N, 1, 4) → squeeze horizon dim → (N, 4)
            va_preds = va_preds.squeeze(1)  # (N, 4)  — P10, P50, P90, P99
            te_preds = te_preds.squeeze(1)
            va_y_1d = va_y.squeeze(1)
            te_y_1d = te_y.squeeze(1)

            results = {
                "horizon": h,
                "fold": fold["fold"],
                "val_pinball": float(va_loss),
                "test_pinball": float(te_loss),
                "n_params": n_params,
                "runtime_sec": elapsed,
                "best_epoch": ckpt["epoch"],
            }

            # Per-quantile test metrics
            for qi, q in enumerate(QUANTILES):
                results[f"test_pinball_P{int(q*100)}"] = pinball_score(
                    te_y_1d, te_preds[:, qi], q
                )

            # P50 MAE / RMSE
            p50 = te_preds[:, QUANTILES.index(0.50)]
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            results["P50_MAE"] = float(mean_absolute_error(te_y_1d, p50))
            results["P50_RMSE"] = float(np.sqrt(mean_squared_error(te_y_1d, p50)))

            # Coverage P10–P90
            p10 = te_preds[:, QUANTILES.index(0.10)]
            p90 = te_preds[:, QUANTILES.index(0.90)]
            results["coverage_P10_P90"] = interval_coverage(te_y_1d, p10, p90)
            results["width_P10_P90"] = interval_width(p10, p90)
            results["winkler_score"] = winkler_score(te_y_1d, p10, p90)

            # Save individual result
            result_path = save_dir / f"fold{fold['fold']}_h{h}_results.json"
            with open(result_path, "w") as f:
                json.dump(results, f, indent=2)

            # Print summary
            print(
                f"  ✅ P50 MAE={results['P50_MAE']:.4f}  R²≈N/A  "
                f"Coverage={results['coverage_P10_P90']:.3f}  "
                f"Width={results['width_P10_P90']:.3f}  "
                f"[{elapsed:.0f}s]"
            )

    log.info("\nDone.")
