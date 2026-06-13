#!/usr/bin/env python3
"""
scripts/train_dlinear.py
────────────────────────
DLinear walk-forward training script. Run on the remote desktop (RTX 4060).

Usage:
    python scripts/train_dlinear.py --config configs/dlinear_config.yaml
    python scripts/train_dlinear.py --config configs/dlinear_config.yaml --device cpu  # smoke test
    python scripts/train_dlinear.py --fold 1   # run a single fold

Outputs (all in results/):
    results/metrics/dlinear_results.csv       — full fold × horizon metrics
    results/models/dlinear/                   — .pt checkpoints + history.json files
"""
import argparse
import sys
import time
import logging
from pathlib import Path

import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader

# ── Make sure repo root is on path ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw
from src.pipeline.feature_pipeline import build_full_feature_matrix, WALK_FORWARD_FOLDS
from src.pipeline.ts_dataset import build_fold_datasets, SEQUENCE_VARIATES, N_VARIATES
from src.models.dlinear import DLinear
from src.pipeline.trainer import train_model_for_fold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/dlinear_config.yaml")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DLinear — wind forecasting")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold (1/2/3); default=all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mp  = cfg["model_params"]
    tp  = cfg["training"]

    # ── Device ────────────────────────────────────────────────────────────────
    device_str = args.device or tp.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Load and engineer data ────────────────────────────────────────────────
    log.info("Loading parquet ...")
    df = load_raw(Path(cfg["data"]["parquet_path"]))

    log.info("Building feature matrix ...")
    t0 = time.time()
    df = build_full_feature_matrix(df, verbose=True)
    log.info(f"Feature matrix ready: {df.shape}  [{time.time()-t0:.0f}s]")

    horizons     = cfg["horizons"]
    seq_len      = mp["seq_len"]
    save_dir     = Path(cfg["output"]["model_dir"])
    metrics_path = Path(cfg["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    folds = WALK_FORWARD_FOLDS
    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]
        if not folds:
            raise ValueError(f"--fold {args.fold} not in [1,2,3]")

    all_metrics: list[dict] = []

    for fold in folds:
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold['fold']}  train < {fold['train_end']}  |  val {fold['val_start']}–{fold['val_end']}")

        # ── Build datasets ────────────────────────────────────────────────────
        log.info("Building sliding-window datasets ...")
        train_ds, val_ds = build_fold_datasets(
            df, fold, seq_len=seq_len, horizons=horizons,
            variates=SEQUENCE_VARIATES,
        )
        log.info(f"  train={len(train_ds):,}  val={len(val_ds):,} windows")

        train_loader = DataLoader(
            train_ds, batch_size=tp["batch_size"], shuffle=True,
            num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=tp["batch_size"] * 2, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        # ── One model per horizon ─────────────────────────────────────────────
        for h_idx, h in enumerate(horizons):
            log.info(f"\n  ── Horizon t+{h}h ──")

            model = DLinear(
                seq_len    = seq_len,
                pred_len   = 1,                    # single-step output per horizon
                enc_in     = N_VARIATES,
                kernel_size= mp["kernel_size"],
                individual = mp["individual"],
            )

            metrics = train_model_for_fold(
                model       = model,
                train_loader= train_loader,
                val_loader  = val_loader,
                cfg         = tp,
                horizon     = h,
                horizon_idx = h_idx,
                fold        = fold["fold"],
                model_name  = "dlinear",
                save_dir    = save_dir,
                device      = device,
            )
            all_metrics.append(metrics)

            # Save running results after each horizon
            pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
            log.info(f"  Metrics saved → {metrics_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_metrics)
    log.info("\n" + "="*60)
    log.info("FINAL DLinear RESULTS")
    log.info("="*60)
    summary = results_df.groupby("horizon").agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
    ).reset_index()
    print(summary.to_string(index=False))
    log.info(f"\n✅ All results → {metrics_path}")


if __name__ == "__main__":
    main()
