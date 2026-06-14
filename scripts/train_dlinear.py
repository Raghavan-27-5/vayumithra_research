#!/usr/bin/env python3
"""
scripts/train_dlinear.py
────────────────────────
DLinear walk-forward training. Run on remote desktop (RTX 4060).

Usage:
    # Full run (all folds, all horizons, engineered features):
    python scripts/train_dlinear.py

    # Raw sequences only (paper default):
    python scripts/train_dlinear.py --feature_mode raw

    # Single fold smoke test on CPU:
    python scripts/train_dlinear.py --fold 3 --device cpu --epochs 5

Outputs (crash-safe — CSV saved after every horizon):
    results/metrics/dlinear_results.csv
    results/models/dlinear/*.pt         — best checkpoint per (fold, horizon)
    results/models/dlinear/*_history.json
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw
from src.models.dlinear import DLinear
from src.pipeline.feature_pipeline import (
    WALK_FORWARD_FOLDS,
    build_full_feature_matrix,
)
from src.pipeline.trainer import train_model_for_fold
from src.pipeline.ts_dataset import (
    N_VARIATES,
    SEQUENCE_VARIATES,
    build_fold_datasets,
    get_feature_variates,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=Path, default=Path("configs/dlinear_config.yaml"))
    parser.add_argument("--device",       type=str,  default=None)
    parser.add_argument("--fold",         type=int,  default=None, help="Run only fold 1/2/3")
    parser.add_argument("--feature_mode", type=str,  default=None,
                        choices=["raw", "engineered"],
                        help="Override config feature_mode: raw=11 variates, engineered=50+ features")
    parser.add_argument("--epochs",       type=int,  default=None, help="Override training epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mp  = cfg["model_params"]
    tp  = cfg["training"]

    # CLI overrides
    if args.epochs:
        tp["epochs"] = args.epochs
    feature_mode = args.feature_mode or cfg.get("feature_mode", "engineered")

    device_str = args.device or tp.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    log.info(f"Device       : {device}")
    log.info(f"Feature mode : {feature_mode}")

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("Loading parquet ...")
    df = load_raw(Path(cfg["data"]["parquet_path"]))

    log.info("Engineering features ...")
    t0 = time.time()
    df = build_full_feature_matrix(df, verbose=True)
    log.info(f"Feature matrix: {df.shape}  [{time.time()-t0:.0f}s]")

    horizons     = cfg["horizons"]
    seq_len      = mp["seq_len"]
    save_dir     = Path(cfg["output"]["model_dir"])
    metrics_path = Path(cfg["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    folds = WALK_FORWARD_FOLDS
    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]
        if not folds:
            raise ValueError(f"--fold {args.fold} not found (must be 1, 2, or 3)")

    all_metrics: list[dict] = []

    for fold in folds:
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold['fold']}  |  train < {fold['train_end']}  "
                 f"val {fold['val_start']} – {fold['val_end']}")

        for h_idx, h in enumerate(horizons):
            log.info(f"\n  ── Horizon t+{h}h ──")

            # Select feature set — horizon-aware for engineered mode
            if feature_mode == "engineered":
                variates = get_feature_variates(h)
            else:
                variates = SEQUENCE_VARIATES

            n_in = len(variates)
            log.info(f"  Input variates: {n_in}")

            # Build datasets (per horizon to get correct variate set)
            train_ds, val_ds = build_fold_datasets(
                df, fold,
                seq_len=seq_len,
                horizons=horizons,
                variates=variates,
            )

            pin = device.type == "cuda"
            train_loader = DataLoader(train_ds, batch_size=tp["batch_size"],
                                      shuffle=True, num_workers=4, pin_memory=pin)
            val_loader   = DataLoader(val_ds,   batch_size=tp["batch_size"] * 2,
                                      shuffle=False, num_workers=4, pin_memory=pin)

            model = DLinear(
                seq_len    = seq_len,
                pred_len   = 1,
                enc_in     = n_in,
                kernel_size= mp["kernel_size"],
                individual = mp["individual"],
            )

            tag = f"dlinear_{feature_mode}"
            metrics = train_model_for_fold(
                model=model, train_loader=train_loader, val_loader=val_loader,
                cfg=tp, horizon=h, horizon_idx=h_idx,
                fold=fold["fold"], model_name=tag,
                save_dir=save_dir, device=device,
            )
            metrics["feature_mode"] = feature_mode
            all_metrics.append(metrics)
            pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
            log.info(f"  → {metrics_path}")

    results_df = pd.DataFrame(all_metrics)
    log.info("\n" + "="*60)
    log.info(f"FINAL DLinear [{feature_mode}] RESULTS")
    summary = results_df.groupby("horizon").agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))
    log.info(f"\n✅ Saved → {metrics_path}")


if __name__ == "__main__":
    main()
