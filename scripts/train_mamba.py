#!/usr/bin/env python3
"""
scripts/train_mamba.py
──────────────────────
Mamba walk-forward training script. Run on remote desktop (RTX 4060).

Usage:
    python scripts/train_mamba.py --config configs/mamba_config.yaml
    python scripts/train_mamba.py --fold 3 --device cpu   # CPU smoke test

Outputs:
    results/metrics/mamba_results.csv
    results/models/mamba/  — .pt checkpoints + *_history.json files
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
from src.models.mamba_ts import MAMBA_AVAILABLE, MambaForecaster
from src.pipeline.feature_pipeline import WALK_FORWARD_FOLDS, build_full_feature_matrix
from src.pipeline.trainer import train_model_for_fold
from src.pipeline.ts_dataset import N_VARIATES, SEQUENCE_VARIATES, build_fold_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Mamba — wind forecasting")
    parser.add_argument("--config", type=Path, default=Path("configs/mamba_config.yaml"))
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: cuda / cpu")
    parser.add_argument("--fold",   type=int, default=None,
                        help="Run only this fold number (1/2/3); default=all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mp  = cfg["model_params"]
    tp  = cfg["training"]

    device_str = args.device or tp.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    log.info(f"Device      : {device}")
    log.info(f"mamba-ssm   : {'available ✅' if MAMBA_AVAILABLE else 'NOT installed — using CPU fallback ⚠️'}")
    if not MAMBA_AVAILABLE and device.type == "cuda":
        log.warning("Install mamba-ssm for full CUDA performance: "
                    "pip install mamba-ssm causal-conv1d")

    # ── Load + engineer data ──────────────────────────────────────────────────
    log.info("Loading parquet ...")
    df = load_raw(Path(cfg["data"]["parquet_path"]))

    log.info("Building feature matrix (needed for target columns + time features) ...")
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
            raise ValueError(f"--fold {args.fold} not found. Must be 1, 2, or 3.")

    all_metrics: list[dict] = []

    for fold in folds:
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold['fold']}  |  "
                 f"train < {fold['train_end']}  "
                 f"val {fold['val_start']} – {fold['val_end']}")

        log.info("Building sliding-window datasets ...")
        train_ds, val_ds = build_fold_datasets(
            df, fold,
            seq_len=seq_len,
            horizons=horizons,
            variates=SEQUENCE_VARIATES,
        )
        log.info(f"  train={len(train_ds):,}  val={len(val_ds):,} windows")

        pin = device.type == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=tp["batch_size"], shuffle=True,
            num_workers=4, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds, batch_size=tp["batch_size"] * 2, shuffle=False,
            num_workers=4, pin_memory=pin,
        )

        for h_idx, h in enumerate(horizons):
            log.info(f"\n  ── Horizon t+{h}h ──")

            model = MambaForecaster(
                seq_len    = seq_len,
                pred_len   = 1,
                enc_in     = N_VARIATES,
                d_model    = mp["d_model"],
                d_state    = mp["d_state"],
                d_conv     = mp["d_conv"],
                expand     = mp["expand"],
                n_layers   = mp["n_layers"],
                patch_size = mp["patch_size"],
                dropout    = mp["dropout"],
            )
            n_params = sum(p.numel() for p in model.parameters())
            log.info(f"  Parameters: {n_params:,}")

            metrics = train_model_for_fold(
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                cfg          = tp,
                horizon      = h,
                horizon_idx  = h_idx,
                fold         = fold["fold"],
                model_name   = "mamba",
                save_dir     = save_dir,
                device       = device,
            )
            all_metrics.append(metrics)
            # Save running CSV after every horizon — safe to interrupt
            pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
            log.info(f"  → {metrics_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_metrics)
    log.info("\n" + "="*60)
    log.info("FINAL Mamba RESULTS  (mean over folds)")
    log.info("="*60)
    summary = results_df.groupby("horizon").agg(
        mae_mean=("mae",  "mean"), mae_std=("mae",  "std"),
        rmse_mean=("rmse","mean"),
        r2_mean=("r2",   "mean"), r2_std=("r2",   "std"),
        runtime_sec=("runtime_sec","mean"),
    ).reset_index()
    print(summary.to_string(index=False))
    log.info(f"\n✅ Results → {metrics_path}")


if __name__ == "__main__":
    main()
