#!/usr/bin/env python3
"""
scripts/train_mamba.py
──────────────────────
Mamba walk-forward training. Run on remote desktop (RTX 4060).

Usage:
    python scripts/train_mamba.py
    python scripts/train_mamba.py --feature_mode raw
    python scripts/train_mamba.py --fold 3 --device cpu --epochs 5   # smoke test

NOTE: mamba-ssm must be installed for GPU-accelerated SSM.
      Without it, SimpleMambaBlock (pure PyTorch fallback) is used.
      Install: pip install mamba-ssm causal-conv1d  (requires CUDA, Linux)
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
from src.pipeline.ts_dataset import (
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
    parser.add_argument("--config",       type=Path, default=Path("configs/mamba_config.yaml"))
    parser.add_argument("--device",       type=str,  default=None)
    parser.add_argument("--fold",         type=int,  default=None)
    parser.add_argument("--feature_mode", type=str,  default=None,
                        choices=["raw", "engineered"])
    parser.add_argument("--epochs",       type=int,  default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mp  = cfg["model_params"]
    tp  = cfg["training"]

    if args.epochs:
        tp["epochs"] = args.epochs
    feature_mode = args.feature_mode or cfg.get("feature_mode", "engineered")

    device_str = args.device or tp.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    log.info(f"Device       : {device}")
    log.info(f"Feature mode : {feature_mode}")
    log.info(f"mamba-ssm    : {'✅ available' if MAMBA_AVAILABLE else '⚠️  NOT installed — using SimpleMambaBlock fallback'}")
    if not MAMBA_AVAILABLE:
        log.warning("To install: pip install mamba-ssm causal-conv1d  (needs CUDA + Linux)")

    log.info("Loading parquet ...")
    df = load_raw(Path(cfg["data"]["parquet_path"]))

    log.info("Engineering features ...")
    t0 = time.time()
    df = build_full_feature_matrix(df, verbose=True)
    log.info(f"Feature matrix: {df.shape}  [{time.time()-t0:.0f}s]")

    horizons     = cfg["horizons"]
    seq_len      = mp["seq_len"]
    patch_size   = mp["patch_size"]
    save_dir     = Path(cfg["output"]["model_dir"])
    metrics_path = Path(cfg["output"]["metrics_path"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    folds = WALK_FORWARD_FOLDS
    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]
        if not folds:
            raise ValueError(f"--fold {args.fold} not found")

    all_metrics: list[dict] = []

    for fold in folds:
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold['fold']}  |  train < {fold['train_end']}  "
                 f"val {fold['val_start']} – {fold['val_end']}")

        for h_idx, h in enumerate(horizons):
            log.info(f"\n  ── Horizon t+{h}h ──")

            variates = get_feature_variates(h) if feature_mode == "engineered" \
                       else SEQUENCE_VARIATES
            n_in = len(variates)

            train_ds, val_ds = build_fold_datasets(
                df, fold, seq_len=seq_len, horizons=horizons, variates=variates,
            )

            pin = device.type == "cuda"
            train_loader = DataLoader(train_ds, batch_size=tp["batch_size"],
                                      shuffle=True, num_workers=4, pin_memory=pin)
            val_loader   = DataLoader(val_ds,   batch_size=tp["batch_size"] * 2,
                                      shuffle=False, num_workers=4, pin_memory=pin)

            model = MambaForecaster(
                seq_len    = seq_len,
                pred_len   = 1,
                enc_in     = n_in,
                d_model    = mp["d_model"],
                d_state    = mp["d_state"],
                d_conv     = mp["d_conv"],
                expand     = mp["expand"],
                n_layers   = mp["n_layers"],
                patch_size = patch_size,
                dropout    = mp["dropout"],
            )
            n_params = sum(p.numel() for p in model.parameters())
            log.info(f"  Parameters : {n_params:,} | Input channels: {n_in}")

            tag = f"mamba_{feature_mode}"
            metrics = train_model_for_fold(
                model=model, train_loader=train_loader, val_loader=val_loader,
                cfg=tp, horizon=h, horizon_idx=h_idx,
                fold=fold["fold"], model_name=tag,
                save_dir=save_dir, device=device,
            )
            metrics["feature_mode"] = feature_mode
            all_metrics.append(metrics)
            pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)

    results_df = pd.DataFrame(all_metrics)
    log.info("\n" + "="*60)
    log.info(f"FINAL Mamba [{feature_mode}] RESULTS")
    summary = results_df.groupby("horizon").agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"),
        runtime_mean=("runtime_sec", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))
    log.info(f"\n✅ Saved → {metrics_path}")


if __name__ == "__main__":
    main()
