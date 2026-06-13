"""
src/pipeline/trainer.py
────────────────────────
Universal training loop for PyTorch models (DLinear, Mamba, KAN).
Designed for remote GPU execution with complete artifact saving.

Saves:
  - Best model checkpoint (.pt) per (model, fold, horizon)
  - Training history JSON per run
  - Returns metrics dict for result aggregation
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer / scheduler builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
        betas=tuple(cfg.get("betas", [0.9, 0.999])),
    )


def _make_scheduler(optimizer, cfg: dict, total_steps: int):
    name = cfg.get("lr_scheduler", "cosine")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=1e-6
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch loops
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pred(pred: torch.Tensor) -> torch.Tensor:
    """Normalise model output to shape (B,) — handles (B,T,C), (B,C), (B,)."""
    if pred.dim() == 3:
        return pred[:, 0, 0]
    if pred.dim() == 2:
        return pred[:, 0]
    return pred


def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    horizon_idx: int,
    scheduler=None,
    clip_grad: float = 1.0,
) -> float:
    model.train()
    total = 0.0
    n     = 0
    for x, y in loader:
        x   = x.to(device)
        y_h = y[:, horizon_idx].to(device)
        optimizer.zero_grad()
        loss = criterion(_extract_pred(model(x)), y_h)
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item()
        n     += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    horizon_idx: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, total = [], [], 0.0
    for x, y in loader:
        x   = x.to(device)
        y_h = y[:, horizon_idx]
        p   = _extract_pred(model(x)).cpu()
        total += criterion(p, y_h).item()
        preds.append(p.numpy())
        ys.append(y_h.numpy())
    return total / max(len(loader), 1), np.concatenate(preds), np.concatenate(ys)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    y_persist: np.ndarray | None = None,
) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    out  = {"mae": mae, "rmse": rmse, "r2": r2}
    if y_persist is not None:
        p_mae = float(mean_absolute_error(y_true, y_persist))
        out["persist_mae"] = p_mae
        out["skill"]       = 1.0 - mae / p_mae if p_mae > 0 else 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_model_for_fold(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          dict,
    horizon:      int,
    horizon_idx:  int,
    fold:         int,
    model_name:   str,
    save_dir:     Path,
    device:       torch.device,
) -> dict:
    """
    Full training loop for one (model, fold, horizon) triple.
    - Saves best checkpoint by validation loss with early stopping.
    - Saves training history as JSON.
    - Returns a metrics dict ready for CSV aggregation.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs   = cfg.get("epochs", 100)
    patience = cfg.get("patience", 10)
    loss_fn  = cfg.get("loss", "mse")

    optimizer  = _make_optimizer(model, cfg)
    criterion  = nn.MSELoss() if loss_fn == "mse" else nn.L1Loss()
    scheduler  = _make_scheduler(optimizer, cfg, epochs * len(train_loader))

    model.to(device)
    best_val_loss = float("inf")
    patience_ctr  = 0
    history: list[dict] = []
    ckpt_path = save_dir / f"{model_name}_fold{fold}_h{horizon}.pt"
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, horizon_idx, scheduler
        )
        va_loss, va_preds, va_y = evaluate_epoch(
            model, val_loader, criterion, device, horizon_idx
        )
        va_mae = float(mean_absolute_error(va_y, va_preds))
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_mae": va_mae})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{epochs} | "
                  f"tr={tr_loss:.4f}  va={va_loss:.4f}  mae={va_mae:.4f} "
                  f"[{time.time()-t0:.0f}s]")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_ctr  = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": va_loss, "val_mae": va_mae,
                        "cfg": cfg, "horizon": horizon, "fold": fold}, ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stop at epoch {epoch} (patience={patience})")
                break

    elapsed = time.time() - t0

    # Reload best weights and evaluate
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, best_preds, best_y = evaluate_epoch(
        model, val_loader, criterion, device, horizon_idx
    )

    metrics = compute_metrics(best_y, best_preds)
    metrics.update({
        "fold": fold, "horizon": horizon, "model": model_name,
        "best_epoch":  ckpt["epoch"],
        "runtime_sec": elapsed,
        "n_params":    sum(p.numel() for p in model.parameters()),
        "checkpoint":  str(ckpt_path),
    })

    # Save history
    with open(save_dir / f"{model_name}_fold{fold}_h{horizon}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"  ✅ MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f} "
          f"[fold={fold} h={horizon}  params={metrics['n_params']:,}  {elapsed:.0f}s]")
    return metrics
