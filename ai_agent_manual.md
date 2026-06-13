# AI Agent Manual — Vayumithra Remote Desktop Training Guide

> **Audience:** AI agent or human operator on the remote Windows desktop  
> **Machine:** i9 CPU · 64 GB RAM · RTX 4060 16 GB VRAM  
> **Repo:** https://github.com/Raghavan-27-5/vayumithra_research

---

## WHAT THIS REPO IS

A wind speed forecasting research stack comparing:
- **LGBM** (already trained — weights in `results/models/lgbm/`)
- **DLinear** — simple linear decomposition model (train this)
- **Mamba** — selective state space model (train this)
- **KAN** — Kolmogorov-Arnold Networks (optional, train if time permits)
- **Hybrid ensembles** — weighted avg, residual correction, stacked meta-learner

The **LGBM models are the baseline to beat**. See the table in `README.md` for exact scores.

---

## STEP 1 — One-Time Environment Setup

Open a terminal (PowerShell or CMD) in the repo directory.

### 1a. Install CUDA PyTorch (do this first, separately)
```powershell
pip install torch==2.2.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

### 1b. Install remaining dependencies
```powershell
pip install -r requirements_windows.txt
```

### 1c. Install Mamba SSM (requires WSL2 or native Linux; skip on pure Windows)
```bash
# In WSL2 terminal:
pip install mamba-ssm causal-conv1d
```
> **NOTE:** If `mamba-ssm` is unavailable (pure Windows), the code automatically falls back  
> to `SimpleMambaBlock` (CPU-compatible). Training will work but be slower and use less memory.

### 1d. Verify GPU is visible
```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 4060
```

---

## STEP 2 — Get the Code and Data

```powershell
git clone https://github.com/Raghavan-27-5/vayumithra_research.git
cd vayumithra_research
git pull origin main
```

The parquet data file is **already in the repo** at `data/processed/wind_data.parquet` (25 MB).  
No CSV extraction needed.

---

## STEP 3 — Verify Everything Is Ready

```powershell
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/wind_data.parquet')
print('Shape:', df.shape)
print('Date range:', df['datetime'].min(), '->', df['datetime'].max())
print('Stations:', df['Index'].nunique())
"
```
Expected output:
```
Shape: (2646408, 13)
Date range: 2013-01-01 05:00:00 -> 2022-09-28 04:00:00
Stations: 31
```

---

## STEP 4 — Train DLinear (Start Here)

DLinear is the simplest deep learning model. Train it first to verify the pipeline works.

### Run all 3 folds, all horizons:
```powershell
python scripts/train_dlinear.py --config configs/dlinear_config.yaml
```

### Run a single fold (faster for debugging):
```powershell
python scripts/train_dlinear.py --config configs/dlinear_config.yaml --fold 3
```

### Quick CPU smoke test (no GPU needed):
```powershell
python scripts/train_dlinear.py --config configs/dlinear_config.yaml --fold 3 --device cpu
```

**Outputs:**
- `results/metrics/dlinear_results.csv` — metrics per fold and horizon (updated after each horizon)
- `results/models/dlinear/*.pt` — best checkpoint per (fold, horizon)
- `results/models/dlinear/*_history.json` — loss/mae curves

**Expected training time on RTX 4060:** ~20-40 min total (all folds × horizons)

---

## STEP 5 — Train Mamba

```powershell
python scripts/train_mamba.py --config configs/mamba_config.yaml
```

Same flags as DLinear (`--fold 3`, `--device cpu`).

**Expected training time on RTX 4060:** ~60-90 min total

> **If mamba-ssm is not installed:** The script will print a warning and use `SimpleMambaBlock`  
> (pure PyTorch fallback). Results will be valid but ~3–5× slower.

---

## STEP 6 — Push Results Back

After each training run completes, push the results CSV (NOT the .pt weight files):

```powershell
git add results/metrics/
git commit -m "results: dlinear all folds complete"
git push origin main
```

> ⚠️ Do NOT `git add results/models/` — PyTorch .pt files are gitignored (too large).  
> Only `results/models/lgbm/*.txt` files are committed.

---

## STEP 7 — Generate Benchmark Comparison

After both DLinear and Mamba are trained:

```powershell
python scripts/compare_models.py
```

This generates `results/metrics/benchmark_table.csv` and prints the selection memo.  
Push this file too:

```powershell
git add results/metrics/benchmark_table.csv
git commit -m "results: benchmark table updated"
git push origin main
```

---

## STEP 8 — Optional: Train KAN

```powershell
pip install pykan
# KAN training script coming in next iteration — placeholder
```

---

## HARD CONSTRAINTS — DO NOT VIOLATE

| Rule | Detail |
|------|--------|
| **No random splits** | All folds are chronological — see `manifests/fold_manifest.yaml` |
| **No data leakage** | Windows end at t−1; target is at t+h |
| **Same folds for all models** | Fold dates are hardcoded in `src/pipeline/feature_pipeline.py` |
| **Scaler fit on train only** | `build_fold_datasets()` handles this automatically |
| **No test data for tuning** | Year 2022+ is not used in any fold |
| **Stacking uses OOF only** | `src/models/hybrid.py` enforces this |

---

## LGBM BASELINE — SCORES TO BEAT

These are the production LGBM results from `notebooks/wind_forecast.ipynb`.

| Horizon | LGBM MAE | LGBM R² | Skill vs Persistence |
|---------|---------|---------|---------------------|
| t+1h | **0.1193** | **0.9962** | 78.92% |
| t+2h | **0.2722** | **0.9814** | 73.19% |
| t+3h | **0.4260** | **0.9560** | 68.52% |
| t+4h | **0.5629** | **0.9246** | 64.68% |
| t+5h | **0.6802** | **0.8915** | 61.08% |
| t+6h | **0.7771** | **0.8591** | 57.36% |
| t+24h | **1.2782** | **0.6045** | — |
| t+48h | **1.5681** | **0.4172** | — |

**Decision rule:** If a new model's MAE is within 1% of LGBM, call it equivalent — not better.

---

## TROUBLESHOOTING

| Problem | Solution |
|---------|---------|
| `CUDA out of memory` | Reduce `batch_size` in the config YAML (try 128 or 64) |
| `mamba-ssm not found` | Code falls back automatically — continue as normal |
| `No valid windows found` | Check that `data/processed/wind_data.parquet` exists and is 25 MB |
| `ModuleNotFoundError` | Run `pip install -r requirements_windows.txt` again |
| Training crashes mid-fold | Results CSV is saved after each horizon — restart with `--fold N` |
| Git push fails | Check `git status` — do NOT commit `*.pt`, `*.csv` raw data, or `data/raw/` |

---

## FILE REFERENCE

```
Key files to read:
  src/pipeline/feature_pipeline.py   — Canonical feature engineering (mirrors notebook exactly)
  src/pipeline/ts_dataset.py         — How windows are built (causality logic is here)
  src/pipeline/trainer.py            — Training loop, checkpoint saving, metrics
  src/models/dlinear.py              — DLinear architecture
  src/models/mamba_ts.py             — Mamba architecture
  src/models/hybrid.py               — Ensemble strategies
  manifests/feature_manifest.yaml   — Every feature explained
  manifests/fold_manifest.yaml       — Walk-forward fold definitions

Key files to NOT edit unless instructed:
  manifests/fold_manifest.yaml       — DO NOT change fold dates
  manifests/feature_manifest.yaml   — DO NOT add features without ablation
  notebooks/wind_forecast.ipynb      — This is the source of truth; read only
```
