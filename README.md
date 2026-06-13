# Vayumithra Research — Wind Speed Forecasting

> Multi-horizon wind speed forecasting for Indian coastal stations (2013–2022).  
> Benchmarking **LGBM · DLinear · Mamba · KAN · Hybrid ensembles** under strict walk-forward validation.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)

---

## 📂 Project Structure

```
vayumithra_research/
├── notebooks/
│   ├── 2013_2022.ipynb          # Exploratory analysis + initial feature study
│   └── wind_forecast.ipynb      # Canonical LGBM pipeline (SOURCE OF TRUTH)
├── data/
│   ├── raw/                     # Raw CSV (gitignored — copy manually)
│   └── processed/
│       └── wind_data.parquet    # 25 MB — committed to git ✅
├── src/
│   ├── pipeline/
│   │   ├── feature_pipeline.py  # Canonical feature engineering (mirrors notebook exactly)
│   │   ├── ts_dataset.py        # Sliding-window PyTorch Dataset for DL models
│   │   └── trainer.py           # Universal training loop (DLinear / Mamba / KAN)
│   ├── models/
│   │   ├── dlinear.py           # DLinear + NLinear (Zeng et al. 2022)
│   │   ├── mamba_ts.py          # Mamba Forecaster (Gu & Dao 2023)
│   │   ├── kan_ts.py            # KAN Forecaster (Liu et al. 2024)
│   │   └── hybrid.py            # WeightedAvg · ResidualCorrector · StackedMeta
│   └── utils/metrics.py         # MAE, RMSE, R², Skill Score
├── scripts/
│   ├── convert_to_parquet.py    # One-time CSV → Parquet conversion
│   ├── train_dlinear.py         # DLinear walk-forward training
│   ├── train_mamba.py           # Mamba walk-forward training
│   └── compare_models.py        # Aggregate results → benchmark table
├── configs/
│   ├── lgbm_config.yaml
│   ├── dlinear_config.yaml
│   └── mamba_config.yaml
├── manifests/
│   ├── feature_manifest.yaml    # Every feature + notebook cell reference
│   └── fold_manifest.yaml       # Canonical walk-forward fold definitions
├── results/
│   ├── models/lgbm/             # LGBM booster .txt files (committed ✅)
│   └── metrics/                 # CSV result tables (committed after training ✅)
├── requirements.txt             # CPU / general dependencies
├── requirements_windows.txt     # Remote desktop (RTX 4060 + CUDA)
├── ai_agent_manual.md           # Step-by-step guide for remote desktop agent
└── README.md
```

---

## 🗄️ Dataset

| Property | Value |
|---|---|
| Source | IMD weather station network |
| Region | Indian peninsula (Lat: 2–18°N, Lon: 72–88°E) |
| Period | 2013-01-01 05:00 → 2022-09-28 04:00 |
| Rows | 2,646,408 |
| Stations | 31 unique coordinate groups |
| Frequency | Hourly |
| Target | `wind_speed` (m/s) at multiple future horizons |

---

## 🏆 LGBM Benchmark — Scores to Beat

These are the **production LGBM metrics** from `wind_forecast.ipynb` (walk-forward, 3 folds, physics features).  
All new models (DLinear, Mamba, KAN, hybrids) **must beat these numbers to claim improvement**.

### Short-Horizon (t+1h to t+6h) — Mean over 3 Walk-Forward Folds

| Horizon | MAE ↓ | R² ↑ | Persistence Skill ↑ |
|---------|-------|------|---------------------|
| **t+1h** | **0.1193** | **0.9962** | 78.92% |
| **t+2h** | **0.2722** | **0.9814** | 73.19% |
| **t+3h** | **0.4260** | **0.9560** | 68.52% |
| **t+4h** | **0.5629** | **0.9246** | 64.68% |
| **t+5h** | **0.6802** | **0.8915** | 61.08% |
| **t+6h** | **0.7771** | **0.8591** | 57.36% |

### Long-Horizon (Spatial Features, Single Split 2021-val)

| Horizon | MAE ↓ | R² ↑ | Notes |
|---------|-------|------|-------|
| **t+24h** | **1.2782** | **0.6045** | Spatial context features included |
| **t+48h** | **1.5681** | **0.4172** | Spatial context features included |

### Persistence Baseline (what NOT to compare against — already beaten)

| Horizon | Persistence MAE |
|---------|----------------|
| t+1h | 0.5398 |
| t+4h | 1.5949 |
| t+24h | ~1.950 |
| t+48h | ~2.100 |

### Saved Model Weights

```
results/models/lgbm/
├── lgbm_short_t1h.txt    # t+1h production model
├── lgbm_short_t2h.txt    # t+2h
├── lgbm_short_t3h.txt    # t+3h
├── lgbm_short_t4h.txt    # t+4h
├── lgbm_short_t5h.txt    # t+5h
├── lgbm_short_t6h.txt    # t+6h
└── lgbm_physics_t1h.txt  # t+1h with ERA5 physics features
```

Load with:
```python
import lightgbm as lgb
model = lgb.Booster(model_file="results/models/lgbm/lgbm_short_t1h.txt")
```

---

## 🔄 Walk-Forward Validation Protocol (identical across ALL models)

| Fold | Train Period | Validation Year |
|------|-------------|-----------------|
| 1 | 2013-01-01 → 2019-01-01 | 2019 |
| 2 | 2013-01-01 → 2020-01-01 | 2020 |
| 3 | 2013-01-01 → 2021-01-01 | 2021 |

**Rules:** No random splits · Train strictly precedes val · Scalers fit on train only · Same folds for ALL models

---

## ⚙️ Setup

### Local machine (code, no training)
```bash
git clone https://github.com/Raghavan-27-5/vayumithra_research.git
cd vayumithra_research
pip install -r requirements.txt
```

### Remote Desktop (RTX 4060, training)
See **[ai_agent_manual.md](ai_agent_manual.md)** for the complete step-by-step guide.

---

## 🔄 Git Workflow

```bash
# ── Local: push code changes ─────────────────────────────────────────────────
git add .
git commit -m "feat: update mamba config"
git push origin main

# ── Remote Desktop: pull and train ───────────────────────────────────────────
git pull origin main
python scripts/train_dlinear.py --config configs/dlinear_config.yaml
git add results/metrics/
git commit -m "results: dlinear fold-3 complete"
git push origin main

# ── Local: check results ──────────────────────────────────────────────────────
git pull origin main
python scripts/compare_models.py
```

---

## 🧠 Models

| Model | Type | Input | Parameters | Notes |
|-------|------|-------|-----------|-------|
| **LGBM** | Gradient Boosting | 80+ tabular features | ~500 trees | Baseline — already trained |
| **DLinear** | Linear + Decomposition | Raw time series (L=336h) | 2×L×T per variate | Trend + seasonal branches |
| **Mamba** | Selective SSM | Patched time series | ~500K | Requires CUDA |
| **KAN** | Spline Networks | Raw time series | ~100K | Interpretable |
| **Hybrids** | Ensemble | OOF predictions | — | 3 fusion strategies |

---

## 📖 References

1. Zeng, A. et al. *"Are Transformers Effective for Time Series Forecasting?"* arXiv:2205.13504 (2022)
2. Gu, A. & Dao, T. *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"* arXiv:2312.00752 (2023)
3. Liu, Z. et al. *"KAN: Kolmogorov-Arnold Networks"* arXiv:2404.19756 (2024)
