#!/usr/bin/env python3
"""
scripts/setup_check.py
───────────────────────
Run this FIRST on the remote desktop to verify everything is ready.
It checks GPU, data, imports, and does a 1-batch forward pass sanity test.

Usage:
    python scripts/setup_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = "✅"; FAIL = "❌"; WARN = "⚠️ "

def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS} {label}: {result}")
        return True
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        return False

print("\n" + "="*60)
print("VAYUMITHRA — Remote Desktop Setup Check")
print("="*60)

# ── Python & GPU ──────────────────────────────────────────────────────────────
print("\n[1] Python & GPU")
check("Python version", lambda: sys.version.split()[0])

import torch
check("PyTorch version", lambda: torch.__version__)
check("CUDA available", lambda: f"{torch.cuda.is_available()} | device='{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}'")
if torch.cuda.is_available():
    check("VRAM (GB)", lambda: f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")

# ── mamba-ssm ────────────────────────────────────────────────────────────────
print("\n[2] Optional: mamba-ssm")
try:
    from mamba_ssm import Mamba
    print(f"  {PASS} mamba-ssm: installed — full CUDA Mamba available")
except ImportError:
    print(f"  {WARN} mamba-ssm: NOT installed — SimpleMambaBlock fallback will be used")
    print(f"       Install: pip install mamba-ssm causal-conv1d")

# ── Core dependencies ────────────────────────────────────────────────────────
print("\n[3] Core dependencies")
for pkg in ["pandas", "numpy", "pyarrow", "lightgbm", "sklearn", "yaml", "einops"]:
    try:
        mod = __import__(pkg if pkg != "sklearn" else "sklearn")
        ver = getattr(mod, "__version__", "?")
        print(f"  {PASS} {pkg}: {ver}")
    except ImportError:
        print(f"  {FAIL} {pkg}: NOT installed — run: pip install -r requirements_windows.txt")

# ── Data ──────────────────────────────────────────────────────────────────────
print("\n[4] Data")
pq_path = Path("data/processed/wind_data.parquet")
if pq_path.exists():
    size_mb = pq_path.stat().st_size / 1e6
    import pandas as pd
    df = pd.read_parquet(pq_path)
    print(f"  {PASS} wind_data.parquet: {size_mb:.1f} MB | {df.shape[0]:,} rows | {df['Index'].nunique()} stations")
    print(f"       Date range: {df['datetime'].min().date()} → {df['datetime'].max().date()}")
else:
    print(f"  {FAIL} wind_data.parquet not found at {pq_path}")
    print(f"       It should be committed to git. Run: git pull origin main")

# ── Source imports ───────────────────────────────────────────────────────────
print("\n[5] Source imports")
check("src.data.loader",              lambda: __import__("src.data.loader",              fromlist=["load_raw"]) and "OK")
check("src.pipeline.feature_pipeline",lambda: __import__("src.pipeline.feature_pipeline",fromlist=["build_full_feature_matrix"]) and "OK")
check("src.pipeline.ts_dataset",      lambda: __import__("src.pipeline.ts_dataset",      fromlist=["WindWindowDataset"]) and "OK")
check("src.pipeline.trainer",         lambda: __import__("src.pipeline.trainer",         fromlist=["train_model_for_fold"]) and "OK")
check("src.models.dlinear",           lambda: __import__("src.models.dlinear",           fromlist=["DLinear"]) and "OK")
check("src.models.mamba_ts",          lambda: __import__("src.models.mamba_ts",          fromlist=["MambaForecaster"]) and "OK")
check("src.models.hybrid",            lambda: __import__("src.models.hybrid",            fromlist=["StackedMetaLearner"]) and "OK")

# ── Quick forward pass (DLinear, CPU) ────────────────────────────────────────
print("\n[6] DLinear forward pass (CPU)")
try:
    from src.models.dlinear import DLinear
    model = DLinear(seq_len=32, pred_len=1, enc_in=11)
    x = torch.randn(4, 32, 11)
    out = model(x)
    print(f"  {PASS} DLinear: input={tuple(x.shape)} → output={tuple(out.shape)}")
except Exception as e:
    print(f"  {FAIL} DLinear forward pass: {e}")

print("\n[7] Mamba forward pass (CPU fallback)")
try:
    from src.models.mamba_ts import MambaForecaster, MAMBA_AVAILABLE
    model = MambaForecaster(seq_len=48, pred_len=1, enc_in=11,
                             d_model=32, d_state=8, d_conv=4,
                             expand=2, n_layers=2, patch_size=8, dropout=0.0)
    x = torch.randn(2, 48, 11)
    out = model(x)
    print(f"  {PASS} Mamba ({'real SSM' if MAMBA_AVAILABLE else 'fallback'}): "
          f"input={tuple(x.shape)} → output={tuple(out.shape)}")
except Exception as e:
    print(f"  {FAIL} Mamba forward pass: {e}")

print("\n" + "="*60)
print("Setup check complete. Fix any ❌ items before training.")
print("="*60 + "\n")
