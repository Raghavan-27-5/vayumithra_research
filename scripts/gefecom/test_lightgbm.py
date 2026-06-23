"""Test: LightGBM for Task 1 only."""
import sys, os, time, numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, r'C:\Users\Nandha\AppData\Local\Temp\opencode')
from train_lightgbm import *

# Load data
print('Loading data...', flush=True)
train_df = load_task(1)
print(f'Train: {len(train_df)} rows', flush=True)

# Test zone 1 only
z = 1
zdf = train_df[train_df['ZONEID'] == z].sort_values('TIMESTAMP').reset_index(drop=True)
print(f'Zone {z}: {len(zdf)} rows, TARGETVAR range [{zdf["TARGETVAR"].min():.4f}, {zdf["TARGETVAR"].max():.4f}]', flush=True)

t0 = time.time()
X, y = build_training_features(zdf)
print(f'Features: {X.shape[0]} samples, {X.shape[1]} features  [{time.time()-t0:.1f}s]', flush=True)
print(f'y range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}', flush=True)
print(f'NaN in X: {np.isnan(X).sum()}, NaN in y: {np.isnan(y).sum()}', flush=True)

# Train quantile 0.5
t1 = time.time()
model = lgb.LGBMRegressor(
    objective='quantile', alpha=0.5,
    n_estimators=200, learning_rate=0.1,
    max_depth=6, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    verbose=1, random_state=42
)
model.fit(X, y)
print(f'Model trained [{time.time()-t1:.1f}s]', flush=True)

# Test prediction
preds = model.predict(X[:100])
e = y[:100] - preds
pb = np.mean(np.maximum(0.5 * e, (0.5 - 1) * e))
print(f'Pinball q=0.5: {pb:.5f}', flush=True)
print(f'Preds: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}', flush=True)

# Now test full evaluation
print('\nFull evaluation for Task 1...', flush=True)
expvars = load_expvars(1)
bench = pd.read_csv(os.path.join(GEFCOM_DIR, 'Task 1', 'benchmark1_W.csv'))
bench['TIMESTAMP'] = pd.to_datetime(bench['TIMESTAMP'], format='%Y%m%d %H:%M')
gt_df = load_task(2)
gt_merged = bench[['ZONEID','TIMESTAMP']].merge(
    gt_df[['ZONEID','TIMESTAMP','TARGETVAR']], on=['ZONEID','TIMESTAMP'], how='inner'
)

# Test features
zexp = expvars[expvars['ZONEID'] == z].sort_values('TIMESTAMP').reset_index(drop=True)
X_test = build_test_features(zdf, zexp)
print(f'Test features: {X_test.shape}', flush=True)
print(f'NaN in X_test: {np.isnan(X_test).sum()} / {X_test.size}', flush=True)

preds = model.predict(X_test)
print(f'Test preds: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}', flush=True)

# Pinball
zgt = gt_merged[gt_merged['ZONEID'] == z].sort_values('TIMESTAMP')
actuals = zgt['TARGETVAR'].values
e = actuals - preds
pb = np.mean(np.maximum(0.5 * e, (0.5 - 1) * e))
print(f'Test pinball (q=0.5 only): {pb:.5f}', flush=True)
