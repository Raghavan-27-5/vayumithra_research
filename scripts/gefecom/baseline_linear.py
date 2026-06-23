"""Quick baseline: predict TARGETVAR from weather only (no time series)."""
import sys, os, zipfile, io, numpy as np, pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
N_ZONES = 10

def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'Task{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

def load_expvars(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'TaskExpVars{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

all_df = pd.concat([load_task(t) for t in range(1, 16)], ignore_index=True)
all_df['TIMESTAMP'] = pd.to_datetime(all_df['TIMESTAMP'], format='%Y%m%d %H:%M')
all_df = all_df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
all_df = all_df.drop_duplicates(subset=['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
all_df = all_df.groupby('ZONEID', group_keys=False).apply(lambda g: g.ffill())

ev = load_expvars(15)
ev['TIMESTAMP'] = pd.to_datetime(ev['TIMESTAMP'], format='%Y%m%d %H:%M')
ev = ev.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)

print("Per-zone Linear: weather -> TARGETVAR (evaluated on Dec 2013 with ExpVars)")
print()

for zone_id in range(1, N_ZONES + 1):
    zdf = all_df[all_df['ZONEID'] == zone_id].copy()
    train_df = zdf[zdf['TIMESTAMP'] < '2013-12-01']
    test_df = zdf[(zdf['TIMESTAMP'] >= '2013-12-01') & (zdf['TIMESTAMP'] < '2014-01-01')]
    ev_z = ev[ev['ZONEID'] == zone_id].sort_values('TIMESTAMP')

    Xtr = train_df[['U10','V10','U100','V100']].values.astype(np.float32)
    ytr = train_df['TARGETVAR'].values.astype(np.float32)
    valid = ~np.isnan(ytr)
    Xtr, ytr = Xtr[valid], ytr[valid]

    # Merge test and ev on timestamp
    merged = test_df.merge(ev_z[['TIMESTAMP','U10','V10','U100','V100']], on='TIMESTAMP', suffixes=('_actual','_forecast'))
    Xte = merged[['U10_forecast','V10_forecast','U100_forecast','V100_forecast']].values.astype(np.float32)
    yte = merged['TARGETVAR'].values.astype(np.float32)

    reg = Ridge(alpha=1.0).fit(Xtr, ytr)
    yp = reg.predict(Xte)
    yp = np.clip(yp, 0.0, 1.0)
    mae = mean_absolute_error(yte, yp)
    r2 = r2_score(yte, yp)
    print(f"  Z{zone_id:2d}: n={len(yte):4d}  MAE={mae:.4f}  R2={r2:.4f}")

# Also try: all zones together with zone as feature
print("\nAll-zone model with zone one-hot encoding:")
zones_list = []
for zone_id in range(1, N_ZONES + 1):
    zdf = all_df[all_df['ZONEID'] == zone_id].copy()
    train_df = zdf[zdf['TIMESTAMP'] < '2013-12-01']
    test_df = zdf[(zdf['TIMESTAMP'] >= '2013-12-01') & (zdf['TIMESTAMP'] < '2014-01-01')]
    ev_z = ev[ev['ZONEID'] == zone_id].sort_values('TIMESTAMP')

    train_df['zone_id'] = zone_id
    merged = test_df.merge(ev_z[['TIMESTAMP','U10','V10','U100','V100']], on='TIMESTAMP', suffixes=('_actual','_forecast'))
    merged['zone_id'] = zone_id
    zones_list.append((train_df, merged))

# Build combined dataset
all_train = pd.concat([z[0] for z in zones_list], ignore_index=True)
all_test = pd.concat([z[1] for z in zones_list], ignore_index=True)

# Features: 4 weather vars + zone one-hot
zone_dummies_train = pd.get_dummies(all_train['zone_id'], prefix='Z')
zone_dummies_test = pd.get_dummies(all_test['zone_id'], prefix='Z')
cols_train = all_train[['U10','V10','U100','V100']].columns.tolist()
Xtr_all = np.column_stack([all_train[cols_train].values.astype(np.float32), zone_dummies_train.values.astype(np.float32)])
ytr_all = all_train['TARGETVAR'].values.astype(np.float32)
valid = ~np.isnan(ytr_all)
Xtr_all, ytr_all = Xtr_all[valid], ytr_all[valid]

cols_test = ['U10_forecast','V10_forecast','U100_forecast','V100_forecast']
Xte_all = np.column_stack([all_test[cols_test].values.astype(np.float32), zone_dummies_test.values.astype(np.float32)])
yte_all = all_test['TARGETVAR'].values.astype(np.float32)
valid = ~np.isnan(yte_all)
# Check all test zones are present
for z in range(1, 11):
    if f'Z{z}' not in zone_dummies_test.columns:
        zone_dummies_test[f'Z{z}'] = 0

# Recompute with all zones
zone_dummies_test = pd.get_dummies(all_test['zone_id'], prefix='Z')
# Ensure all 10 zones present
for z in range(1, 11):
    if f'Z{z}' not in zone_dummies_test.columns:
        zone_dummies_test[f'Z{z}'] = 0
zone_dummies_test = zone_dummies_test[[f'Z{z}' for z in range(1, 11)]]

zone_dummies_train = pd.get_dummies(all_train['zone_id'], prefix='Z')
for z in range(1, 11):
    if f'Z{z}' not in zone_dummies_train.columns:
        zone_dummies_train[f'Z{z}'] = 0
zone_dummies_train = zone_dummies_train[[f'Z{z}' for z in range(1, 11)]]

Xtr_all = np.column_stack([all_train[cols_train].values.astype(np.float32), zone_dummies_train.values.astype(np.float32)])
Xte_all = np.column_stack([all_test[cols_test].values.astype(np.float32), zone_dummies_test.values.astype(np.float32)])

Xtr_all, ytr_all = Xtr_all[valid], ytr_all[valid]
Xte_all, yte_all = Xte_all[valid], yte_all[valid]

reg_all = Ridge(alpha=1.0).fit(Xtr_all, ytr_all)
yp_all = np.clip(reg_all.predict(Xte_all), 0.0, 1.0)
print(f"  All zones: n={len(yte_all):4d}  MAE={mean_absolute_error(yte_all, yp_all):.4f}  R2={r2_score(yte_all, yp_all):.4f}")
