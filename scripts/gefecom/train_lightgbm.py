"""LightGBM for GEFCom2014 Wind track following kPower winner's approach.

Key features from the kPower paper (Landry et al. 2016):
1. Gradient Boosted Machines with quantile loss
2. Independent models per zone x quantile (990 models)
3. Raw + derived weather features (wind speed, energy, direction)
4. Bidirectional lagged 100m wind energy (smoothing)
5. Lagged TARGETVAR power values
6. Time features (hour, month, day of week)
7. Two-layer: cross-zone predictions (optional)
"""

import sys, os, zipfile, io, time, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed, dump, load

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
N_ZONES = 10
QUANTILES = np.arange(0.01, 1.0, 0.01)  # 99 quantiles

# Feature definitions
LAG_TARGET = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
              48, 72, 168, 336]
LAG_WEATHER = [1, 2, 3, 6, 12, 24]
LEAD_WEATHER = [1, 2, 3, 6, 12]

FEATURE_COLS = (
    ['U10', 'V10', 'U100', 'V100',
     'WS10', 'WS100', 'WE10', 'WE100',
     'WD10_sin', 'WD10_cos', 'WD100_sin', 'WD100_cos',
     'SHEAR', 'SHEAR_DIR',
     'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
     'WE100_roll_6h', 'TARGETVAR_roll_6h', 'TARGETVAR_roll_12h', 'TARGETVAR_roll_24h']
    + [f'TARGETVAR_lag_{l}' for l in LAG_TARGET]
    + [f'WE100_lag_{l}' for l in LAG_WEATHER]
    + [f'WE100_lead_{l}' for l in LEAD_WEATHER]
)

N_FEATURES = len(FEATURE_COLS)
print(f'Feature dimension: {N_FEATURES}', flush=True)


def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'Task{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    df = pd.concat(frames, ignore_index=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    df = df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
    df = df.groupby('ZONEID', group_keys=False).apply(lambda g: g.ffill())
    return df


def load_expvars(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'TaskExpVars{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    df = pd.concat(frames, ignore_index=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    df = df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
    return df


def add_derived_features(df):
    """Add derived weather and time features in-place."""
    df['WS10'] = np.sqrt(df['U10'] ** 2 + df['V10'] ** 2)
    df['WS100'] = np.sqrt(df['U100'] ** 2 + df['V100'] ** 2)
    df['WE10'] = 0.5 * (df['U10'] ** 2 + df['V10'] ** 2)
    df['WE100'] = 0.5 * (df['U100'] ** 2 + df['V100'] ** 2)
    wd10 = np.arctan2(df['V10'], df['U10'])
    wd100 = np.arctan2(df['V100'], df['U100'])
    df['WD10_sin'] = np.sin(wd10); df['WD10_cos'] = np.cos(wd10)
    df['WD100_sin'] = np.sin(wd100); df['WD100_cos'] = np.cos(wd100)
    df['SHEAR'] = df['WS100'] - df['WS10']
    df['SHEAR_DIR'] = wd100 - wd10
    ts = df['TIMESTAMP']
    hour = ts.dt.hour; month = ts.dt.month; dow = ts.dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)


def build_training_features(zone_df):
    """Build training features+target for one zone.
    zone_df: sorted by TIMESTAMP, columns include TARGETVAR, U10, V10, U100, V100
    Returns (X, y).
    """
    df = zone_df.copy()
    add_derived_features(df)

    # Rolling means
    df['WE100_roll_6h'] = df['WE100'].rolling(6, min_periods=1).mean()
    df['TARGETVAR_roll_6h'] = df['TARGETVAR'].rolling(6, min_periods=1).mean()
    df['TARGETVAR_roll_12h'] = df['TARGETVAR'].rolling(12, min_periods=1).mean()
    df['TARGETVAR_roll_24h'] = df['TARGETVAR'].rolling(24, min_periods=1).mean()

    # Lags (shift forward to avoid future leakage)
    for lag in LAG_TARGET:
        df[f'TARGETVAR_lag_{lag}'] = df['TARGETVAR'].shift(lag)
    for lag in LAG_WEATHER:
        df[f'WE100_lag_{lag}'] = df['WE100'].shift(lag)
    for lag in LEAD_WEATHER:
        df[f'WE100_lead_{lag}'] = df['WE100'].shift(-lag)

    # Drop rows with NaN features (first 336 rows will have NaN lags)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df[FEATURE_COLS].values, df['TARGETVAR'].values


def build_test_features(train_zone_df, expvars_zone_df):
    """Build test features for one zone.
    
    Since all test predictions are submitted at once (no sequential feedback),
    TARGETVAR lags are computed from the training boundary: ALL test hours
    share the same lag vector (last N training TARGETVAR values).
    """
    tv_train = train_zone_df['TARGETVAR'].values
    n_train = len(tv_train)
    n_test = len(expvars_zone_df)
    X_test = np.zeros((n_test, N_FEATURES), dtype=np.float32)

    # Pre-compute TARGETVAR lags from training boundary (same for all test hours)
    tv_lags = {}
    tv_mean = float(np.mean(tv_train))
    for lag in LAG_TARGET:
        src = n_train - lag
        tv_lags[lag] = tv_train[src] if src >= 0 else tv_mean

    tv_roll_6 = np.mean(tv_train[-6:]) if n_train >= 6 else tv_mean
    tv_roll_12 = np.mean(tv_train[-12:]) if n_train >= 12 else tv_mean
    tv_roll_24 = np.mean(tv_train[-24:]) if n_train >= 24 else tv_mean

    # Combined weather for WE100 (used for weather lags/leads)
    all_u100 = np.concatenate([train_zone_df['U100'].values, expvars_zone_df['U100'].values])
    all_v100 = np.concatenate([train_zone_df['V100'].values, expvars_zone_df['V100'].values])
    all_we100 = 0.5 * (all_u100 ** 2 + all_v100 ** 2)

    # Pre-compute WE100 lags from boundary (same for all test hours)
    we_lags = {}
    for lag in LAG_WEATHER:
        src = n_train - lag
        we_lags[lag] = all_we100[src] if src >= 0 else 0.0

    for i in range(n_test):
        row = expvars_zone_df.iloc[i]
        u10 = row['U10']; v10 = row['V10']; u100 = row['U100']; v100 = row['V100']
        ws10 = np.sqrt(u10 ** 2 + v10 ** 2)
        ws100 = np.sqrt(u100 ** 2 + v100 ** 2)
        we10 = 0.5 * (u10 ** 2 + v10 ** 2)
        we100 = 0.5 * (u100 ** 2 + v100 ** 2)
        wd10 = np.arctan2(v10, u10)
        wd100 = np.arctan2(v100, u100)

        X_test[i, 0] = u10;     X_test[i, 1] = v10
        X_test[i, 2] = u100;    X_test[i, 3] = v100
        X_test[i, 4] = ws10;    X_test[i, 5] = ws100
        X_test[i, 6] = we10;    X_test[i, 7] = we100
        X_test[i, 8] = np.sin(wd10);  X_test[i, 9] = np.cos(wd10)
        X_test[i, 10] = np.sin(wd100); X_test[i, 11] = np.cos(wd100)
        X_test[i, 12] = ws100 - ws10; X_test[i, 13] = wd100 - wd10

        ts = row['TIMESTAMP']
        hour = ts.hour; month = ts.month; dow = ts.dayofweek
        X_test[i, 14] = np.sin(2 * np.pi * hour / 24)
        X_test[i, 15] = np.cos(2 * np.pi * hour / 24)
        X_test[i, 16] = np.sin(2 * np.pi * month / 12)
        X_test[i, 17] = np.cos(2 * np.pi * month / 12)
        X_test[i, 18] = np.sin(2 * np.pi * dow / 7)
        X_test[i, 19] = np.cos(2 * np.pi * dow / 7)

        # WE100 rolling mean (uses only available past data)
        X_test[i, 20] = np.nanmean(all_we100[max(0, n_train + i - 5):n_train + i + 1])
        # TARGETVAR rolling means (from training boundary)
        X_test[i, 21] = tv_roll_6
        X_test[i, 22] = tv_roll_12
        X_test[i, 23] = tv_roll_24

        # TARGETVAR lags (same for all test hours)
        col = 24
        for lag in LAG_TARGET:
            X_test[i, col] = tv_lags[lag]
            col += 1

        # WE100 lags (same for all test hours)
        for lag in LAG_WEATHER:
            X_test[i, col] = we_lags[lag]
            col += 1

        # WE100 leads (different per test hour — uses future ExpVars)
        for lag in LEAD_WEATHER:
            src = n_train + i + lag
            if src < n_train + n_test:
                X_test[i, col] = all_we100[src] if not np.isnan(all_we100[src]) else 0.0
            col += 1

    return X_test


def train_zone_models(zone_id, task_data, task_num, model_dir, n_jobs=4):
    """Train 99 quantile models for one zone. Returns list of models."""
    zdf = task_data[task_data['ZONEID'] == zone_id].sort_values('TIMESTAMP').reset_index(drop=True)
    X, y = build_training_features(zdf)
    print(f'  Zone {zone_id}: {len(X)} training samples, {N_FEATURES} features', flush=True)

    def _train_q(qi):
        q = QUANTILES[qi]
        model = lgb.LGBMRegressor(
            objective='quantile', alpha=q,
            n_estimators=300, learning_rate=0.1,
            max_depth=5, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=30, reg_alpha=0.1, reg_lambda=0.1,
            verbose=-1, random_state=42 + qi,
            n_jobs=1
        )
        model.fit(X, y)
        return model

    models = Parallel(n_jobs=n_jobs)(
        delayed(_train_q)(qi) for qi in range(99)
    )
    dump(models, os.path.join(model_dir, f'task{task_num}_zone{zone_id}_models.joblib'))
    print(f'  Zone {zone_id}: 99 models trained and saved', flush=True)
    return models


def predict_zone(zone_id, train_df, expvars, models):
    """Predict all 99 quantiles for all test hours for one zone."""
    ztrain = train_df[train_df['ZONEID'] == zone_id].sort_values('TIMESTAMP').reset_index(drop=True)
    zexp = expvars[expvars['ZONEID'] == zone_id].sort_values('TIMESTAMP').reset_index(drop=True)
    X_test = build_test_features(ztrain, zexp)
    n_test = len(X_test)

    preds = np.zeros((n_test, 99), dtype=np.float32)
    for qi, model in enumerate(models):
        preds[:, qi] = model.predict(X_test)

    # Monotonicity correction: ensure q_i <= q_{i+1}
    preds = np.maximum.accumulate(preds, axis=1)
    # Clip to [0, 1]
    preds = np.clip(preds, 0.0, 1.0)
    return preds


def pinball_score(actual, preds, quantiles):
    """Average pinball across all quantiles."""
    scores = np.zeros(len(quantiles))
    for qi, q in enumerate(quantiles):
        e = actual - preds[:, qi]
        scores[qi] = np.mean(np.maximum(q * e, (q - 1) * e))
    return np.mean(scores)


def evaluate_task(task_num, model_dir, all_task_data, all_expvars):
    """Evaluate one task."""
    train_df = all_task_data[task_num]
    expvars = all_expvars[task_num]
    bench = pd.read_csv(os.path.join(GEFCOM_DIR, f'Task {task_num}', f'benchmark{task_num}_W.csv'))
    bench['TIMESTAMP'] = pd.to_datetime(bench['TIMESTAMP'], format='%Y%m%d %H:%M')

    # Ground truth from task_num + 1
    gt_df = all_task_data[task_num + 1]
    gt_merged = bench[['ZONEID', 'TIMESTAMP']].merge(
        gt_df[['ZONEID', 'TIMESTAMP', 'TARGETVAR']], on=['ZONEID', 'TIMESTAMP'], how='inner'
    )

    # Check if models exist
    models_exist = all(
        os.path.exists(os.path.join(model_dir, f'task{task_num}_zone{z}_models.joblib'))
        for z in range(1, N_ZONES + 1)
    )

    if models_exist:
        print(f'  Loading existing models for Task {task_num}...', flush=True)
        zone_models = {}
        for z in range(1, N_ZONES + 1):
            zone_models[z] = load(os.path.join(model_dir, f'task{task_num}_zone{z}_models.joblib'))
    else:
        print(f'  Training models for Task {task_num}...', flush=True)
        zone_models = {}
        for z in range(1, N_ZONES + 1):
            zone_models[z] = train_zone_models(z, train_df, task_num, model_dir)

    # Predict for each zone
    print(f'  Predicting for Task {task_num}...', flush=True)
    zone_pinballs = []
    for z in range(1, N_ZONES + 1):
        preds = predict_zone(z, train_df, expvars, zone_models[z])

        zgt = gt_merged[gt_merged['ZONEID'] == z].sort_values('TIMESTAMP')
        actuals = zgt['TARGETVAR'].values
        if len(actuals) != len(preds):
            print(f'  WARNING Zone {z}: {len(actuals)} actuals vs {len(preds)} predictions', flush=True)
            n = min(len(actuals), len(preds))
            actuals = actuals[:n]
            preds = preds[:n]

        pb = pinball_score(actuals, preds, QUANTILES)
        zone_pinballs.append(pb)

    avg_pb = np.mean(zone_pinballs)
    return avg_pb


def main():
    model_dir = r'C:\Projects\raghavan\vayumithra_research\results\models\lightgbm'
    os.makedirs(model_dir, exist_ok=True)

    # Pre-load ALL task data and ExpVars
    print('Loading all task data...', flush=True)
    all_task_data = {}
    for tn in range(1, 16):
        t1 = time.time()
        all_task_data[tn] = load_task(tn)
        df = all_task_data[tn]
        print(f'  Task {tn:>2}: {len(df):>6} rows  {df["TIMESTAMP"].min().strftime("%Y-%m-%d")} to {df["TIMESTAMP"].max().strftime("%Y-%m-%d")}  [{time.time() - t1:.0f}s]', flush=True)

    print('Loading ExpVars...', flush=True)
    all_expvars = {}
    for tn in range(1, 13):
        all_expvars[tn] = load_expvars(tn)
    print('Done loading data.', flush=True)

    # Evaluate tasks 1-12
    results = {}
    for task_num in range(1, 13):
        t0 = time.time()
        pb = evaluate_task(task_num, model_dir, all_task_data, all_expvars)
        elapsed = time.time() - t0
        results[task_num] = pb
        print(f'Task {task_num:>2}: {len(all_expvars[task_num]):>4} hrs  pinball={pb:.5f}  [{elapsed:.0f}s]', flush=True)

    print('\n' + '=' * 60)
    print('LightGBM - Performance for all 12 weeks')
    print('=' * 60)
    for wk in range(1, 13):
        print(f'Week {wk:>2}: {results[wk]:.5f}')
    avg = np.mean([results[wk] for wk in range(1, 13)])
    print(f'Average: {avg:.5f}')
    print(f'Total time: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
