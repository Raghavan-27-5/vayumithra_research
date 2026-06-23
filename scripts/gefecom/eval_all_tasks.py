"""
Evaluate iTransformer+NHiTS GEFCom model on all 12 GEFCom2014 Wind tasks.
Computes average pinball per week (task) matching the leaderboard format.
"""
from __future__ import annotations
import sys, os, zipfile, io, time, numpy as np, pandas as pd, torch

sys.path.insert(0, r'C:\Projects\raghavan\vayumithra_research')
from src.models.gefecom_model import iTransformerNHiTS_GEFCom, GEFCOM_QUANTILES
from src.models.probabilistic_model import PinballLoss

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
SEQ_LEN, N_ZONES = 336, 10

device = torch.device('cpu')

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

# Load model
class Cfg:
    seq_len=SEQ_LEN; pred_len=1; enc_in=50; d_model=128; n_heads=4; e_layers=2
    d_ff=512; dropout=0.15; activation='gelu'; embed='timeF'; freq='h'; factor=1
    use_norm=True; output_attention=False; quantiles=GEFCOM_QUANTILES
    target_channels=[0,5,10,15,20,25,30,35,40,45]; n_targets=10; n_zones=10

model = iTransformerNHiTS_GEFCom(Cfg()).to(device)
ckpt_path = r'C:\Projects\raghavan\vayumithra_research\results\models\gefecom_v2\gefecom_model_v3.pt'
if not os.path.exists(ckpt_path):
    # Try v2 path
    ckpt_path = r'C:\Projects\raghavan\vayumithra_research\results\models\gefecom_v2\gefecom_model_v2.pt'
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f'Model loaded from {ckpt_path}', flush=True)

def pinball_score(y, yh, q):
    e = y - yh
    return np.mean(np.maximum(q * e, (q - 1) * e))

def build_mdf(task_df):
    mdf_list = []
    for z in range(1, N_ZONES + 1):
        zdf = task_df[task_df['ZONEID'] == z].set_index('TIMESTAMP')[['TARGETVAR','U10','V10','U100','V100']]
        mdf_list.append(zdf)
    mdf = pd.concat(mdf_list, axis=1, keys=[f'Z{z}' for z in range(1, N_ZONES + 1)])
    return mdf.sort_index()

def build_wdf(ev_df):
    wdf_list = []
    for z in range(1, N_ZONES + 1):
        zdf = ev_df[ev_df['ZONEID'] == z].set_index('TIMESTAMP')[['U10','V10','U100','V100']]
        wdf_list.append(zdf)
    wdf = pd.concat(wdf_list, axis=1, keys=[f'Z{z}' for z in range(1, N_ZONES + 1)])
    return wdf.sort_index()

target_channels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
results = {}

for task_num in range(1, 13):
    t_start = time.time()
    print(f'\nTask {task_num}...', flush=True, end=' ')

    # Load data
    task_df = load_task(task_num)
    task_df['TIMESTAMP'] = pd.to_datetime(task_df['TIMESTAMP'], format='%Y%m%d %H:%M')
    task_df = task_df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
    task_df = task_df.groupby('ZONEID', group_keys=False).apply(lambda g: g.ffill())

    ev_df = load_expvars(task_num)
    ev_df['TIMESTAMP'] = pd.to_datetime(ev_df['TIMESTAMP'], format='%Y%m%d %H:%M')
    ev_df = ev_df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)

    # Load benchmark to get test timestamps
    bench = pd.read_csv(os.path.join(GEFCOM_DIR, f'Task {task_num}', f'benchmark{task_num}_W.csv'))
    bench['TIMESTAMP'] = pd.to_datetime(bench['TIMESTAMP'], format='%Y%m%d %H:%M')

    # Build full multivariate (all data)
    mdf = build_mdf(task_df)
    wdf = build_wdf(ev_df)
    full_data = mdf.values.astype(np.float32)

    # Merge benchmark timestamps with task data to get ground truth
    test_df = bench[['ZONEID', 'TIMESTAMP']].merge(
        task_df[['ZONEID', 'TIMESTAMP', 'TARGETVAR']], on=['ZONEID', 'TIMESTAMP']
    )
    test_df = test_df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)

    # Get unique test timestamps
    test_timestamps = test_df['TIMESTAMP'].unique()
    print(f'{len(test_timestamps)} hours, {len(test_df)} samples', flush=True, end=' ')

    # Evaluate: process by hour (batched) to get all 10 zones at once
    batch_size = 32
    all_preds = {}
    all_actuals = {}

    for i in range(0, len(test_timestamps), batch_size):
        batch_ts = test_timestamps[i:i+batch_size]
        batch_inputs = []
        batch_weather = []
        batch_valid = []

        for ts in batch_ts:
            past = mdf[mdf.index < ts]
            if len(past) < SEQ_LEN:
                batch_valid.append(False)
                continue
            inp = past.iloc[-SEQ_LEN:].values.astype(np.float32)
            if np.isnan(inp).any():
                batch_valid.append(False)
                continue

            wrow = wdf[wdf.index == ts]
            if len(wrow) == 0:
                batch_valid.append(False)
                continue
            wvals = wrow.values[0].astype(np.float32)

            batch_inputs.append(inp)
            batch_weather.append(wvals)
            batch_valid.append(True)

        if not any(batch_valid):
            continue

        B = sum(batch_valid)
        x_arr = np.stack([batch_inputs[j] for j, v in enumerate(batch_valid) if v], axis=0)
        w_arr = np.stack([batch_weather[j] for j, v in enumerate(batch_valid) if v], axis=0)

        x_t = torch.from_numpy(x_arr).to(device)
        w_t = torch.from_numpy(w_arr).to(device)

        with torch.no_grad():
            out = model(x_enc=x_t, x_future_weather=w_t)  # (B, 10, 1, 99)

        out_np = out.cpu().numpy()  # (B, 10, 1, 99)
        out_np = np.nan_to_num(out_np, nan=0.5, posinf=1.0, neginf=0.0)
        out_np = np.clip(out_np, 0.0, 1.0)

        # Store per-zone predictions
        valid_indices = [j for j, v in enumerate(batch_valid) if v]
        for bi, vi in enumerate(valid_indices):
            ts_key = str(batch_ts[vi])
            all_preds[ts_key] = out_np[bi, :, 0, :]  # (10, 99)

    # Now compute pinball per zone
    zone_pinballs = []
    for zone_id in range(1, N_ZONES + 1):
        zdf = test_df[test_df['ZONEID'] == zone_id].sort_values('TIMESTAMP')
        z_preds, z_actuals = [], []
        for _, row in zdf.iterrows():
            ts_key = str(row['TIMESTAMP'])
            actual = row['TARGETVAR']
            if np.isnan(actual):
                continue
            if ts_key not in all_preds:
                continue
            z_preds.append(all_preds[ts_key][zone_id - 1])
            z_actuals.append(actual)

        if len(z_preds) == 0:
            continue
        z_preds = np.array(z_preds)
        z_actuals = np.array(z_actuals)

        # Pinball per quantile
        pb = np.mean([pinball_score(z_actuals, z_preds[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)])
        zone_pinballs.append(pb)

    if len(zone_pinballs) == 0:
        avg_pinball = float('nan')
    else:
        avg_pinball = np.mean(zone_pinballs)

    results[task_num] = avg_pinball
    elapsed = time.time() - t_start
    print(f'pinball={avg_pinball:.5f} [{elapsed:.0f}s]', flush=True)

# Print results in template
print('\n' + '='*60)
print('Our Model - Performance for all 12 weeks')
print('='*60)
total = 0.0
count = 0
for wk in range(1, 13):
    s = results.get(wk, float('nan'))
    if not np.isnan(s):
        total += s
        count += 1
    print(f'Week {wk:2d}: {s:.5f}')
print(f'Average: {total/count:.5f}' if count > 0 else 'Average: NaN')

# Save results
np.savez(r'C:\Users\Nandha\AppData\Local\Temp\opencode\our_12_week_results.npz', **{f'wk{k}': v for k, v in results.items()})
print(f'\nResults saved.', flush=True)
