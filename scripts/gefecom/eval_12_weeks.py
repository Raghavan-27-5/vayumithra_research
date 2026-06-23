"""
Evaluate our iTransformer+NHiTS model on all 12 GEFCom2014 Wind tasks.
Ground truth: Task N+1's data contains TARGETVAR for Task N's test month.
Uses ExpVars for future weather injection.
"""
from __future__ import annotations
import sys, os, zipfile, io, time, numpy as np, pandas as pd, torch
sys.path.insert(0, r'C:\Projects\raghavan\vayumithra_research')
sys.path.insert(0, r'C:\Projects\raghavan')
from src.models.gefecom_model import iTransformerNHiTS_GEFCom, GEFCOM_QUANTILES

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
SEQ_LEN, N_ZONES = 336, 10
device = torch.device('cpu')

def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'Task{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    df = pd.concat(frames, ignore_index=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    df = df.sort_values(['ZONEID','TIMESTAMP']).reset_index(drop=True)
    df = df.groupby('ZONEID', group_keys=False).apply(lambda g: g.ffill())
    return df

def load_expvars(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'TaskExpVars{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    df = pd.concat(frames, ignore_index=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    df = df.sort_values(['ZONEID','TIMESTAMP']).reset_index(drop=True)
    return df

# Pre-load all task data (Tasks 1-15) for ground truth access
print('Loading all task data...', flush=True)
all_task_data = {}
for tn in range(1, 16):
    all_task_data[tn] = load_task(tn)
    z1 = all_task_data[tn][all_task_data[tn]['ZONEID']==1]
    print(f'  Task {tn:2d}: {len(all_task_data[tn]):6d} rows  {z1.TIMESTAMP.min().date()} to {z1.TIMESTAMP.max().date()}', flush=True)

# Also pre-load ExpVars for all tasks (for weather injection)
print('Loading ExpVars...', flush=True)
all_expvars = {}
for tn in range(1, 16):
    all_expvars[tn] = load_expvars(tn)
print('Done loading data.', flush=True)

# Load model
class Cfg:
    seq_len=SEQ_LEN; pred_len=1; enc_in=50; d_model=128; n_heads=4; e_layers=2
    d_ff=512; dropout=0.15; activation='gelu'; embed='timeF'; freq='h'; factor=1
    use_norm=True; output_attention=False; quantiles=GEFCOM_QUANTILES
    target_channels=[0,5,10,15,20,25,30,35,40,45]; n_targets=10; n_zones=10

model = iTransformerNHiTS_GEFCom(Cfg()).to(device)
ckpt_path = r'C:\Projects\raghavan\vayumithra_research\results\models\gefecom_v2\gefecom_model_v3.pt'
if not os.path.exists(ckpt_path):
    ckpt_path = r'C:\Projects\raghavan\vayumithra_research\results\models\gefecom_v2\gefecom_model_v2.pt'
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f'Model loaded from {ckpt_path}', flush=True)

def build_mdf(task_df):
    mdf_list = []
    for z in range(1, N_ZONES + 1):
        zdf = task_df[task_df['ZONEID'] == z].set_index('TIMESTAMP')[['TARGETVAR','U10','V10','U100','V100']]
        mdf_list.append(zdf)
    mdf = pd.concat(mdf_list, axis=1, keys=[f'Z{z}' for z in range(1, N_ZONES + 1)])
    return mdf.sort_index()

def build_weather_df(ev_df):
    wdf_list = []
    for z in range(1, N_ZONES + 1):
        zdf = ev_df[ev_df['ZONEID'] == z].set_index('TIMESTAMP')[['U10','V10','U100','V100']]
        wdf_list.append(zdf)
    wdf = pd.concat(wdf_list, axis=1, keys=[f'Z{z}' for z in range(1, N_ZONES + 1)])
    return wdf.sort_index()

target_channels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

def pinball_score(y, yh, q):
    e = y - yh
    return np.mean(np.maximum(q * e, (q - 1) * e))

results = {}
overall_start = time.time()

for task_num in range(1, 13):
    t_start = time.time()

    # Load benchmark to get test timestamps
    bench = pd.read_csv(os.path.join(GEFCOM_DIR, f'Task {task_num}', f'benchmark{task_num}_W.csv'))
    bench['TIMESTAMP'] = pd.to_datetime(bench['TIMESTAMP'], format='%Y%m%d %H:%M')
    test_timestamps = np.sort(bench['TIMESTAMP'].unique())
    n_hrs = len(test_timestamps)

    # Build multivariate array from task N's data (past window source)
    mdf = build_mdf(all_task_data[task_num])

    # Build weather df from ExpVars N (future weather for test period)
    wdf = build_weather_df(all_expvars[task_num])

    # Ground truth: TARGETVAR from Task N+1's data at test timestamps
    gt_data = all_task_data[task_num + 1]
    gt_merged = bench[['ZONEID','TIMESTAMP']].merge(
        gt_data[['ZONEID','TIMESTAMP','TARGETVAR']], on=['ZONEID','TIMESTAMP'], how='inner'
    )

    # Predict at each test hour (batched)
    batch_size = 64
    all_preds = {}

    for i in range(0, len(test_timestamps), batch_size):
        batch_ts = test_timestamps[i:i+batch_size]

        batch_inputs, batch_weather, batch_valid = [], [], []
        for ts in batch_ts:
            past = mdf[mdf.index < ts]
            if len(past) < SEQ_LEN:
                batch_valid.append(False); continue
            inp = past.iloc[-SEQ_LEN:].values.astype(np.float32)
            if np.isnan(inp).any():
                batch_valid.append(False); continue
            wrow = wdf[wdf.index == ts]
            if len(wrow) == 0:
                batch_valid.append(False); continue
            wvals = wrow.values[0].astype(np.float32)
            batch_inputs.append(inp); batch_weather.append(wvals); batch_valid.append(True)

        if not any(batch_valid):
            continue

        valid_idx = [j for j, v in enumerate(batch_valid) if v]
        x_arr = np.stack([batch_inputs[j] for j in valid_idx], axis=0)
        w_arr = np.stack([batch_weather[j] for j in valid_idx], axis=0)

        x_t = torch.from_numpy(x_arr).to(device)
        w_t = torch.from_numpy(w_arr).to(device)

        with torch.no_grad():
            out = model(x_enc=x_t, x_future_weather=w_t)

        out_np = out.cpu().numpy()
        out_np = np.nan_to_num(out_np, nan=0.5, posinf=1.0, neginf=0.0)
        out_np = np.clip(out_np, 0.0, 1.0)

        for bi, vi in enumerate(valid_idx):
            ts_key = pd.Timestamp(batch_ts[vi])
            all_preds[ts_key] = out_np[bi, :, 0, :]

    # Compute per-zone pinball
    zone_pinballs = []
    for zone_id in range(1, N_ZONES + 1):
        zgt = gt_merged[gt_merged['ZONEID'] == zone_id].sort_values('TIMESTAMP')
        preds, actuals = [], []
        for _, row in zgt.iterrows():
            ts_key = str(row['TIMESTAMP'])
            actual = row['TARGETVAR']
            ts_key = row['TIMESTAMP']
            if np.isnan(actual) or ts_key not in all_preds:
                continue
            preds.append(all_preds[ts_key][zone_id - 1])
            actuals.append(actual)
        if len(preds) == 0:
            continue
        preds = np.array(preds); actuals = np.array(actuals)
        pb = np.mean([pinball_score(actuals, preds[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)])
        zone_pinballs.append(pb)

    avg_pb = np.mean(zone_pinballs) if zone_pinballs else float('nan')
    results[task_num] = avg_pb
    elapsed = time.time() - t_start
    print(f'Task {task_num:2d}: {n_hrs:3d} hrs  pinball={avg_pb:.5f}  [{elapsed:.0f}s]', flush=True)

# Print results in template
print('\n' + '='*60)
print('Our Model - Performance for all 12 weeks')
print('='*60)
total = 0.0
count = 0
for wk in range(1, 13):
    s = results.get(wk, float('nan'))
    if not np.isnan(s):
        total += s; count += 1
    print(f'Week {wk:2d}: {s:.5f}')
avg = total / count if count > 0 else float('nan')
print(f'Average: {avg:.5f}')
print(f'\nTotal time: {time.time() - overall_start:.0f}s', flush=True)
