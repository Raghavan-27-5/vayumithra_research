"""Debug: run task 1 only with detailed tracing."""
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
    return df.sort_values(['ZONEID','TIMESTAMP']).reset_index(drop=True)

def load_expvars(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'TaskExpVars{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    df = pd.concat(frames, ignore_index=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    return df.sort_values(['ZONEID','TIMESTAMP']).reset_index(drop=True)

class Cfg:
    seq_len=SEQ_LEN; pred_len=1; enc_in=50; d_model=128; n_heads=4; e_layers=2
    d_ff=512; dropout=0.15; activation='gelu'; embed='timeF'; freq='h'; factor=1
    use_norm=True; output_attention=False; quantiles=GEFCOM_QUANTILES
    target_channels=[0,5,10,15,20,25,30,35,40,45]; n_targets=10; n_zones=10

model = iTransformerNHiTS_GEFCom(Cfg()).to(device)
ckpt = torch.load(r'C:\Projects\raghavan\vayumithra_research\results\models\gefecom_v2\gefecom_model_v3.pt', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

task_num = 1
print(f'Loading Task {task_num}...', flush=True)
t1 = load_task(task_num)
t2 = load_task(task_num + 1)
ev1 = load_expvars(task_num)
bench = pd.read_csv(os.path.join(GEFCOM_DIR, f'Task {task_num}', f'benchmark{task_num}_W.csv'))
bench['TIMESTAMP'] = pd.to_datetime(bench['TIMESTAMP'], format='%Y%m%d %H:%M')

print(f'Task {task_num} rows: {len(t1)}', flush=True)
print(f'Task {task_num+1} rows: {len(t2)}', flush=True)
print(f'ExpVars {task_num} rows: {len(ev1)}', flush=True)
print(f'Benchmark {task_num} rows: {len(bench)}', flush=True)

# Build mdf
mdf_list = []
for z in range(1, N_ZONES + 1):
    zdf = t1[t1['ZONEID'] == z].set_index('TIMESTAMP')[['TARGETVAR','U10','V10','U100','V100']]
    mdf_list.append(zdf)
mdf = pd.concat(mdf_list, axis=1, keys=[f'Z{z}' for z in range(1, N_ZONES + 1)])
mdf = mdf.sort_index()
print(f'mdf: {mdf.shape}, NaN: {mdf.isna().sum().sum()}', flush=True)

# Build wdf
wdf_list = []
for z in range(1, N_ZONES + 1):
    zdf = ev1[ev1['ZONEID'] == z].set_index('TIMESTAMP')[['U10','V10','U100','V100']]
    wdf_list.append(zdf)
wdf = pd.concat(wdf_list, axis=1, keys=[f'Z{z}' for z in range(1, N_ZONES + 1)])
wdf = wdf.sort_index()
print(f'wdf: {wdf.shape}', flush=True)

# Ground truth merge
gt_merged = bench[['ZONEID','TIMESTAMP']].merge(
    t2[['ZONEID','TIMESTAMP','TARGETVAR']], on=['ZONEID','TIMESTAMP'], how='inner'
)
print(f'gt_merged: {len(gt_merged)} rows', flush=True)
print(f'gt_merged NaN in TARGETVAR: {gt_merged["TARGETVAR"].isna().sum()}', flush=True)

# Test timestamps
test_ts = np.sort(bench['TIMESTAMP'].unique())
print(f'Test timestamps: {len(test_ts)} unique, first={test_ts[0]}, last={test_ts[-1]}', flush=True)

# Batch loop
batch_size = 64
all_preds = {}
n_valid = 0
n_skipped = 0

for i in range(0, min(130, len(test_ts)), batch_size):
    batch_ts = test_ts[i:i+batch_size]
    bi, bw, bv = [], [], []
    for ts in batch_ts:
        past = mdf[mdf.index < ts]
        if len(past) < SEQ_LEN:
            n_skipped += 1; bv.append(False); continue
        inp = past.iloc[-SEQ_LEN:].values.astype(np.float32)
        if np.isnan(inp).any():
            n_skipped += 1; bv.append(False); continue
        wrow = wdf[wdf.index == ts]
        if len(wrow) == 0:
            n_skipped += 1; bv.append(False); continue
        wvals = wrow.values[0].astype(np.float32)
        bi.append(inp); bw.append(wvals); bv.append(True)

    if not any(bv):
        continue

    vi = [j for j, v in enumerate(bv) if v]
    xa = np.stack([bi[j] for j in vi], axis=0)
    wa = np.stack([bw[j] for j in vi], axis=0)
    n_valid += len(vi)

    xt = torch.from_numpy(xa).to(device)
    wt = torch.from_numpy(wa).to(device)
    with torch.no_grad():
        out = model(x_enc=xt, x_future_weather=wt)
    out_np = out.cpu().numpy()
    out_np = np.nan_to_num(out_np, nan=0.5, posinf=1.0, neginf=0.0)
    out_np = np.clip(out_np, 0.0, 1.0)

    for bi2, vi2 in enumerate(vi):
        ts_key = str(batch_ts[vi2])
        all_preds[ts_key] = out_np[bi2, :, 0, :]

print(f'Processed: {n_valid} valid, {n_skipped} skipped, {len(all_preds)} predictions stored', flush=True)

# Compute zone pinballs
target_channels = [0,5,10,15,20,25,30,35,40,45]
def pinball_score(y, yh, q):
    e = y - yh
    return np.mean(np.maximum(q * e, (q - 1) * e))

zone_pinballs = []
for zone_id in range(1, N_ZONES + 1):
    zgt = gt_merged[gt_merged['ZONEID'] == zone_id].sort_values('TIMESTAMP')
    preds, actuals = [], []
    for _, row in zgt.iterrows():
        ts_key = str(row['TIMESTAMP'])
        actual = row['TARGETVAR']
        if np.isnan(actual) or ts_key not in all_preds:
            continue
        preds.append(all_preds[ts_key][zone_id - 1])
        actuals.append(actual)
    if len(preds) == 0:
        print(f'  Zone {zone_id}: 0 predictions', flush=True)
        continue
    preds = np.array(preds); actuals = np.array(actuals)
    pb = np.mean([pinball_score(actuals, preds[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)])
    zone_pinballs.append(pb)
    print(f'  Zone {zone_id}: {len(preds)} samples, pinball={pb:.5f}', flush=True)

avg_pb = np.mean(zone_pinballs) if zone_pinballs else float('nan')
print(f'\nTask {task_num} pinball: {avg_pb:.5f}', flush=True)
