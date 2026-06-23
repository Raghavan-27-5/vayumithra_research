"""
Evaluate GEFCom2014 Task 15 using trained probabilistic model.
"""
import sys, os, json, zipfile, io, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
from src.models.probabilistic_model import GEFCOM_QUANTILES, iTransformerNHiTS_Probabilistic, PinballLoss
from sklearn.metrics import mean_absolute_error

GEFCOM_DIR = r"C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind"
SEQ_LEN, N_ZONES, FEATS = 336, 10, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ── Load all pre-2013 data (same as training) ────────────────────────────
def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f"Task {task_num}")
    zf = zipfile.ZipFile(os.path.join(zd, f"Task{task_num}_W_Zone1_10.zip"))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

def build_mdf(df, max_ts=None):
    sub = df[df["TIMESTAMP"] < max_ts].copy() if max_ts else df.copy()
    parts = []
    for z in range(1, N_ZONES + 1):
        zdf = sub[sub["ZONEID"] == z].set_index("TIMESTAMP")[["TARGETVAR","U10","V10","U100","V100"]]
        parts.append(zdf)
    mdf = pd.concat(parts, axis=1, keys=[f"Z{z}" for z in range(1, N_ZONES + 1)])
    return mdf.sort_index()

all_df = pd.concat([load_task(t) for t in range(1, 16)], ignore_index=True)
all_df["TIMESTAMP"] = pd.to_datetime(all_df["TIMESTAMP"], format="%Y%m%d %H:%M")

# Build model
class Cfg:
    seq_len=SEQ_LEN; pred_len=1; enc_in=N_ZONES*FEATS
    d_model=128; n_heads=4; e_layers=2; d_ff=512; dropout=0.15
    activation="gelu"; embed="timeF"; freq="h"; factor=1
    class_strategy="projection"; use_norm=True; output_attention=False
    ws_channel=-1; quantiles=GEFCOM_QUANTILES
    n_targets=N_ZONES
    target_channels=list(range(0, N_ZONES*FEATS, FEATS))  # [0,5,10,15,20,25,30,35,40,45]

model = iTransformerNHiTS_Probabilistic(Cfg()).to(device)
ckpt = torch.load(r"C:\Projects\raghavan\results\models\gefecom\gefecom_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)

# ── Evaluate on Task 15 ──────────────────────────────────────────────────
sol = pd.read_csv(os.path.join(GEFCOM_DIR, "Solution to Task 15", "solution15_W.csv"))
sol["TIMESTAMP"] = pd.to_datetime(sol["TIMESTAMP"], format="%Y%m%d %H:%M")

bench = pd.read_csv(os.path.join(GEFCOM_DIR, "Task 15", "benchmark15_W.csv"))
bench["TIMESTAMP"] = pd.to_datetime(bench["TIMESTAMP"], format="%Y%m%d %H:%M")
bench_q_cols = [f"{q:.2f}".rstrip('0') if q >= 0.1 else f"{q:.2f}" for q in GEFCOM_QUANTILES]
bench_q_cols = [c.rstrip('.') for c in bench_q_cols]  # handle '0.1' vs '0.10'

# Build full multivariate data up to Jan 2014 (includes all training + Dec 2013 data)
mdf = build_mdf(all_df, max_ts="2014-01-01")
print(f"Multi-variate data: {mdf.shape}, range: {mdf.index.min()} to {mdf.index.max()}", flush=True)
target_channels = Cfg.target_channels

def pinball_score(y, yh, q):
    e = y - yh; return np.mean(np.maximum(q*e, (q-1)*e))

all_model_preds, all_bench_preds, all_actuals = [], [], []

for zone_id in range(1, N_ZONES + 1):
    tgt_ch = target_channels[zone_id - 1]
    z_sol = sol[sol["ZONEID"] == zone_id].sort_values("TIMESTAMP")
    z_bench = bench[bench["ZONEID"] == zone_id].sort_values("TIMESTAMP")
    
    zone_preds, zone_actuals, zone_bench_q = [], [], []
    
    for idx, row in z_sol.iterrows():
        ts = row["TIMESTAMP"]
        actual = row["TARGETVAR"]
        if np.isnan(actual):
            continue
        
        past = mdf[mdf.index < ts]
        if len(past) < SEQ_LEN:
            continue
        inp = past.iloc[-SEQ_LEN:].values.astype(np.float32)
        if np.isnan(inp).any():
            continue
        
        x_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x_enc=x_t)  # (1, N_zones, 1, 99)
        pred_q = out[0, zone_id - 1, 0, :].cpu().numpy()
        
        pred_q = np.nan_to_num(pred_q, nan=0.5)  # replace NaN with median
        pred_q = np.clip(pred_q, 0.0, 1.0)
        
        # Benchmark
        br = z_bench[z_bench["TIMESTAMP"] == ts]
        if len(br) == 0:
            continue
        bq = br[bench_q_cols].values[0].astype(np.float32)
        
        zone_preds.append(pred_q)
        zone_actuals.append(actual)
        zone_bench_q.append(bq)
    
    if len(zone_preds) == 0:
        continue
    
    zone_preds = np.array(zone_preds)
    zone_actuals = np.array(zone_actuals)
    zone_bench_q = np.array(zone_bench_q)
    
    model_pb = [pinball_score(zone_actuals, zone_preds[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)]
    bench_pb = [pinball_score(zone_actuals, zone_bench_q[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)]
    
    all_model_preds.append(zone_preds)
    all_bench_preds.append(zone_bench_q)
    all_actuals.append(zone_actuals)
    
    print(f"  Zone {zone_id}: {len(zone_actuals)} samples | Model avg: {np.mean(model_pb):.6f} | Bench avg: {np.mean(bench_pb):.6f} | Improv: {(np.mean(bench_pb)-np.mean(model_pb))/np.mean(bench_pb)*100:.1f}%", flush=True)

# Overall results
all_preds = np.concatenate(all_model_preds, axis=0)
all_bench = np.concatenate(all_bench_preds, axis=0)
all_actual = np.concatenate(all_actuals, axis=0)

overall_model = np.mean([pinball_score(all_actual, all_preds[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)])
overall_bench = np.mean([pinball_score(all_actual, all_bench[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)])

print(f"\n{'='*60}", flush=True)
print(f"GEFCom2014 Task 15 — Overall Results", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Model pinball avg (99 q): {overall_model:.6f}", flush=True)
print(f"  Bench pinball avg (99 q): {overall_bench:.6f}", flush=True)
print(f"  Improvement:              {(overall_bench - overall_model) / overall_bench * 100:.1f}%", flush=True)
print(f"  Model P50 MAE:            {mean_absolute_error(all_actual, all_preds[:, 49]):.4f}", flush=True)
print(f"  Bench P50 MAE:            {mean_absolute_error(all_actual, all_bench[:, 49]):.4f}", flush=True)
print(f"  Model P50 RMSE:           {np.sqrt(np.mean((all_actual - all_preds[:, 49])**2)):.4f}", flush=True)
print(f"  Bench P50 RMSE:           {np.sqrt(np.mean((all_actual - all_bench[:, 49])**2)):.4f}", flush=True)
print(f"  Total samples: {len(all_actual)}", flush=True)

# Per-quantile breakdown
print(f"\n  Per-quantile pinball (all zones):", flush=True)
print(f"  {'Quantile':>8} | {'Model':>10} | {'Bench':>10} | {'Delta%':>8}", flush=True)
print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}", flush=True)
for qi in [9, 24, 49, 74, 89]:
    mp = pinball_score(all_actual, all_preds[:, qi], GEFCOM_QUANTILES[qi])
    bp = pinball_score(all_actual, all_bench[:, qi], GEFCOM_QUANTILES[qi])
    dp = (bp - mp) / bp * 100
    print(f"  {GEFCOM_QUANTILES[qi]:8.2f} | {mp:10.6f} | {bp:10.6f} | {dp:8.1f}%", flush=True)

# Save
np.savez_compressed(r"C:\Projects\raghavan\vayumithra_research\results\gefecom_task15_results.npz",
    model_preds=all_preds, bench_preds=all_bench, actuals=all_actual,
    quantiles=GEFCOM_QUANTILES)
print(f"\nResults saved.", flush=True)
