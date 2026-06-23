"""
GEFCom2014-W v2 training: full data range + ExpVars (future weather) injection.

Architecture flow:
  Input: (B, 336, 50) past window + (B, 40) future weather → iTransformer encode
  → inject weather token → NHiTS quantile head → (B, 10, 1, 99)

Data changes from v1:
  - Train on ALL available data (Jan 2012 → Nov 2013), not just pre-2013
  - Proper deduplication of cumulative tasks
  - Future weather (U10/V10/U100/V100) fed as additional token at target hour
  - Evaluation uses actual ExpVars for Dec 2013
"""
from __future__ import annotations
import sys, os, json, zipfile, io, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
from src.models.gefecom_model import iTransformerNHiTS_GEFCom, GEFCOM_QUANTILES
from src.models.probabilistic_model import PinballLoss

GEFCOM_DIR = r"C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind"
SEQ_LEN, N_ZONES, FEATS = 336, 10, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# ── Data loading ────────────────────────────────────────────────────────────
def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f"Task {task_num}")
    zf = zipfile.ZipFile(os.path.join(zd, f"Task{task_num}_W_Zone1_10.zip"))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

def load_expvars(task_num):
    """Load future weather forecasts for the test period of a task."""
    zd = os.path.join(GEFCOM_DIR, f"Task {task_num}")
    zf = zipfile.ZipFile(os.path.join(zd, f"TaskExpVars{task_num}_W_Zone1_10.zip"))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

# Load all tasks (1-15) and deduplicate
all_df = pd.concat([load_task(t) for t in range(1, 16)], ignore_index=True)
all_df["TIMESTAMP"] = pd.to_datetime(all_df["TIMESTAMP"], format="%Y%m%d %H:%M")
all_df = all_df.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
all_df = all_df.drop_duplicates(subset=["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
# Forward-fill sparse NaN (Z1-Z3 have ~112 NaN TARGETVAR values)
nan_before = all_df.isna().sum().sum()
all_df = all_df.groupby("ZONEID", group_keys=False).apply(lambda g: g.ffill())
nan_after = all_df.isna().sum().sum()
print(f"All data: {len(all_df)} rows ({all_df['TIMESTAMP'].min()} to {all_df['TIMESTAMP'].max()})  NaN: {nan_before}->{nan_after}", flush=True)

# Build multivariate array (10 zones × 5 features = 50 columns)
mdf_list = []
for z in range(1, N_ZONES + 1):
    zdf = all_df[all_df["ZONEID"] == z].set_index("TIMESTAMP")[["TARGETVAR","U10","V10","U100","V100"]]
    mdf_list.append(zdf)
mdf = pd.concat(mdf_list, axis=1, keys=[f"Z{z}" for z in range(1, N_ZONES + 1)])
mdf = mdf.sort_index()
target_mask = np.array([c[1] == "TARGETVAR" for c in mdf.columns])
target_channels = np.where(target_mask)[0].tolist()
weather_mask = np.array([c[1] != "TARGETVAR" for c in mdf.columns])  # 40 weather cols
weather_channels = np.where(weather_mask)[0].tolist()
print(f"Multivariate array: {mdf.shape} ({pd.Timestamp(mdf.index[0]).date()} to {pd.Timestamp(mdf.index[-1]).date()})", flush=True)
print(f"Target channels (10 TARGETVAR): {target_channels}", flush=True)
print(f"Weather channels (40): {len(weather_channels)}", flush=True)

# Split: train = before Dec 2013, test = Dec 2013
train_mdf = mdf[mdf.index < "2013-12-01"]
test_mdf = mdf[(mdf.index >= "2013-12-01") & (mdf.index < "2014-01-01")]
train_data = train_mdf.values.astype(np.float32)
test_data = test_mdf.values.astype(np.float32)
print(f"Train: {train_data.shape} ({train_mdf.index.min().date()} to {train_mdf.index.max().date()})", flush=True)
print(f"Test:  {test_data.shape} ({test_mdf.index.min().date()} to {test_mdf.index.max().date()})", flush=True)

# Load ExpVars for Task 15 test period
ev15 = load_expvars(15)
ev15["TIMESTAMP"] = pd.to_datetime(ev15["TIMESTAMP"], format="%Y%m%d %H:%M")
ev15 = ev15.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)
# Build weather-only multivariate (no TARGETVAR)
wdf_list = []
for z in range(1, N_ZONES + 1):
    zdf = ev15[ev15["ZONEID"] == z].set_index("TIMESTAMP")[["U10","V10","U100","V100"]]
    wdf_list.append(zdf)
wdf = pd.concat(wdf_list, axis=1, keys=[f"Z{z}" for z in range(1, N_ZONES + 1)])
wdf = wdf.sort_index()
print(f"ExpVars weather: {wdf.shape} ({wdf.index.min().date()} to {wdf.index.max().date()})", flush=True)

# ── Window generation with future weather ──────────────────────────────────
def make_windows_with_weather(data, seq_len, target_ch, weather_ch, pred_len=1):
    """Generate (past_window, future_weather, target) triples."""
    xs, ws, ys = [], [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        xw = data[i:i+seq_len]
        if np.isnan(xw).any():
            continue
        target_idx = i + seq_len + pred_len - 1
        ys.append(data[target_idx, target_ch])
        ws.append(data[target_idx, weather_ch])
        xs.append(xw)
    return (np.array(xs, dtype=np.float32), np.array(ws, dtype=np.float32),
            np.array(ys, dtype=np.float32))

tx, tw, ty = make_windows_with_weather(train_data, SEQ_LEN, target_channels, weather_channels)
print(f"Train windows: {len(tx)}", flush=True)

n_val = int(len(tx) * 0.1)

# ── Model ───────────────────────────────────────────────────────────────────
class Cfg:
    seq_len=SEQ_LEN; pred_len=1; enc_in=train_data.shape[1]; d_model=128
    n_heads=4; e_layers=2; d_ff=512; dropout=0.15; activation="gelu"
    embed="timeF"; freq="h"; factor=1; use_norm=True; output_attention=False
    quantiles=GEFCOM_QUANTILES; target_channels=target_channels; n_targets=len(target_channels)
    n_zones=N_ZONES

model = iTransformerNHiTS_GEFCom(Cfg()).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# Time-based validation: last 10% of chronological data
n_val = int(len(tx) * 0.1)
train_ds = torch.utils.data.TensorDataset(
    torch.from_numpy(tx[:-n_val]), torch.from_numpy(tw[:-n_val]), torch.from_numpy(ty[:-n_val]))
val_ds = torch.utils.data.TensorDataset(
    torch.from_numpy(tx[-n_val:]), torch.from_numpy(tw[-n_val:]), torch.from_numpy(ty[-n_val:]))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

criterion = PinballLoss(quantiles=GEFCOM_QUANTILES).to(device)
save_dir = "results/models/gefecom_v2"; os.makedirs(save_dir, exist_ok=True)
model_path = f"{save_dir}/gefecom_model_v3.pt"

opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

best_val, ctr, t0 = float("inf"), 0, time.time()
for ep in range(1, 101):
    model.train(); tr = 0.0; nb = 0
    for bxb, bwb, byb in train_loader:
        bxb, bwb, byb = bxb.to(device), bwb.to(device), byb.to(device)
        opt.zero_grad()
        pred = model(x_enc=bxb, x_future_weather=bwb)
        B, NT, S, NQ = pred.shape
        loss = criterion(pred.reshape(B*NT, S, NQ), byb.reshape(B*NT, S))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tr += loss.item(); nb += 1
    model.eval(); va = 0.0; nv = 0
    with torch.no_grad():
        for bxb, bwb, byb in val_loader:
            bxb, bwb, byb = bxb.to(device), bwb.to(device), byb.to(device)
            pred = model(x_enc=bxb, x_future_weather=bwb)
            B, NT, S, NQ = pred.shape
            va += criterion(pred.reshape(B*NT, S, NQ), byb.reshape(B*NT, S)).item(); nv += 1
    sched.step()
    va_avg = va / nv
    if ep % 5 == 0 or ep == 1:
        print(f"  Ep {ep:2d} | tr={tr/nb:.6f} va={va_avg:.6f} [{time.time()-t0:.0f}s]", flush=True)
    if va < best_val:
        best_val = va; ctr = 0
        torch.save({"model_state": model.state_dict()}, model_path)
    else:
        ctr += 1
        if ctr >= 15:
            print(f"  Early stop {ep} best={(best_val/nv):.6f}", flush=True); break

print(f"Training done in {time.time()-t0:.0f}s", flush=True)

# ── Evaluate on Task 15 (Dec 2013) with ExpVars ────────────────────────────
model.load_state_dict(torch.load(model_path, map_location=device)["model_state"])
model.eval()

# Load solution
sol = pd.read_csv(os.path.join(GEFCOM_DIR, "Solution to Task 15", "solution15_W.csv"))
sol["TIMESTAMP"] = pd.to_datetime(sol["TIMESTAMP"], format="%Y%m%d %H:%M")

# Load benchmark
bench = pd.read_csv(os.path.join(GEFCOM_DIR, "Task 15", "benchmark15_W.csv"))
bench["TIMESTAMP"] = pd.to_datetime(bench["TIMESTAMP"], format="%Y%m%d %H:%M")
bench_q_cols = [str(q) for q in GEFCOM_QUANTILES]
bench_q_vals = bench[bench_q_cols].values.astype(np.float32)

# Build full multivariate (all data including Dec for past context)
mdf_full_list = []
for z in range(1, N_ZONES + 1):
    zdf = all_df[all_df["ZONEID"] == z].set_index("TIMESTAMP")[["TARGETVAR","U10","V10","U100","V100"]]
    mdf_full_list.append(zdf)
mdf_full = pd.concat(mdf_full_list, axis=1, keys=[f"Z{z}" for z in range(1, N_ZONES + 1)])
mdf_full = mdf_full.sort_index()
full_data = mdf_full.values.astype(np.float32)

# ExpVars weather array: 10 zones x 4 vars (U10,V10,U100,V100) = 40 cols
wdf_list2 = []
for z in range(1, N_ZONES + 1):
    zdf = ev15[ev15["ZONEID"] == z].set_index("TIMESTAMP")[["U10","V10","U100","V100"]]
    wdf_list2.append(zdf)
wdf2 = pd.concat(wdf_list2, axis=1, keys=[f"Z{z}" for z in range(1, N_ZONES + 1)])
wdf2 = wdf2.sort_index()
weather_data = wdf2.values.astype(np.float32)  # (N_test, 40)
print(f"Weather data (ExpVars): {weather_data.shape}", flush=True)

def pinball_score(y, yh, q):
    e = y - yh; return np.mean(np.maximum(q*e, (q-1)*e))

all_model_preds, all_bench_preds, all_actuals = [], [], []

for zone_id in range(1, N_ZONES + 1):
    z_sol = sol[sol["ZONEID"] == zone_id].sort_values("TIMESTAMP")
    z_bench = bench[bench["ZONEID"] == zone_id].sort_values("TIMESTAMP")
    tgt_ch = target_channels[zone_id - 1]
    w_ch_slice = slice((zone_id - 1) * 4, zone_id * 4)  # 4 weather cols per zone

    zone_preds, zone_actuals, zone_bench_q = [], [], []

    for idx, row in z_sol.iterrows():
        ts = row["TIMESTAMP"]
        actual = row["TARGETVAR"]

        past = mdf_full[mdf_full.index < ts]
        if len(past) < SEQ_LEN:
            continue
        inp = past.iloc[-SEQ_LEN:].values.astype(np.float32)
        if np.isnan(inp).any():
            continue

        if np.isnan(actual):
            continue
        # Look up ExpVars weather for this timestamp
        wrow = wdf2[wdf2.index == ts]
        if len(wrow) == 0:
            continue
        wvals = wrow.values[0].astype(np.float32)  # (40,)

        x_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        w_t = torch.from_numpy(wvals).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x_enc=x_t, x_future_weather=w_t)
        pred_q = out[0, zone_id - 1, 0, :].cpu().numpy()
        pred_q = np.nan_to_num(pred_q, nan=0.5, posinf=1.0, neginf=0.0)
        pred_q = np.clip(pred_q, 0.0, 1.0)

        bench_row = z_bench[z_bench["TIMESTAMP"] == ts]
        if len(bench_row) == 0:
            continue
        bq = bench_row[bench_q_cols].values[0].astype(np.float32)

        zone_preds.append(pred_q)
        zone_actuals.append(actual)
        zone_bench_q.append(bq)

    if len(zone_preds) == 0:
        continue

    zone_preds = np.array(zone_preds)
    zone_actuals = np.array(zone_actuals)
    zone_bench_q = np.array(zone_bench_q)

    # Filter out NaN actuals
    valid = ~np.isnan(zone_actuals)
    if valid.sum() < len(valid):
        print(f"    Filtering {len(valid)-valid.sum()} NaN actuals", flush=True)
        zone_preds = zone_preds[valid]
        zone_actuals = zone_actuals[valid]
        zone_bench_q = zone_bench_q[valid]

    model_pinballs = [pinball_score(zone_actuals, zone_preds[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)]
    bench_pinballs = [pinball_score(zone_actuals, zone_bench_q[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)]
    model_avg = np.mean(model_pinballs)
    bench_avg = np.mean(bench_pinballs)
    impr = (bench_avg - model_avg) / bench_avg * 100

    all_model_preds.append(zone_preds)
    all_bench_preds.append(zone_bench_q)
    all_actuals.append(zone_actuals)

    print(f"  Zone {zone_id}: {len(zone_actuals)} samples  model={model_avg:.6f}  bench={bench_avg:.6f}  impr={impr:.1f}%", flush=True)

# Overall
all_preds = np.concatenate(all_model_preds, axis=0)
all_bench = np.concatenate(all_bench_preds, axis=0)
all_actual = np.concatenate(all_actuals, axis=0)

overall_model = np.mean([pinball_score(all_actual, all_preds[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)])
overall_bench = np.mean([pinball_score(all_actual, all_bench[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)])
overall_impr = (overall_bench - overall_model) / overall_bench * 100

from sklearn.metrics import mean_absolute_error
print(f"\n{'='*50}", flush=True)
print(f"OVERALL Task 15 Results (with ExpVars):", flush=True)
print(f"  Model pinball avg: {overall_model:.6f}", flush=True)
print(f"  Bench pinball avg: {overall_bench:.6f}", flush=True)
print(f"  Improvement: {overall_impr:.1f}%", flush=True)
print(f"  P50 MAE:     {mean_absolute_error(all_actual, all_preds[:, 49]):.4f}", flush=True)
print(f"  Bench P50 MAE: {mean_absolute_error(all_actual, all_bench[:, 49]):.4f}", flush=True)

# Per-quantile breakdown
print(f"\nPer-quantile pinball:", flush=True)
print(f"{'q':>5} | {'Model':>10} | {'Bench':>10} | {'Delta%':>8}", flush=True)
print("-" * 40, flush=True)
for qi in [9, 24, 49, 74, 89]:
    mp = pinball_score(all_actual, all_preds[:, qi], GEFCOM_QUANTILES[qi])
    bp = pinball_score(all_actual, all_bench[:, qi], GEFCOM_QUANTILES[qi])
    dp = (bp - mp) / bp * 100
    print(f"{GEFCOM_QUANTILES[qi]:5.2f} | {mp:10.6f} | {bp:10.6f} | {dp:8.1f}%", flush=True)

# Save
np.savez_compressed(f"{save_dir}/task15_v2_results.npz",
    model_preds=all_preds, bench_preds=all_bench, actuals=all_actual)
print(f"\nResults saved to {save_dir}/task15_v2_results.npz", flush=True)
