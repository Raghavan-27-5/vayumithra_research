import sys, os, zipfile, io, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
from src.models.dlinear_probabilistic import DLinear_Probabilistic
from src.models.probabilistic_model import GEFCOM_QUANTILES, PinballLoss

GEFCOM_DIR = r"C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind"
SEQ_LEN, N_ZONES, FEATS = 336, 10, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f"Task {task_num}")
    zf = zipfile.ZipFile(os.path.join(zd, f"Task{task_num}_W_Zone1_10.zip"))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

all_df = pd.concat([load_task(t) for t in range(1, 15)], ignore_index=True)
all_df["TIMESTAMP"] = pd.to_datetime(all_df["TIMESTAMP"], format="%Y%m%d %H:%M")
all_df = all_df.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)

train_df = all_df[all_df["TIMESTAMP"] < "2013-01-01"].copy()
mdf_list = []
for z in range(1, N_ZONES + 1):
    zdf = train_df[train_df["ZONEID"] == z].set_index("TIMESTAMP")[["TARGETVAR","U10","V10","U100","V100"]]
    mdf_list.append(zdf)
mdf = pd.concat(mdf_list, axis=1, keys=[f"Z{z}" for z in range(1, N_ZONES + 1)])
mdf = mdf.sort_index()
train_data = mdf.values.astype(np.float32)
target_mask = np.array([c[1] == "TARGETVAR" for c in mdf.columns])
target_channels = np.where(target_mask)[0].tolist()
print(f"Train data: {train_data.shape} ({pd.Timestamp(mdf.index[0]).date()} to {pd.Timestamp(mdf.index[-1]).date()})", flush=True)
print(f"Target channels: {target_channels}", flush=True)

def make_windows(data, seq_len, pred_len=1):
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        xw = data[i:i+seq_len]
        if np.isnan(xw).any(): continue
        ys.append(data[i+seq_len+pred_len-1, target_channels])
        xs.append(xw)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

tx, ty = make_windows(train_data, SEQ_LEN)
print(f"Windows: {len(tx)}", flush=True)
n_val = int(len(tx) * 0.1)
perm = np.random.RandomState(42).permutation(len(tx))
tx, ty = tx[perm], ty[perm]

class Cfg:
    seq_len=SEQ_LEN; pred_len=1; enc_in=train_data.shape[1]
    d_model=128; kernel_size=25; use_norm=True
    quantiles=GEFCOM_QUANTILES
    target_channels=target_channels; n_targets=len(target_channels)

model = DLinear_Probabilistic(Cfg()).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

bs = 64
train_ds = torch.utils.data.TensorDataset(torch.from_numpy(tx[:-n_val]), torch.from_numpy(ty[:-n_val]))
val_ds = torch.utils.data.TensorDataset(torch.from_numpy(tx[-n_val:]), torch.from_numpy(ty[-n_val:]))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs*2, shuffle=False, num_workers=0)

criterion = PinballLoss(quantiles=GEFCOM_QUANTILES).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
save_dir = "results/models/dlinear_gefecom"
os.makedirs(save_dir, exist_ok=True)

best_val, ctr, t0 = float("inf"), 0, time.time()
for ep in range(1, 51):
    model.train(); tr = 0.0; nb = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        pred = model(x_enc=bx)
        B, NT, S, NQ = pred.shape
        loss = criterion(pred.reshape(B*NT, S, NQ), by.reshape(B*NT, S))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tr += loss.item(); nb += 1
    model.eval(); va = 0.0; nv = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(x_enc=bx)
            B, NT, S, NQ = pred.shape
            va += criterion(pred.reshape(B*NT, S, NQ), by.reshape(B*NT, S)).item(); nv += 1
    sched.step()
    if ep % 5 == 0 or ep == 1:
        print(f"  Ep {ep:2d} | tr={tr/nb:.6f} va={va/nv:.6f} [{time.time()-t0:.0f}s]", flush=True)
    if va < best_val:
        best_val = va; ctr = 0
        torch.save({"model_state": model.state_dict()}, f"{save_dir}/dlinear_model.pt")
    else:
        ctr += 1
        if ctr >= 10:
            print(f"  Early stop {ep} best={best_val:.6f}", flush=True); break

print(f"Trained in {time.time()-t0:.0f}s", flush=True)
print(f"Best val pinball: {best_val:.6f}", flush=True)

# ── Evaluate on Task 15 ────────────────────────────────────────────────────
model.load_state_dict(torch.load(f"{save_dir}/dlinear_model.pt", map_location=device)["model_state"])
model.eval()

task15_df = load_task(15)
task15_df["TIMESTAMP"] = pd.to_datetime(task15_df["TIMESTAMP"], format="%Y%m%d %H:%M")

sol = pd.read_csv(os.path.join(GEFCOM_DIR, "Solution to Task 15", "solution15_W.csv"))
sol["TIMESTAMP"] = pd.to_datetime(sol["TIMESTAMP"], format="%Y%m%d %H:%M")

bench = pd.read_csv(os.path.join(GEFCOM_DIR, "Task 15", "benchmark15_W.csv"))
bench["TIMESTAMP"] = pd.to_datetime(bench["TIMESTAMP"], format="%Y%m%d %H:%M")

full15_df = all_df[all_df["TIMESTAMP"] < "2014-01-01"].copy()
mdf15_list = []
for z in range(1, N_ZONES + 1):
    zdf = full15_df[full15_df["ZONEID"] == z].set_index("TIMESTAMP")[["TARGETVAR","U10","V10","U100","V100"]]
    mdf15_list.append(zdf)
mdf15 = pd.concat(mdf15_list, axis=1, keys=[f"Z{z}" for z in range(1, N_ZONES + 1)])
mdf15 = mdf15.sort_index()

from sklearn.metrics import mean_absolute_error

def pinball_score(y, yh, q):
    e = y - yh
    return float(np.mean(np.maximum(q*e, (q-1)*e)))

all_model_preds = []
all_bench_preds = []
all_actuals = []

for zone_id in range(1, N_ZONES + 1):
    z_sol = sol[sol["ZONEID"] == zone_id].sort_values("TIMESTAMP")
    z_bench = bench[bench["ZONEID"] == zone_id].sort_values("TIMESTAMP")
    tgt_ch = target_channels[zone_id - 1]

    zone_preds = []
    zone_actuals = []
    zone_bench_q = []

    for idx, row in z_sol.iterrows():
        ts = row["TIMESTAMP"]
        actual = row["TARGETVAR"]
        past = mdf15[mdf15.index < ts]
        if len(past) < SEQ_LEN:
            continue
        inp = past.iloc[-SEQ_LEN:].values.astype(np.float32)
        if np.isnan(inp).any():
            continue

        x_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x_enc=x_t)
        pred_q = out[0, zone_id - 1, 0, :].cpu().numpy()

        bench_row = z_bench[z_bench["TIMESTAMP"] == ts]
        if len(bench_row) == 0:
            continue
        bq = bench_row[[f"{q:.2f}" for q in GEFCOM_QUANTILES]].values[0].astype(np.float32)

        zone_preds.append(pred_q)
        zone_actuals.append(actual)
        zone_bench_q.append(bq)

    if len(zone_preds) == 0:
        continue

    zone_preds = np.array(zone_preds)
    zone_actuals = np.array(zone_actuals)
    zone_bench_q = np.array(zone_bench_q)

    model_pinballs = [pinball_score(zone_actuals, zone_preds[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)]
    bench_pinballs = [pinball_score(zone_actuals, zone_bench_q[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)]
    model_avg = np.mean(model_pinballs)
    bench_avg = np.mean(bench_pinballs)

    all_model_preds.append(zone_preds)
    all_bench_preds.append(zone_bench_q)
    all_actuals.append(zone_actuals)

    print(f"  Zone {zone_id}: {len(zone_actuals)} samples", flush=True)
    print(f"    DLinear pinball avg: {model_avg:.6f}", flush=True)
    print(f"    Bench pinball avg:   {bench_avg:.6f}", flush=True)

all_preds = np.concatenate(all_model_preds, axis=0)
all_bench = np.concatenate(all_bench_preds, axis=0)
all_actual = np.concatenate(all_actuals, axis=0)

overall_model = np.mean([pinball_score(all_actual, all_preds[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)])
overall_bench = np.mean([pinball_score(all_actual, all_bench[:, qi], GEFCOM_QUANTILES[qi]) for qi in range(99)])

print(f"\n{'='*50}", flush=True)
print(f"DLinear OVERALL Task 15 Results:", flush=True)
print(f"  Model pinball avg: {overall_model:.6f}", flush=True)
print(f"  Bench pinball avg: {overall_bench:.6f}", flush=True)
print(f"  P50 MAE:     {mean_absolute_error(all_actual, all_preds[:, 49]):.4f}", flush=True)
print(f"  Bench P50 MAE: {mean_absolute_error(all_actual, all_bench[:, 49]):.4f}", flush=True)

for qi in [9, 24, 49, 74, 89]:
    mp = pinball_score(all_actual, all_preds[:, qi], GEFCOM_QUANTILES[qi])
    bp = pinball_score(all_actual, all_bench[:, qi], GEFCOM_QUANTILES[qi])
    print(f"  P{int(GEFCOM_QUANTILES[qi]*100)}: DLinear={mp:.6f} Bench={bp:.6f} Ratio={mp/bp:.2f}x")

np.savez_compressed(f"{save_dir}/dlinear_task15_results.npz",
    model_preds=all_preds, bench_preds=all_bench, actuals=all_actual)
print(f"\nResults saved to {save_dir}/dlinear_task15_results.npz", flush=True)
