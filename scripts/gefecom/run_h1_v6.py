import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
from src.models.probabilistic_model import QUANTILES, iTransformerNHiTS_Probabilistic, PinballLoss

DATA_PATH = r"C:\Projects\raghavan\iTransformer\dataset\vayumithra\vayumithra_10st_uv.csv"
TARGET_STATION = "s5"

df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
ws_col = f"ws_{TARGET_STATION}"
df[ws_col] = np.sqrt(df[f"u_{TARGET_STATION}"].values**2 + df[f"v_{TARGET_STATION}"].values**2)
feature_cols = [c for c in df.columns if c != "datetime"]

class Config:
    def __init__(self, **kwargs):
        self.seq_len=336; self.pred_len=1; self.enc_in=len(feature_cols)
        self.d_model=256; self.n_heads=4; self.e_layers=2
        self.d_ff=1024; self.dropout=0.10; self.activation="gelu"
        self.embed="timeF"; self.freq="h"; self.factor=1
        self.class_strategy="projection"; self.use_norm=True
        self.output_attention=False; self.ws_channel=-1
        self.__dict__.update(kwargs)

def make_windows(df, fcols, ws_col, start, end, seq_len, pred_len):
    mask = (df["datetime"] >= start) & (df["datetime"] < end)
    sub = df[mask]
    vals = sub[fcols].values.astype(np.float32)
    ws_raw = sub[ws_col].values.astype(np.float32)
    xs, ys = [], []
    n = len(sub)
    for i in range(0, n - seq_len - pred_len + 1):
        x_win = vals[i: i + seq_len]
        if np.isnan(x_win).any(): continue
        y_win = ws_raw[i + seq_len + pred_len - 1]
        if np.isnan(y_win): continue
        xs.append(x_win); ys.append([y_win])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

fold = {"train_end": "2019-01-01", "val_start": "2019-01-01", "val_end": "2020-01-01",
        "test_start": "2020-01-01", "test_end": "2021-01-01"}

t0 = time.time()
tx, ty = make_windows(df, feature_cols, ws_col, df["datetime"].min(), fold["train_end"], 336, 1)
vx, vy = make_windows(df, feature_cols, ws_col, fold["val_start"], fold["val_end"], 336, 1)
tex, tey = make_windows(df, feature_cols, ws_col, fold["test_start"], fold["test_end"], 336, 1)
print(f"Windows ({time.time()-t0:.0f}s): train={len(tx)} val={len(vx)} test={len(tex)}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config()
model = iTransformerNHiTS_Probabilistic(cfg).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

batch_size = 16
train_ds = torch.utils.data.TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty))
val_ds = torch.utils.data.TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy))
test_ds = torch.utils.data.TensorDataset(torch.from_numpy(tex), torch.from_numpy(tey))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)

criterion = PinballLoss(quantiles=QUANTILES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_val_loss = float("inf")
patience_ctr = 0
t_start = time.time()
save_dir = "results/models/probabilistic"

for epoch in range(1, 101):
    model.train()
    tr_loss = 0.0; nb = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_enc=bx), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item(); nb += 1
    tr_loss /= nb

    model.eval()
    va_loss = 0.0; nv = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            va_loss += criterion(model(x_enc=bx), by).item(); nv += 1
    va_loss /= nv
    scheduler.step()

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Ep {epoch:3d}/100 | tr={tr_loss:.4f} va={va_loss:.4f} [{time.time()-t_start:.0f}s]", flush=True)

    if va_loss < best_val_loss:
        best_val_loss = va_loss; patience_ctr = 0
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": va_loss},
                   f"{save_dir}/probabilistic_fold1_h1.pt")
    else:
        patience_ctr += 1
        if patience_ctr >= 15:
            print(f"  Early stop at epoch {epoch} (best={best_val_loss:.4f})", flush=True)
            break

model.load_state_dict(torch.load(f"{save_dir}/probabilistic_fold1_h1.pt", map_location=device)["model_state"])
model.eval()
preds_list, ys_list = [], []
with torch.no_grad():
    for bx, by in test_loader:
        bx = bx.to(device)
        preds_list.append(model(x_enc=bx).cpu().numpy())
        ys_list.append(by.numpy())

preds = np.concatenate(preds_list, axis=0).squeeze(1)
y_true = np.concatenate(ys_list, axis=0).squeeze(1)

from sklearn.metrics import mean_absolute_error, mean_squared_error

def pinball(y, yh, q): e = y - yh; return float(np.mean(np.maximum(q*e, (q-1)*e)))
def coverage(y, lo, hi): return float(np.mean((y>=lo)&(y<=hi)))
def width(lo, hi): return float(np.mean(hi-lo))
def winkler(y, lo, hi, a=0.20): return float(np.mean((hi-lo) + (2/a)*np.maximum(lo-y,0) + (2/a)*np.maximum(y-hi,0)))

p10, p50, p90, p99 = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
print(f"\nResults:", flush=True)
print(f"  P50 MAE:     {mean_absolute_error(y_true, p50):.4f}", flush=True)
print(f"  P50 RMSE:    {np.sqrt(mean_squared_error(y_true, p50)):.4f}", flush=True)
print(f"  Coverage P10-P90: {coverage(y_true, p10, p90):.4f} (target ~0.80)", flush=True)
print(f"  Coverage P10-P99: {coverage(y_true, p10, p99):.4f}", flush=True)
print(f"  Interval width P10-P90: {width(p10, p90):.4f} m/s", flush=True)
print(f"  Winkler P10-P90: {winkler(y_true, p10, p90):.4f}", flush=True)
for lbl, q, pred_q in zip(["P10","P50","P90","P99"], [0.1,0.5,0.9,0.99], [p10,p50,p90,p99]):
    print(f"  {lbl} pinball: {pinball(y_true, pred_q, q):.4f}", flush=True)
print(f"  Pinball avg: {np.mean([pinball(y_true, p10, 0.1), pinball(y_true, p50, 0.5), pinball(y_true, p90, 0.9), pinball(y_true, p99, 0.99)]):.4f}", flush=True)
