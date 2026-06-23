import sys, os, time, numpy as np, pandas as pd, torch
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
from src.models.probabilistic_model import QUANTILES, iTransformerNHiTS_Probabilistic, PinballLoss

DATA_PATH = r"C:\Projects\raghavan\iTransformer\dataset\vayumithra\vayumithra_10st_uv.csv"
TARGET_STATION = "s5"
SAVE_DIR = "results/models/probabilistic"
HORIZONS = [2, 3, 4, 5, 6, 12, 24]

df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
ws_col = f"ws_{TARGET_STATION}"
df[ws_col] = np.sqrt(df[f"u_{TARGET_STATION}"].values**2 + df[f"v_{TARGET_STATION}"].values**2)
feature_cols = [c for c in df.columns if c != "datetime"]

class Config:
    def __init__(self, **kwargs):
        self.seq_len=336; self.pred_len=1; self.enc_in=len(feature_cols)
        self.d_model=128; self.n_heads=4; self.e_layers=2
        self.d_ff=512; self.dropout=0.15; self.activation="gelu"
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = PinballLoss(quantiles=QUANTILES).to(device)

for h in HORIZONS:
    save_path = f"{SAVE_DIR}/probabilistic_fold1_h{h}.pt"
    if os.path.exists(save_path):
        print(f"\n  H{h} already trained, skipping.", flush=True)
        continue
    print(f"\n{'='*60}", flush=True)
    print(f"Training H{h} ...", flush=True)
    t0 = time.time()
    tx, ty = make_windows(df, feature_cols, ws_col, df["datetime"].min(), fold["train_end"], 336, h)
    vx, vy = make_windows(df, feature_cols, ws_col, fold["val_start"], fold["val_end"], 336, h)
    print(f"  Windows: train={len(tx)} val={len(vx)} [{time.time()-t0:.0f}s]", flush=True)
    if len(tx) == 0 or len(vx) == 0:
        print(f"  SKIP: no windows", flush=True)
        continue

    cfg = Config(pred_len=h)
    model = iTransformerNHiTS_Probabilistic(cfg).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    batch_size = 128
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_loss = float("inf")
    patience_ctr = 0
    t_start = time.time()

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

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/100 | tr={tr_loss:.4f} va={va_loss:.4f} [{time.time()-t_start:.0f}s]", flush=True)

        if va_loss < best_val_loss:
            best_val_loss = va_loss; patience_ctr = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": va_loss}, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= 15:
                print(f"  Early stop at epoch {epoch} (best={best_val_loss:.4f})", flush=True)
                break

    print(f"  Done H{h} in {time.time()-t_start:.0f}s (best val={best_val_loss:.4f})", flush=True)
