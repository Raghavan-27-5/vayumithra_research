import sys, time, traceback
print("Script started", flush=True)
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
print("Path set", flush=True)
from src.models.probabilistic_model import QUANTILES, iTransformerNHiTS_Probabilistic, PinballLoss
print("Imports OK", flush=True)

import numpy as np, pandas as pd, torch
import argparse, json, logging
from pathlib import Path

DATA_PATH = r"C:\Projects\raghavan\iTransformer\dataset\vayumithra\vayumithra_10st_uv.csv"
TARGET_STATION = "s5"
ALL_HORIZONS = [1, 2, 3, 4, 5, 6, 12, 24, 48]

WALK_FORWARD_FOLDS = [
    {"fold": 1, "train_end": "2019-01-01", "val_start": "2019-01-01",
     "val_end": "2020-01-01", "test_start": "2020-01-01", "test_end": "2021-01-01"},
]

class Config:
    def __init__(self, **kwargs):
        self.seq_len = 336; self.pred_len = 1; self.enc_in = 50
        self.d_model = 256; self.n_heads = 4; self.e_layers = 2
        self.d_ff = 1024; self.dropout = 0.1; self.activation = "gelu"
        self.embed = "timeF"; self.freq = "h"; self.factor = 1
        self.class_strategy = "projection"; self.use_norm = True
        self.output_attention = False
        self.__dict__.update(kwargs)

def load_vayumithra_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    ws_key = f"ws_{TARGET_STATION}"
    df[ws_key] = np.sqrt(df[f"u_{TARGET_STATION}"].values ** 2 + df[f"v_{TARGET_STATION}"].values ** 2)
    return df

def make_windows(df, feature_cols, ws_col, start, end, train_mean, train_std, seq_len=336, pred_len=1):
    mask = (df["datetime"] >= start) & (df["datetime"] < end)
    sub = df[mask]
    vals = (sub[feature_cols].values.astype(np.float32) - train_mean) / train_std
    ws_raw = sub[ws_col].values.astype(np.float32)
    xs, ys = [], []
    n = len(sub)
    for i in range(0, n - seq_len - pred_len + 1):
        x_win = vals[i: i + seq_len]
        if np.isnan(x_win).any(): continue
        y_win = ws_raw[i + seq_len + pred_len - 1]
        if np.isnan(y_win): continue
        xs.append(x_win)
        ys.append([y_win])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

print("Loading data...", flush=True)
df = load_vayumithra_data()
ws_col = f"ws_{TARGET_STATION}"
feature_cols = [ws_col] + [c for c in df.columns if c != "datetime"]
print(f"Data: {len(df)} rows", flush=True)

for h in [1]:
    print(f"\n-- Horizon t+{h}h --", flush=True)
    fold = WALK_FORWARD_FOLDS[0]
    train_mask = df["datetime"] < fold["train_end"]
    train_df = df[train_mask]
    train_mean = train_df[feature_cols].mean().values.astype(np.float32)
    train_std = (train_df[feature_cols].std().values + 1e-8).astype(np.float32)
    t0 = time.time()
    tx, ty = make_windows(df, feature_cols, ws_col, df["datetime"].min(), fold["train_end"], train_mean, train_std, 336, h)
    vx, vy = make_windows(df, feature_cols, ws_col, fold["val_start"], fold["val_end"], train_mean, train_std, 336, h)
    tex, tey = make_windows(df, feature_cols, ws_col, fold["test_start"], fold["test_end"], train_mean, train_std, 336, h)
    print(f"Windows: train={len(tx)} val={len(vx)} test={len(tex)} [{time.time()-t0:.0f}s]", flush=True)

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy))
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(tex), torch.from_numpy(tey))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(pred_len=h, enc_in=tx.shape[2])
    model = iTransformerNHiTS_Probabilistic(cfg).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    criterion = PinballLoss(quantiles=QUANTILES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_loss = float("inf")
    patience_ctr = 0
    t_start = time.time()

    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0; n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(x_enc=bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item(); n_batches += 1
        train_loss /= n_batches

        model.eval()
        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(x_enc=bx)
                val_loss += criterion(pred, by).item(); n_val += 1
        val_loss /= n_val
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch}/50 | tr={train_loss:.4f} va={val_loss:.4f} [{time.time()-t_start:.0f}s]", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss; patience_ctr = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch},
                       f"results/models/probabilistic/probabilistic_fold1_h{h}.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= 10:
                print(f"  Early stop at epoch {epoch}", flush=True)
                break

    print(f"Best: epoch={epoch}, val_loss={best_val_loss:.4f} [{time.time()-t_start:.0f}s]", flush=True)
    print("DONE", flush=True)
