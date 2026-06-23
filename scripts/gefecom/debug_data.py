import sys, traceback, numpy as np, pandas as pd, torch
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
from src.models.probabilistic_model import (
    QUANTILES, iTransformerNHiTS_Probabilistic, PinballLoss,
)
print("Imports OK")

DATA_PATH = r"C:\Projects\raghavan\iTransformer\dataset\vayumithra\vayumithra_10st_uv.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
ws_col = "ws_s5"
df[ws_col] = np.sqrt(df["u_s5"].values ** 2 + df["v_s5"].values ** 2)
print(f"Data: {len(df)} rows")

feature_cols = [c for c in df.columns if c != "datetime"]
feature_cols = [ws_col] + feature_cols

# fold 1: train < 2019, val 2019-2020, test 2020-2021
train_mask = df["datetime"] < "2019-01-01"
train_df = df[train_mask]
train_mean = train_df[feature_cols].mean().values.astype(np.float32)
train_std = (train_df[feature_cols].std().values + 1e-8).astype(np.float32)

def make_windows(start, end, seq_len=336, pred_len=1):
    mask = (df["datetime"] >= start) & (df["datetime"] < end)
    sub = df[mask]
    vals = (sub[feature_cols].values.astype(np.float32) - train_mean) / train_std
    ws_raw = sub[ws_col].values.astype(np.float32)
    xs, ys = [], []
    n = len(sub)
    for i in range(0, n - seq_len - pred_len + 1):
        x_win = vals[i: i + seq_len]
        if np.isnan(x_win).any():
            continue
        y_win = ws_raw[i + seq_len + pred_len - 1]
        if np.isnan(y_win):
            continue
        xs.append(x_win)
        ys.append([y_win])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

t0 = __import__('time').time()
tx, ty = make_windows(df["datetime"].min(), "2019-01-01")
vx, vy = make_windows("2019-01-01", "2020-01-01")
tex, tey = make_windows("2020-01-01", "2021-01-01")
print(f"Windows in {__import__('time').time()-t0:.0f}s: train={len(tx)} val={len(vx)} test={len(tex)}")

tx_t = torch.from_numpy(tx)
ty_t = torch.from_numpy(ty)

class Cfg:
    seq_len, pred_len, enc_in = 336, 1, tx.shape[2]
    d_model, n_heads, e_layers = 256, 4, 2
    d_ff, dropout = 1024, 0.1
    activation, embed, freq = "gelu", "timeF", "h"
    factor, class_strategy = 1, "projection"
    use_norm, output_attention = True, False

model = iTransformerNHiTS_Probabilistic(Cfg())
train_ds = torch.utils.data.TensorDataset(tx_t[:1000], ty_t[:1000])
loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

criterion = PinballLoss(quantiles=QUANTILES)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()
for epoch in range(2):
    for bx, by in loader:
        optimizer.zero_grad()
        pred = model(x_enc=bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")

print("Smoke test PASSED")
