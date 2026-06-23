import sys, os, json, numpy as np, pandas as pd, torch
sys.path.insert(0, r"C:\Projects\raghavan\vayumithra_research")
from src.models.probabilistic_model import QUANTILES, iTransformerNHiTS_Probabilistic

DATA_PATH = r"C:\Projects\raghavan\iTransformer\dataset\vayumithra\vayumithra_10st_uv.csv"
SAVE_DIR = "results/models/probabilistic"
HORIZONS = [1, 2, 3, 4, 5, 6, 12, 24]

df = pd.read_csv(DATA_PATH, parse_dates=["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
ws_col = "ws_s5"
df[ws_col] = np.sqrt(df["u_s5"].values**2 + df["v_s5"].values**2)
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

fold = {"test_start": "2020-01-01", "test_end": "2021-01-01"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {}

for h in HORIZONS:
    save_path = f"{SAVE_DIR}/probabilistic_fold1_h{h}.pt"
    if not os.path.exists(save_path):
        print(f"  H{h}: no checkpoint found, skipping", flush=True)
        continue

    tex, tey = make_windows(df, feature_cols, ws_col, fold["test_start"], fold["test_end"], 336, h)
    print(f"H{h}: {len(tex)} test windows", flush=True)

    cfg = Config(pred_len=h)
    model = iTransformerNHiTS_Probabilistic(cfg).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device)["model_state"])
    model.eval()

    batch_size = 256
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(tex), torch.from_numpy(tey))
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds_list, ys_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds_list.append(model(x_enc=bx).cpu().numpy())
            ys_list.append(by.numpy())

    preds = np.concatenate(preds_list, axis=0)  # (N, S, 4)
    y_true = np.concatenate(ys_list, axis=0).squeeze(-1)  # (N, S) or (N, 1)

    # The model was trained to predict y(t+S) for ALL S steps.
    # Take the LAST step prediction and compare to the single target.
    if preds.shape[2] == 4 and len(y_true.shape) == 1:
        # Single target per sample (y(t+S) only)
        last_step = preds[:, -1, :]  # (N, 4)
        metrics_y = y_true
    else:
        last_step = preds[:, -1, :]  # (N, 4)
        metrics_y = y_true[:, -1] if y_true.ndim > 1 else y_true

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    def pinball(y, yh, q):
        e = y - yh
        return float(np.mean(np.maximum(q*e, (q-1)*e)))

    def coverage(y, lo, hi):
        return float(np.mean((y >= lo) & (y <= hi)))

    def avg_width(lo, hi):
        return float(np.mean(hi - lo))

    def winkler(y, lo, hi, a=0.20):
        return float(np.mean((hi - lo) + (2/a)*np.maximum(lo-y, 0) + (2/a)*np.maximum(y-hi, 0)))

    p10, p50, p90, p99 = last_step[:, 0], last_step[:, 1], last_step[:, 2], last_step[:, 3]

    r = {
        "horizon": h,
        "test_samples": len(metrics_y),
        "p50_mae": round(float(mean_absolute_error(metrics_y, p50)), 4),
        "p50_rmse": round(float(np.sqrt(mean_squared_error(metrics_y, p50))), 4),
        "coverage_p10_p90": round(coverage(metrics_y, p10, p90), 4),
        "coverage_p10_p99": round(coverage(metrics_y, p10, p99), 4),
        "width_p10_p90": round(avg_width(p10, p90), 4),
        "width_p10_p99": round(avg_width(p10, p99), 4),
        "winkler_p10_p90": round(winkler(metrics_y, p10, p90), 4),
        "pinball_p10": round(pinball(metrics_y, p10, 0.1), 4),
        "pinball_p50": round(pinball(metrics_y, p50, 0.5), 4),
        "pinball_p90": round(pinball(metrics_y, p90, 0.9), 4),
        "pinball_p99": round(pinball(metrics_y, p99, 0.99), 4),
        "pinball_avg": round(np.mean([pinball(metrics_y, p10, 0.1), pinball(metrics_y, p50, 0.5), pinball(metrics_y, p90, 0.9), pinball(metrics_y, p99, 0.99)]), 4),
    }
    results[h] = r

    print(f"  P50 MAE:    {r['p50_mae']:.4f}", flush=True)
    print(f"  Coverage:   {r['coverage_p10_p90']:.4f} (target ~0.80)", flush=True)
    print(f"  Pinball avg:{r['pinball_avg']:.4f}", flush=True)
    print(f"  Width:      {r['width_p10_p90']:.4f} m/s", flush=True)
    print()

out_path = "results/probabilistic_results.json"
json.dump(results, open(out_path, "w"), indent=2)
print(f"Results saved to {out_path}", flush=True)

print("\n=== SUMMARY TABLE ===", flush=True)
print(f"{'H':>4} | {'P50 MAE':>8} | {'P50 RMSE':>8} | {'Cov 10-90':>9} | {'Width':>6} | {'Winkler':>8} | {'PinAvg':>6}", flush=True)
print("-" * 65, flush=True)
for h in [1, 2, 3, 4, 5, 6, 12, 24]:
    r = results.get(h)
    if r:
        print(f"{h:4d} | {r['p50_mae']:8.4f} | {r['p50_rmse']:8.4f} | {r['coverage_p10_p90']:9.4f} | {r['width_p10_p90']:6.2f} | {r['winkler_p10_p90']:8.4f} | {r['pinball_avg']:6.4f}", flush=True)
