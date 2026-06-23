"""MLP baseline: predict TARGETVAR from future weather ONLY (no past window)."""
import sys, os, zipfile, io, time, numpy as np, pandas as pd, torch

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
N_ZONES = 10

def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'Task{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

def load_expvars(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'TaskExpVars{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

all_df = pd.concat([load_task(t) for t in range(1, 16)], ignore_index=True)
all_df['TIMESTAMP'] = pd.to_datetime(all_df['TIMESTAMP'], format='%Y%m%d %H:%M')
all_df = all_df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
all_df = all_df.drop_duplicates(subset=['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
all_df = all_df.groupby('ZONEID', group_keys=False).apply(lambda g: g.ffill())

# Build training data: for each timestamp, use weather at that time to predict TARGETVAR
train_df = all_df[all_df['TIMESTAMP'] < '2013-12-01'].copy()
train_df['U10_scaled'] = train_df['U10']
train_df['V10_scaled'] = train_df['V10']
train_df['U100_scaled'] = train_df['U100']
train_df['V100_scaled'] = train_df['V100']

# Also add zone one-hot
zone_dummies = pd.get_dummies(train_df['ZONEID'], prefix='Z').astype(np.float32)
X_train = np.column_stack([train_df[['U10','V10','U100','V100']].values.astype(np.float32), zone_dummies.values])
y_train = train_df['TARGETVAR'].values.astype(np.float32)
valid = ~np.isnan(y_train)
X_train, y_train = X_train[valid], y_train[valid]
print(f'Training: {len(X_train)} samples')

# Build test data: use ExpVars + solution
sol = pd.read_csv(os.path.join(GEFCOM_DIR, 'Solution to Task 15', 'solution15_W.csv'))
sol['TIMESTAMP'] = pd.to_datetime(sol['TIMESTAMP'], format='%Y%m%d %H:%M')
ev = load_expvars(15)
ev['TIMESTAMP'] = pd.to_datetime(ev['TIMESTAMP'], format='%Y%m%d %H:%M')

# Merge solution with ExpVars on timestamp+zoneid
test_df = sol.merge(ev, on=['TIMESTAMP','ZONEID'], suffixes=('_actual','_forecast'))
zone_dummies_test = pd.get_dummies(test_df['ZONEID'], prefix='Z').astype(np.float32)
X_test = np.column_stack([test_df[['U10','V10','U100','V100']].values.astype(np.float32), zone_dummies_test.values])
y_test = test_df['TARGETVAR'].values.astype(np.float32)
print(f'Test: {len(X_test)} samples')

# Standardize
x_mean, x_std = X_train.mean(0), X_train.std(0) + 1e-8
X_train_s = (X_train - x_mean) / x_std
X_test_s = (X_test - x_mean) / x_std

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MLP: weather -> TARGETVAR quantiles
class MLPQuantile(torch.nn.Module):
    def __init__(self, in_dim, n_quantiles=99):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, n_quantiles),
        )
    def forward(self, x):
        return self.net(x)
    def enforce_monotonicity(self, preds):
        base = torch.sigmoid(preds[..., :1]) * 0.5
        diffs = torch.nn.functional.softplus(preds[..., 1:] - preds[..., :-1])
        return torch.cumsum(torch.cat([base, diffs], dim=-1), dim=-1).clamp(0.0, 1.0)

in_dim = X_train_s.shape[1]
GEFCOM_QUANTILES = [round(i*0.01, 2) for i in range(1, 100)]

class PinballLoss(torch.nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32))
    def forward(self, pred, target):
        error = target.unsqueeze(-1) - pred
        return torch.max(self.quantiles * error, (self.quantiles - 1) * error).mean()

def enforce_monotonicity_np(preds):
    """Enforce monotonic increasing quantiles."""
    base = preds[:, :1].copy()
    diffs = np.maximum(0, np.diff(preds))
    return np.concatenate([base, diffs], axis=1).cumsum(axis=1)

model = MLPQuantile(in_dim).to(device)
criterion = PinballLoss(GEFCOM_QUANTILES).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

B = 1024
dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train_s), torch.from_numpy(y_train))
loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True, num_workers=0)

best_loss = float('inf')
t0 = time.time()
for ep in range(1, 101):
    model.train()
    tr = 0.0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        opt.zero_grad()
        raw = model(bx)
        pred = model.enforce_monotonicity(raw)
        loss = criterion(pred, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tr += loss.item()
    sched.step()
    if ep % 10 == 0 or ep == 1:
        print(f'  Ep {ep:3d} | loss={tr/len(loader):.6f} [{time.time()-t0:.0f}s]', flush=True)

# Evaluate
model.eval()
X_te_t = torch.from_numpy(X_test_s).to(device)
with torch.no_grad():
    raw = model(X_te_t)
    pred_q = model.enforce_monotonicity(raw).cpu().numpy()
    pred_q = np.clip(pred_q, 0.0, 1.0)

# Filter NaN actuals
valid = ~np.isnan(y_test)
y_test = y_test[valid]
pred_q = pred_q[valid]
test_df = test_df[valid].reset_index(drop=True)

# Compute pinball per zone
print()
for zone_id in range(1, N_ZONES + 1):
    mask = test_df['ZONEID'].values == zone_id
    if mask.sum() == 0:
        continue
    yz = y_test[mask]
    pz = pred_q[mask]
    pz_m = enforce_monotonicity_np(pz)
    mp = np.mean([np.mean(np.maximum(GEFCOM_QUANTILES[qi]*(yz-pz_m[:,qi]), (GEFCOM_QUANTILES[qi]-1)*(yz-pz_m[:,qi]))) for qi in range(99)])
    print(f'  Z{zone_id:2d}: {mask.sum():4d} samples  pinball={mp:.6f}')

# Overall
pred_q_m = enforce_monotonicity_np(pred_q)
overall = np.mean([np.mean(np.maximum(GEFCOM_QUANTILES[qi]*(y_test-pred_q_m[:,qi]), (GEFCOM_QUANTILES[qi]-1)*(y_test-pred_q_m[:,qi]))) for qi in range(99)])
print(f'\nOVERALL MLP (weather only): {overall:.6f}')
print(f'Benchmark:                   0.0792')
print(f'Impr: {(0.0792-overall)/0.0792*100:.1f}%')
