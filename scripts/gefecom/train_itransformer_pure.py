"""Pure iTransformer for GEFCom2014 Wind track.
Adapted from https://github.com/thuml/iTransformer

Architecture:
- Input: (B, 336, 50) — 336 time steps × 50 channels (10 zones × 5 vars)
- DataEmbedding_inverted: permute to (B, 50, 336) → Linear(336, 128) → (B, 50, 128)
  Each variable becomes a token with its 336-length history embedded
- Encoder: 2-layer Transformer with self-attention across 50 variable tokens
- Projector: Linear(128, 99) → outputs 99 quantiles per variable token
- Only TARGETVAR channels (0,5,10,...,45) are used for pinball loss

Test: auto-regressive with weather updating from ExpVars
"""

import sys, os, zipfile, io, time, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'C:\Users\Nandha\AppData\Local\Temp\opencode\iTransformer')
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
N_ZONES = 10
QUANTILES = np.arange(0.01, 1.0, 0.01)
SEQ_LEN = 336
N_CHANNELS = 50  # 10 zones × 5 vars (TARGETVAR, U10, V10, U100, V100)
TARGET_CHANNELS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}', flush=True)


class iTransformer_GEFCom(nn.Module):
    """Pure iTransformer adapted for GEFCom2014 with quantile output."""

    def __init__(self, seq_len=SEQ_LEN, d_model=128, n_heads=4, e_layers=2,
                 d_ff=512, dropout=0.15, activation='gelu', use_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.use_norm = use_norm

        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, 'timeF', 'h', dropout)

        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(FullAttention(False, 1, attention_dropout=dropout, output_attention=False),
                               d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation=activation
            ) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projector = nn.Linear(d_model, 99)

    def forward(self, x_enc):
        """x_enc: (B, L, N) where L=seq_len, N=channels."""
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        enc_out = self.enc_embedding(x_enc, None)  # (B, N, d_model)
        enc_out, _ = self.encoder(enc_out)          # (B, N, d_model)
        dec_out = self.projector(enc_out)           # (B, N, 99)

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(-1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(-1))

        return dec_out  # (B, N, 99)


def pinball_loss(preds, target, quantiles):
    """preds: (B, N, 99), target: (B, N) — loss only on target channels."""
    B, N, Q = preds.shape
    target = target.unsqueeze(-1).expand_as(preds)
    delta = target - preds
    loss = torch.max(quantiles.view(1, 1, Q) * delta, (quantiles.view(1, 1, Q) - 1) * delta)
    return loss.mean()


def monotonicity_loss(preds):
    """Soft penalty for quantile crossing: encourage q_i <= q_{i+1}."""
    diff = preds[:, :, 1:] - preds[:, :, :-1]
    return F.relu(-diff).mean()


def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'Task{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    df = pd.concat(frames, ignore_index=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    df = df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
    df = df.groupby('ZONEID', group_keys=False).apply(lambda g: g.ffill())
    return df


def load_expvars(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'TaskExpVars{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    df = pd.concat(frames, ignore_index=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    df = df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
    return df


def build_training_samples(task_df, seq_len=SEQ_LEN):
    """Build (X, y) pairs from task data.
    X: (n_samples, seq_len, N_CHANNELS)
    y: (n_samples, N_CHANNELS) — next step's TARGETVAR (weather ignored in loss)
    """
    all_X, all_y = [], []
    for z in range(1, N_ZONES + 1):
        zdf = task_df[task_df['ZONEID'] == z].sort_values('TIMESTAMP').reset_index(drop=True)
        vals = zdf[['TARGETVAR', 'U10', 'V10', 'U100', 'V100']].values.astype(np.float32)
        n = len(vals)
        if n <= seq_len:
            continue
        for i in range(seq_len, n):
            all_X.append(vals[i - seq_len:i])       # (seq_len, 5)
            all_y.append(vals[i])                    # (5,) — all 5 channels
    if not all_X:
        return np.zeros((0, seq_len, 5)), np.zeros((0, 5))
    X = np.stack(all_X, axis=0)  # (n_samples, seq_len, 5)
    y = np.stack(all_y, axis=0)  # (n_samples, 5)
    return X, y


def interleave_zones(X_per_zone, y_per_zone):
    """Interleave zone samples to create multi-zone training data.
    Each sample covers all 10 zones simultaneously.
    X: (total_samples_per_zone, seq_len, 5) per zone
    y: (total_samples_per_zone, 5) per zone
    Returns: X_all (n, seq_len, 50), y_all (n, 50)
    """
    # Find common sample count
    n = min(len(Xz) for Xz in X_per_zone.values())
    if n == 0:
        return np.zeros((0, SEQ_LEN, N_CHANNELS)), np.zeros((0, N_CHANNELS))
    
    X_list, y_list = [], []
    for i in range(n):
        x_row = np.concatenate([X_per_zone[z][i] for z in range(1, N_ZONES + 1)], axis=-1)  # (seq_len, 50)
        y_row = np.concatenate([y_per_zone[z][i] for z in range(1, N_ZONES + 1)])           # (50,)
        X_list.append(x_row)
        y_list.append(y_row)
    return np.stack(X_list, axis=0), np.stack(y_list, axis=0)


def build_dataset(task_df):
    """Build multi-zone dataset for one task."""
    X_per_zone, y_per_zone = {}, {}
    for z in range(1, N_ZONES + 1):
        Xz, yz = build_training_samples(task_df, SEQ_LEN)
        X_per_zone[z] = Xz
        y_per_zone[z] = yz
    return interleave_zones(X_per_zone, y_per_zone)


def train_model(model, X_train, y_train, n_epochs=50, lr=1e-3, batch_size=128):
    """Train iTransformer model with pinball loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    quantiles_t = torch.tensor(QUANTILES, dtype=torch.float32, device=DEVICE)
    n = len(X_train)
    best_loss = float('inf')
    
    for epoch in range(n_epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.from_numpy(X_train[idx]).to(DEVICE)
            yb = torch.from_numpy(y_train[idx]).to(DEVICE)
            
            opt.zero_grad()
            preds = model(xb)  # (B, 50, 99)
            
            # Pinball loss on target channels only
            target_preds = preds[:, TARGET_CHANNELS, :]  # (B, 10, 99)
            target_actual = yb[:, TARGET_CHANNELS]       # (B, 10)
            loss = pinball_loss(target_preds, target_actual, quantiles_t)
            loss += 0.01 * monotonicity_loss(preds)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}/{n_epochs}  loss={avg_loss:.6f}', flush=True)
    return best_loss


def predict_test_month(model, train_df, expvars_df, task_num, batch_size=128):
    """Auto-regressive prediction with weather updating.
    For each test hour, shift the 336-hr window by 1, update weather from ExpVars,
    and use previous TARGETVAR prediction for the overlapping test hours.
    """
    model.eval()
    
    # Build initial window: last SEQ_LEN hours of training
    train_vals = {}
    for z in range(1, N_ZONES + 1):
        zdf = train_df[train_df['ZONEID'] == z].sort_values('TIMESTAMP').reset_index(drop=True)
        train_vals[z] = zdf[['TARGETVAR', 'U10', 'V10', 'U100', 'V100']].values.astype(np.float32)
    
    # ExpVars data per zone
    exp_vals = {}
    for z in range(1, N_ZONES + 1):
        zdf = expvars_df[expvars_df['ZONEID'] == z].sort_values('TIMESTAMP').reset_index(drop=True)
        exp_vals[z] = zdf[['U10', 'V10', 'U100', 'V100']].values.astype(np.float32)
    
    n_test = len(exp_vals[1])  # number of test hours
    predictions = np.zeros((n_test, N_ZONES, 99), dtype=np.float32)
    
    # Build initial window: last SEQ_LEN known rows (all channels)
    # Window shape: (SEQ_LEN, 50)
    window = np.zeros((SEQ_LEN, N_CHANNELS), dtype=np.float32)
    for zi, z in enumerate(range(1, N_ZONES + 1)):
        tv = train_vals[z]
        n_train = len(tv)
        start = max(0, n_train - SEQ_LEN)
        n_avail = min(SEQ_LEN, n_train)
        # Fill the last n_avail positions of window
        window[SEQ_LEN - n_avail:, zi * 5:(zi + 1) * 5] = tv[-n_avail:]
        # If window not full, pad with first training value
        if n_avail < SEQ_LEN:
            window[:SEQ_LEN - n_avail, zi * 5:(zi + 1) * 5] = tv[0]
    
    for t in range(n_test):
        # Predict from the current window (which contains known data only)
        # For the first test hour, window holds last 336 training hours (no weather hint)
        # For subsequent hours, window includes previous test hours with ExpVars weather
        x_tensor = torch.from_numpy(window[np.newaxis, :, :]).to(DEVICE)  # (1, 336, 50)
        with torch.no_grad():
            pred = model(x_tensor)  # (1, 50, 99)
        
        # Extract TARGETVAR predictions (first column of each zone)
        for zi, z in enumerate(range(1, N_ZONES + 1)):
            predictions[t, zi] = pred[0, zi * 5, :].cpu().numpy()  # TARGETVAR quantiles
        
        # Shift window: drop first row, make room for the new test hour
        window = np.roll(window, -1, axis=0)
        # Populate new last row: TARGETVAR from prediction, weather from ExpVars
        for zi, z in enumerate(range(1, N_ZONES + 1)):
            window[-1, zi * 5] = predictions[t, zi, 49]  # median quantile as TARGETVAR
            we = exp_vals[z][t]
            window[-1, zi * 5 + 1:(zi + 1) * 5] = we  # U10, V10, U100, V100 from ExpVars
    
    return predictions


def pinball_score(actual, preds, quantiles):
    scores = np.zeros(len(quantiles))
    for qi, q in enumerate(quantiles):
        e = actual - preds[:, qi]
        scores[qi] = np.mean(np.maximum(q * e, (q - 1) * e))
    return np.mean(scores)


def evaluate_task(task_num, model, all_task_data, all_expvars):
    """Evaluate iTransformer on one task."""
    train_df = all_task_data[task_num]
    expvars = all_expvars[task_num]
    bench = pd.read_csv(os.path.join(GEFCOM_DIR, f'Task {task_num}', f'benchmark{task_num}_W.csv'))
    bench['TIMESTAMP'] = pd.to_datetime(bench['TIMESTAMP'], format='%Y%m%d %H:%M')
    gt_df = all_task_data[task_num + 1]
    gt_merged = bench[['ZONEID', 'TIMESTAMP']].merge(
        gt_df[['ZONEID', 'TIMESTAMP', 'TARGETVAR']], on=['ZONEID', 'TIMESTAMP'], how='inner'
    )
    
    predictions = predict_test_month(model, train_df, expvars, task_num)
    
    zone_pinballs = []
    for z in range(1, N_ZONES + 1):
        zgt = gt_merged[gt_merged['ZONEID'] == z].sort_values('TIMESTAMP')
        actuals = zgt['TARGETVAR'].values
        preds = predictions[:, z - 1, :]
        if len(actuals) > len(preds):
            actuals = actuals[:len(preds)]
        elif len(preds) > len(actuals):
            preds = preds[:len(actuals)]
        
        # Monotonicity correction
        preds = np.maximum.accumulate(preds, axis=1)
        preds = np.clip(preds, 0.0, 1.0)
        
        pb = pinball_score(actuals, preds, QUANTILES)
        zone_pinballs.append(pb)
    
    return np.mean(zone_pinballs)


def main():
    import warnings
    warnings.filterwarnings('ignore')
    
    model_dir = r'C:\Projects\raghavan\vayumithra_research\results\models\itransformer_pure'
    os.makedirs(model_dir, exist_ok=True)
    
    print('Loading all task data...', flush=True)
    all_task_data = {}
    for tn in range(1, 16):
        t1 = time.time()
        all_task_data[tn] = load_task(tn)
        df = all_task_data[tn]
        print(f'  Task {tn:>2}: {len(df):>6} rows  [{time.time() - t1:.0f}s]', flush=True)
    
    print('Loading ExpVars...', flush=True)
    all_expvars = {}
    for tn in range(1, 13):
        all_expvars[tn] = load_expvars(tn)
    print('Done loading data.', flush=True)
    
    results = {}
    for task_num in range(1, 13):
        t0 = time.time()
        print(f'\nTask {task_num}:', flush=True)
        
        # Build model
        model = iTransformer_GEFCom().to(DEVICE)
        
        # Build training data
        X_train, y_train = build_dataset(all_task_data[task_num])
        n_train = len(X_train)
        print(f'  Training samples: {n_train}', flush=True)
        
        if n_train < 100:
            print(f'  SKIP: insufficient training data', flush=True)
            results[task_num] = float('nan')
            continue
        
        # Check if model exists
        model_path = os.path.join(model_dir, f'itransformer_task{task_num}.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f'  Loaded saved model', flush=True)
        else:
            loss = train_model(model, X_train, y_train)
            torch.save(model.state_dict(), model_path)
            print(f'  Trained: loss={loss:.6f}', flush=True)
        
        pb = evaluate_task(task_num, model, all_task_data, all_expvars)
        elapsed = time.time() - t0
        results[task_num] = pb
        print(f'  Pinball={pb:.5f}  [{elapsed:.0f}s]', flush=True)
        
        # Free GPU memory
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    print('\n' + '=' * 60)
    print('Pure iTransformer - Performance for all 12 weeks')
    print('=' * 60)
    for wk in range(1, 13):
        print(f'Week {wk:>2}: {results[wk]:.5f}')
    avg = np.mean([results[wk] for wk in range(1, 13)])
    print(f'Average: {avg:.5f}')


if __name__ == '__main__':
    main()
