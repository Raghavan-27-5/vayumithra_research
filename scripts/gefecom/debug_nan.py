import sys, os, numpy as np, pandas as pd, zipfile, io

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
N_ZONES, SEQ_LEN = 10, 336

def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'Task{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

all_df = pd.concat([load_task(t) for t in range(1, 16)], ignore_index=True)
all_df['TIMESTAMP'] = pd.to_datetime(all_df['TIMESTAMP'], format='%Y%m%d %H:%M')
all_df = all_df.sort_values(['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
all_df = all_df.drop_duplicates(subset=['ZONEID', 'TIMESTAMP']).reset_index(drop=True)
print(f'Total rows: {len(all_df)}')
print(f'Unique zone+ts: {all_df.groupby(["ZONEID","TIMESTAMP"]).size().max()}')

mdf_list = []
for z in range(1, N_ZONES + 1):
    zdf = all_df[all_df['ZONEID'] == z].set_index('TIMESTAMP')[['TARGETVAR','U10','V10','U100','V100']]
    mdf_list.append(zdf)
mdf = pd.concat(mdf_list, axis=1, keys=[f'Z{z}' for z in range(1, N_ZONES + 1)])
mdf = mdf.sort_index()
print(f'mdf shape: {mdf.shape}, NaN count: {mdf.isna().sum().sum()}')
if mdf.isna().sum().sum() > 0:
    nan_rows = mdf[mdf.isna().any(axis=1)]
    print(f'NaN rows: {len(nan_rows)}, range: {nan_rows.index.min()} to {nan_rows.index.max()}')

train_mdf = mdf[mdf.index < '2013-12-01']
train_data = train_mdf.values.astype(np.float32)
print(f'Train data shape: {train_data.shape}, NaN in data: {np.isnan(train_data).any()}')

target_channels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
weather_channels = [i for i in range(50) if i % 5 != 0]

nan_windows = 0
ok_windows = 0
for i in range(len(train_data) - SEQ_LEN - 1 + 1):
    xw = train_data[i:i+SEQ_LEN]
    if np.isnan(xw).any():
        nan_windows += 1
        continue
    target_idx = i + SEQ_LEN
    fw = train_data[target_idx, weather_channels]
    if np.isnan(fw).any():
        print(f'  Win {i}: weather NaN at idx={target_idx}/{len(train_data)-1}')
        nan_windows += 1
        continue
    ty = train_data[target_idx, target_channels]
    if np.isnan(ty).any():
        print(f'  Win {i}: target NaN at idx={target_idx}')
        nan_windows += 1
        continue
    ok_windows += 1

print(f'Windows: OK={ok_windows} NaN={nan_windows}')
