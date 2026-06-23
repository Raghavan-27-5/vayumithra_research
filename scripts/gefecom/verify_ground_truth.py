import pandas as pd, numpy as np, zipfile, io, os

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'

def load_task(task_num):
    zd = os.path.join(GEFCOM_DIR, f'Task {task_num}')
    zf = zipfile.ZipFile(os.path.join(zd, f'Task{task_num}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    return pd.concat(frames, ignore_index=True)

# Check all 15 tasks
for tn in range(1, 16):
    df = load_task(tn)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d %H:%M')
    z1 = df[df['ZONEID']==1]
    print(f'Task {tn:2d} Z1: {len(z1):5d} rows  {z1.TIMESTAMP.min().date()} to {z1.TIMESTAMP.max().date()}')

print()
# Verify Task N benchmark timestamps exist in Task N+1 data
for tn in range(1, 15):
    bench = pd.read_csv(os.path.join(GEFCOM_DIR, f'Task {tn}', f'benchmark{tn}_W.csv'))
    bench['TIMESTAMP'] = pd.to_datetime(bench['TIMESTAMP'], format='%Y%m%d %H:%M')
    
    tn1 = load_task(tn + 1)
    tn1['TIMESTAMP'] = pd.to_datetime(tn1['TIMESTAMP'], format='%Y%m%d %H:%M')
    
    merged = bench[['ZONEID','TIMESTAMP']].merge(tn1, on=['ZONEID','TIMESTAMP'], how='inner')
    print(f'Bench {tn:2d} x Task {tn+1:2d}: {len(merged):5d} rows  (expected {len(bench):5d})  has_TARGETVAR={"TARGETVAR" in merged.columns}')

# Also check solution for Task 15
sol = pd.read_csv(os.path.join(GEFCOM_DIR, 'Solution to Task 15', 'solution15_W.csv'))
sol['TIMESTAMP'] = pd.to_datetime(sol['TIMESTAMP'], format='%Y%m%d %H:%M')
print(f'\nSolution 15: {len(sol)} rows, {sol.TIMESTAMP.min()} to {sol.TIMESTAMP.max()}')
