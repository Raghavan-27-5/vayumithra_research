import pandas as pd, zipfile, io, os

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'

tn = 8
tdir = os.path.join(GEFCOM_DIR, f'Task {tn}')
zf = zipfile.ZipFile(os.path.join(tdir, f'Task{tn}_W_Zone1_10.zip'))
frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
tdf = pd.concat(frames, ignore_index=True)
tdf['TS'] = pd.to_datetime(tdf['TIMESTAMP'], format='%Y%m%d %H:%M')

print('Total rows:', len(tdf))
print('NaN TARGETVAR rows:', tdf['TARGETVAR'].isna().sum())

# Get NaN rows
nan_df = tdf[tdf['TARGETVAR'].isna()]
print('NaN timestamp range:', nan_df['TS'].min(), '-', nan_df['TS'].max())
print('NaN zones:', nan_df['ZONEID'].nunique())
print('NaN unique timestamps:', nan_df['TS'].nunique())

# Check benchmark
bench = pd.read_csv(os.path.join(tdir, f'benchmark{tn}_W.csv'))
bench['TS'] = pd.to_datetime(bench['TIMESTAMP'], format='%Y%m%d %H:%M')
print()
print('Benchmark TS range:', bench['TS'].min(), '-', bench['TS'].max())
print('Benchmark unique TS:', bench['TS'].nunique())

# Check how many NaN timestamps match benchmark
nan_ts = set(nan_df['TS'].unique())
bench_ts = set(bench['TS'].unique())
common = nan_ts & bench_ts
print('NaN-Bench overlap:', len(common))
if len(common) > 0:
    print('Sample:', list(common)[:5])
else:
    # Check if NaN rows are before benchmark
    print('NaN timestamps before benchmark:', sum(1 for t in nan_ts if t < bench['TS'].min()))
    print('NaN timestamps after benchmark:', sum(1 for t in nan_ts if t > bench['TS'].max()))
    print('NaN sample:', sorted(nan_ts)[:5])
