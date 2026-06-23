import pandas as pd, zipfile, io, os

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'

for tn in range(1, 16):
    tdir = os.path.join(GEFCOM_DIR, f'Task {tn}')
    if not os.path.exists(tdir):
        continue
    zf = zipfile.ZipFile(os.path.join(tdir, f'Task{tn}_W_Zone1_10.zip'))
    frames = [pd.read_csv(io.BytesIO(zf.read(n))) for n in zf.namelist()[1:]]
    tdf = pd.concat(frames, ignore_index=True)
    tdf['TS'] = pd.to_datetime(tdf['TIMESTAMP'], format='%Y%m%d %H:%M')
    
    bench = pd.read_csv(os.path.join(tdir, f'benchmark{tn}_W.csv'))
    bench['TS'] = pd.to_datetime(bench['TIMESTAMP'], format='%Y%m%d %H:%M')
    
    overlap = set(tdf['TS'].unique()) & set(bench['TS'].unique())
    hasNaN = tdf['TARGETVAR'].isna().any()
    print(f'Task {tn}: train={tdf["TS"].min()}->{tdf["TS"].max()}, test={bench["TS"].min()}->{bench["TS"].max()}, overlap={len(overlap)}, hasNaN={hasNaN}')
