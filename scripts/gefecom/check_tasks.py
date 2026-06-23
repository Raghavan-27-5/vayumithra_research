import zipfile, os, pandas as pd, io
base = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
for t in range(1, 16):
    zpath = os.path.join(base, 'Task ' + str(t), 'Task' + str(t) + '_W_Zone1_10.zip')
    zf = zipfile.ZipFile(zpath)
    names = zf.namelist()[1:]
    # Read Zone1 CSV
    fname = [n for n in names if 'Zone1.csv' in n][0]
    df = pd.read_csv(io.BytesIO(zf.read(fname)))
    print(f'Task {t:2d}: {df["TIMESTAMP"].min()} to {df["TIMESTAMP"].max()}  ({len(df)} rows)')
