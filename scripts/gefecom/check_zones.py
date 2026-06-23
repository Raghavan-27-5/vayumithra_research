import zipfile, os, pandas as pd, io
base = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
t = 1
zpath = os.path.join(base, f'Task {t}', f'Task{t}_W_Zone1_10.zip')
zf = zipfile.ZipFile(zpath)
names = zf.namelist()[1:]
for fname in names:
    df = pd.read_csv(io.BytesIO(zf.read(fname)))
    zone = fname.split('Zone')[1].split('.')[0]
    print(f'Zone {zone}: {df.shape}')
