import zipfile, os, pandas as pd, io
base = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
t = 1
zpath = os.path.join(base, 'Task ' + str(t), 'Task' + str(t) + '_W_Zone1_10.zip')
zf = zipfile.ZipFile(zpath)
names = zf.namelist()[1:]
for fname in names:
    df = pd.read_csv(io.BytesIO(zf.read(fname)))
    # Extract zone number from filename like Task1_W_Zone1.csv
    zone = fname.split('Zone')[1].split('.')[0]
    print('Zone ' + zone + ': ' + str(df.shape))
print()
# Benchmark15 details
bench = pd.read_csv(r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind\Task 15\benchmark15_W.csv')
qcols = [c for c in bench.columns if c not in ('ZONEID','TIMESTAMP')]
print('benchmark15 quantile cols: ' + str(qcols[:5]))
print('benchmark15 shape: ' + str(bench.shape))
print('benchmark15 ZONEIDs: ' + str(sorted(bench['ZONEID'].unique())))
print('benchmark15 TS range: ' + str(bench['TIMESTAMP'].min()) + ' to ' + str(bench['TIMESTAMP'].max()))
