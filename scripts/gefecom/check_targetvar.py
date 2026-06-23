import zipfile, io, pandas as pd
zf = zipfile.ZipFile(r"C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind\Task 1\Task1_W_Zone1_10.zip")
for n in zf.namelist():
    if n.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(zf.read(n)))
        print(f"File: {n}")
        print(f"TARGETVAR range: {df['TARGETVAR'].min():.4f} - {df['TARGETVAR'].max():.4f}")
        print(f"mean={df['TARGETVAR'].mean():.4f} std={df['TARGETVAR'].std():.4f}")
        print(f"First 3 rows:")
        print(df[["TIMESTAMP","TARGETVAR"]].head(3).to_string(index=False))
        break

sol = pd.read_csv(r"C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind\Solution to Task 15\solution15_W.csv")
print()
print("Solution TARGETVAR:")
print(f"  range: {sol['TARGETVAR'].min():.4f} - {sol['TARGETVAR'].max():.4f}")
print(f"  mean: {sol['TARGETVAR'].mean():.4f}  std: {sol['TARGETVAR'].std():.4f}")
print(f"  rows: {len(sol)}")
print(sol.head(3).to_string(index=False))

# Also check benchmark: are benchmark values quantiles (0-1 range) or power values?
bench = pd.read_csv(r"C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind\Task 15\benchmark15_W.csv")
print()
print(f"Benchmark columns: {bench.columns.tolist()[:6]}")
print(f"Benchmark value range: {bench.iloc[:, 3:].values.min():.4f} - {bench.iloc[:, 3:].values.max():.4f}")
print(bench.head(3).to_string(index=False))
