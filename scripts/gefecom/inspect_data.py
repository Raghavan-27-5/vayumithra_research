import pandas as pd
df = pd.read_csv(r"C:\Projects\raghavan\iTransformer\dataset\vayumithra\vayumithra_10st_uv.csv")
print("Shape:", df.shape)
print("Cols:", list(df.columns[:12]), "...")
print("Date range:", df.iloc[:,0].min(), "to", df.iloc[:,0].max())
print("Numeric cols:", len([c for c in df.columns if c != 'date']))
print("All column names:")
for i, c in enumerate(df.columns):
    print(f"  [{i}] {c}")
