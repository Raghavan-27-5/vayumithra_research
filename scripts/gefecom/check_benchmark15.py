import pandas as pd
bench = pd.read_csv(r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind\Task 15\benchmark15_W.csv')
print('Columns:', list(bench.columns))
print('Shape:', bench.shape)
# Check quantile column names format
q_cols = [c for c in bench.columns if c not in ('ZONEID','TIMESTAMP')]
print(f'First 5 quantile column names: {q_cols[:5]}')
print(f'Col 0.01 repr: {repr(q_cols[0])}')
print(f'Col 0.10 repr: {repr(q_cols[9])}')
# Check all unique zoneids
print(f'ZONEID unique: {sorted(bench["ZONEID"].unique())}')
# Check timestamp format
print(f'TIMESTAMP sample: {bench["TIMESTAMP"].iloc[0]}')
print(f'TIMESTAMP range: {bench["TIMESTAMP"].min()} to {bench["TIMESTAMP"].max()}')
print(f'Rows per zone: {bench.groupby(\"ZONEID\").size()}')
