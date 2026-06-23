import pandas as pd, numpy as np
sol = pd.read_csv(r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind\Solution to Task 15\solution15_W.csv')
print('Solution NaN:', sol.isna().sum().sum())
print('Target NaN:', sol['TARGETVAR'].isna().sum())
print('Zones:', sol['ZONEID'].nunique(), 'Rows:', len(sol))

bench = pd.read_csv(r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind\Task 15\benchmark15_W.csv')
print('Bench NaN:', bench.isna().sum().sum())
print('Bench rows:', len(bench))

# Check quantile columns look right
q_cols = [str(round(i*0.01, 2)) for i in range(1, 100)]
missing = [c for c in q_cols if c not in bench.columns]
if missing:
    print('Missing columns:', missing[:10])
else:
    print('All quantile columns found')
    print('Sample cols:', list(bench.columns[:15]))
