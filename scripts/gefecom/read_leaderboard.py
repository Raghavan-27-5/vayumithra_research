import pandas as pd
xls = pd.ExcelFile(r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\Provisional_Leaderboard_V2.xlsx')
print('Sheet names:', xls.sheet_names)
for sname in xls.sheet_names:
    print(f'\n=== Sheet: {sname} ===')
    df = pd.read_excel(xls, sname)
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print()
    print(df.head(40).to_string())
