# Try with xlrd engine if available, or just show error
import pandas as pd
try:
    df = pd.read_excel(r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\Provisional_Leaderboard_V2.xlsx', engine='xlrd')
    print('Success with xlrd')
    print(df.head(50).to_string())
except Exception as e1:
    print(f'xlrd failed: {e1}')
    try:
        df = pd.read_excel(r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\Provisional_Leaderboard_V2.xlsx', engine='calamine')
        print('Success with calamine')
        print(df.head(50).to_string())
    except Exception as e2:
        print(f'calamine failed: {e2}')
        # Just try reading as a binary file
        with open(r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\Provisional_Leaderboard_V2.xlsx', 'rb') as f:
            header = f.read(200)
            print(f'Binary header: {header[:100]}')
