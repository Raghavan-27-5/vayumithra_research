"""
Double-verify GEFCom2014 Wind benchmark metrics from source data.
Checks: leaderboard, benchmark CSVs, solution, pinball computation.
"""
import os, zipfile, io, numpy as np, pandas as pd

GEFCOM_DIR = r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\GEFCom2014-W_V2\Wind'
QUANTILES = [round(i*0.01, 2) for i in range(1, 100)]

# ===== VERIFY 1: Leaderboard W-score-0 benchmark per week =====
print("=" * 60)
print("VERIFY 1: W-score-0 benchmark per-week scores")
print("=" * 60)
xl = pd.ExcelFile(r'C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\Provisional_Leaderboard_V2.xlsx')
ws0 = pd.read_excel(xl, 'W-score-0', header=None)
# Find benchmark row in W-score-0
for i in range(len(ws0)):
    row = ws0.iloc[i].astype(str).tolist()
    if 'BENCHMARK' in str(row[0]).upper():
        print(f"Row {i}: {ws0.iloc[i].tolist()}")
        break

print()

# ===== VERIFY 2: Task 15 Benchmark pinball =====
print("=" * 60)
print("VERIFY 2: Task 15 benchmark pinball (against solution)")
print("=" * 60)
bench15 = pd.read_csv(os.path.join(GEFCOM_DIR, 'Task 15', 'benchmark15_W.csv'))
sol15 = pd.read_csv(os.path.join(GEFCOM_DIR, 'Solution to Task 15', 'solution15_W.csv'))
bench15['TIMESTAMP'] = pd.to_datetime(bench15['TIMESTAMP'], format='%Y%m%d %H:%M')
sol15['TIMESTAMP'] = pd.to_datetime(sol15['TIMESTAMP'], format='%Y%m%d %H:%M')

# Check alignment
merged = sol15.merge(bench15, on=['ZONEID','TIMESTAMP'], suffixes=('_sol','_bench'))
y_true = merged['TARGETVAR'].values.astype(np.float32)

for qi, q in enumerate(QUANTILES[:10]):
    qcol = str(q)
    y_pred = merged[qcol].values.astype(np.float32)
    valid = ~np.isnan(y_true)
    e = y_true[valid] - y_pred[valid]
    pb = np.mean(np.maximum(q * e, (q - 1) * e))
    print(f"  q={q:.2f}: pinball={pb:.6f}")

# Compute overall
overall_pb = 0.0
for qi, q in enumerate(QUANTILES):
    qcol = str(q)
    y_pred = merged[qcol].values.astype(np.float32)
    valid = ~np.isnan(y_true)
    e = y_true[valid] - y_pred[valid]
    pb = np.mean(np.maximum(q * e, (q - 1) * e))
    overall_pb += pb
overall_pb /= 99
print(f"\n  OVERALL benchmark pinball (Task 15): {overall_pb:.10f}")

print()

# ===== VERIFY 3: Benchmark per-week scores (recomputed) =====
print("=" * 60)
print("VERIFY 3: Recompute benchmark per-week scores from CSV")
print("=" * 60)
# The weeks correspond to tasks 3-14 (tasks 1-2 had different scoring)
# Actually tasks 1-12 are the 12 weeks in W-score-0
for task_num in range(1, 13):
    sol = pd.read_csv(os.path.join(GEFCOM_DIR, f'Task {task_num}', f'benchmark{task_num}_W.csv'))
    # Load training data to get ground truth (tasks 1-12 have solution built-in?)
    # Actually only Task 15 has a separate solution file. Other tasks have no ground truth
    # The benchmark scores in W-score-0 are official GEFCom results
    print(f"  Task {task_num}: {len(sol)} rows, cols={len(sol.columns)}")
print("  (Can't recompute - no solution files for tasks 1-14)")

print()

# ===== VERIFY 4: Compare kPower scores per week =====
print("=" * 60)
print("VERIFY 4: kPower scores vs benchmark per week")
print("=" * 60)
# Read from W-score-0
for i in range(len(ws0)):
    row = ws0.iloc[i].astype(str).tolist()
    if 'KPOWER' in str(row[0]).upper():
        print(f"kPower row {i}: {ws0.iloc[i].tolist()}")
    if 'BENCHMARK' in str(row[0]).upper():
        print(f"Benchmark row {i}: {ws0.iloc[i].tolist()}")
    if 'WEEK' in str(row[0]).upper() or 'SCORE' in str(row[0]).upper():
        pass  # header

# Also check W-score-3 for final ratings
ws3 = pd.read_excel(xl, 'W-score-3', header=None)
print()
print("=" * 60)
print("VERIFY 5: W-score-3 final ratings")
print("=" * 60)
for i in range(len(ws3)):
    row = ws3.iloc[i].astype(str).tolist()
    if any(kw in str(row[0]).upper() for kw in ['KPOWER', 'BENCHMARK', 'DMLAB', 'RATING']):
        print(f"  Row {i}: {ws3.iloc[i].tolist()}")
# Print all rows
for i in range(min(len(ws3), 20)):
    print(f"  W3 Row {i}: {ws3.iloc[i].tolist()}")
