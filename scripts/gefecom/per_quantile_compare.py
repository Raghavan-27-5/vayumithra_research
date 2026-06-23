import numpy as np, openpyxl

data = np.load(r"C:\Projects\raghavan\vayumithra_research\results\gefecom_task15_results.npz")
model_preds = data["model_preds"]
bench_preds = data["bench_preds"]
actuals = data["actuals"]
quantiles = data["quantiles"]

def pinball(y, yh, q):
    e = y - yh
    return float(np.mean(np.maximum(q*e, (q-1)*e)))

per_q_model = np.array([pinball(actuals, model_preds[:, qi], q) for qi, q in enumerate(quantiles)])
per_q_bench = np.array([pinball(actuals, bench_preds[:, qi], q) for qi, q in enumerate(quantiles)])

wb = openpyxl.load_workbook(r"C:\Projects\raghavan\GEFCom2014\GEFCom2014 Data\Provisional_Leaderboard_V2.xlsx")
ws = wb["W-score-0"]
leaderboard = {}
bench_name = None
for r in range(2, ws.max_row + 1):
    name = ws.cell(r, 1).value
    if name is None: continue
    scores = [ws.cell(r, c).value for c in range(2, 14)]
    scores = [s for s in scores if isinstance(s, (int, float))]
    if not scores: continue
    if "Benchmark" in str(name):
        bench_name = name
    leaderboard[name] = np.mean(scores)

bench_lb = leaderboard.pop(bench_name, None)
sorted_lb = sorted(leaderboard.items(), key=lambda x: x[1])

print("=" * 90)
print("GEFCom2014 Task 15 — Per-Quantile Pinball: Model vs Benchmark")
print("=" * 90)
print("  Quantile |   Model     |   Bench     |  M/B Ratio  |  LB-1 Est")
print("  " + "-" * 65)
for q_idx in list(range(0, 99, 10)) + [49]:
    q = quantiles[q_idx]
    mp = per_q_model[q_idx]
    bp = per_q_bench[q_idx]
    ratio = mp / bp if bp > 0 else float("inf")
    lb_est = 0.037 * (bp / np.mean(per_q_bench))
    print(f"  {q:8.2f} | {mp:10.6f} | {bp:10.6f} | {ratio:9.2f}x | {lb_est:9.6f}")

avg_m = np.mean(per_q_model)
avg_b = np.mean(per_q_bench)
print(f"  {'AVG':>8} | {avg_m:10.6f} | {avg_b:10.6f} | {avg_m/avg_b:9.2f}x |")

print()
print("=" * 90)
print("GEFCom2014 Leaderboard (Wind) — Avg Pinball per Task (Tasks 1-12)")
print("=" * 90)
print("  Rank | Team                       | Avg Pinball  | Ratio")
print("  " + "-" * 70)
for rank, (name, avg) in enumerate(sorted_lb, 1):
    ratio = avg / bench_lb if bench_lb else 1
    print(f"  {rank:4d} | {name:<27} | {avg:11.6f} | {ratio:5.2f}x bench")
print(f"     - | {'Benchmark':<27} | {bench_lb:11.6f} |  1.00x")
print()

print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"  Our Task 15 avg pinball:  {avg_m:.4f}")
print(f"  Benchmark Task 15:        {avg_b:.4f}")
print(f"  Our ratio to bench:       {avg_m/avg_b:.2f}x")
print(f"  kPower ratio to bench:    0.46x (tasks 1-12)")
print(f"  Gap to winning level:     {(avg_m/avg_b)/0.46:.1f}x worse than kPower")
print()
print("Per-quantile analysis:")
for label, q_idx in [("P10 (q=0.10)", 9), ("P25 (q=0.25)", 24), 
                       ("P50 (q=0.50)", 49), ("P75 (q=0.75)", 74),
                       ("P90 (q=0.90)", 89)]:
    mp = per_q_model[q_idx]
    bp = per_q_bench[q_idx]
    print(f"  {label}: Model={mp:.4f}  Bench={bp:.4f}  Ratio={mp/bp:.2f}x")
