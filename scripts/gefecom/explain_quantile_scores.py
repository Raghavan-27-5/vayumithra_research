import numpy as np

data = np.load(r"C:\Projects\raghavan\vayumithra_research\results\gefecom_task15_results.npz")
mp = data["model_preds"]
bp = data["bench_preds"]
ac = data["actuals"]
q = data["quantiles"]

def pinball(y, yh, qv):
    e = y - yh
    return float(np.mean(np.maximum(qv*e, (qv-1)*e)))

print("=" * 60)
print("GEFCom2014 Task 15: Pinball per Quantile")
print("=" * 60)
print(f"  Quantile |   Ours     |   Bench    |  Ratio")
print("  " + "-" * 42)
for qi in range(0, 99, 10):
    p = pinball(ac, mp[:, qi], q[qi])
    b = pinball(ac, bp[:, qi], q[qi])
    print(f"  {q[qi]:8.2f} | {p:10.6f} | {b:10.6f} | {p/b:7.2f}x")

p50 = pinball(ac, mp[:, 49], q[49])
b50 = pinball(ac, bp[:, 49], q[49])
print(f"  {q[49]:8.2f} | {p50:10.6f} | {b50:10.6f} | {p50/b50:7.2f}x")

avg_m = np.mean([pinball(ac, mp[:, qi], q[qi]) for qi in range(99)])
avg_b = np.mean([pinball(ac, bp[:, qi], q[qi]) for qi in range(99)])
print(f"  {'ALL':>8} | {avg_m:10.6f} | {avg_b:10.6f} | {avg_m/avg_b:7.2f}x")
print()

print("What the scores mean:")
print("  Each score = Pinball Loss averaged across ALL 99 quantiles")
print("  (q=0.01 through q=0.99), ALL hours in test period, ALL zones")
print()
print(f"  Benchmark (tasks 1-12 avg): 0.0867 -- avg(99 quantiles, 12 tasks)")
print(f"  kPower (tasks 1-12 avg):     0.0371 -- same metric, better")
print(f"  Our benchmark (Task 15):      0.0792 -- slightly different task")
print(f"  Our model (Task 15):          0.1782 -- 2.25x worse than its benchmark")
print()
print("There is NO horizon dimension in GEFCom. Unlike VayuMithra")
print("(where we predict H1, H2, ..., H24 separately), GEFCom predicts")
print("the entire test month at once. The 'horizon' varies within each")
print("task (1 to ~744 hours from training end).")
print()
print("VayuMithra vs GEFCom comparison:")
print("  VayuMithra H1:  avg pinball=0.0597 (P10+P50+P90+P99 weighted)")
print("  GEFCom kPower:  avg pinball=0.0371 (all 99 quantiles)")
print("  GEFCom is easier because it uses weather forecast inputs")
