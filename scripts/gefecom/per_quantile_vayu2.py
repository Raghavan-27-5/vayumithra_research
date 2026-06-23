import json
with open(r"C:\Projects\raghavan\vayumithra_research\results\probabilistic_results.json") as f:
    r = json.load(f)

horizons = ["1","2","3","4","5","6","12","24"]
print("=" * 110)
print("VayuMithra  Per-Horizon, Per-Quantile Pinball vs Deterministic Baseline")
print("=" * 110)
print("  Horizon |   P10   |   P50   |   P90   |   P99   |   Avg   | Det P50 MAE | Ratio  |  Cov")
print("  " + "-" * 95)

det_mae = {"1": 0.1247, "24": 0.8175}
for h in horizons:
    d = r.get(h, {})
    if not d:
        print(f"  H{h:>6} | no data")
        continue
    p10 = d.get("pinball_p10", 0)
    p50 = d.get("pinball_p50", 0)
    p90 = d.get("pinball_p90", 0)
    p99 = d.get("pinball_p99", 0)
    avg = d.get("pinball_avg", 0)
    mae = d.get("p50_mae", 0)
    cov = d.get("coverage_p10_p90", 0)
    dmae = det_mae.get(h)
    ratio = f"{mae/dmae:.2f}x" if dmae else "N/A"
    dm_str = f"{dmae:.4f}" if dmae else "  N/A  "
    print(f"  H{h:>6} | {p10:7.4f} | {p50:7.4f} | {p90:7.4f} | {p99:7.4f} | {avg:7.4f} | {dm_str:>11} | {ratio:>7} | {cov:.3f}")
