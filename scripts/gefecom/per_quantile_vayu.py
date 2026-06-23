import json, numpy as np

with open(r"C:\Projects\raghavan\vayumithra_research\results\probabilistic_results.json") as f:
    results = json.load(f)

horizons = ["H1","H2","H3","H4","H5","H6","H12","H24"]
print("=" * 100)
print("VayuMithra — Per-Horizon, Per-Quantile Pinball Loss (Fold 1)")
print("=" * 100)
print(f"  {'Horizon':>7} | {'P10':>10} | {'P25':>10} | {'P50':>10} | {'P75':>10} | {'P90':>10} | {'P99':>10} | {'AVG':>10}")
print("  " + "-" * 93)

for h in horizons:
    r = results.get(h, {})
    pinball = r.get("pinball_by_quantile", {})
    if pinball:
        p10 = pinball.get("P10", 0)
        p25 = pinball.get("P25", 0)
        p50 = pinball.get("P50", 0)
        p75 = pinball.get("P75", 0)
        p90 = pinball.get("P90", 0)
        p99 = pinball.get("P99", 0)
        avg = np.mean([p10, p25, p50, p75, p90, p99])
        print(f"  {h:>7} | {p10:10.4f} | {p25:10.4f} | {p50:10.4f} | {p75:10.4f} | {p90:10.4f} | {p99:10.4f} | {avg:10.4f}")

print()
print("=" * 100)
print("VayuMithra — Per-Horizon Coverage & MAE")
print("=" * 100)
print(f"  {'Horizon':>7} | {'P50 MAE':>8} | {'Coverage':>9} | {'P50 MAE (det)':>13} | {'Ratio':>7}")
print("  " + "-" * 55)
# Deterministic baseline from VAYUMITHRA_CONTEXT.md
det_mae = {"H1": 0.1247, "H2": None, "H3": None, "H4": None, "H5": None, "H6": None, "H12": None, "H24": 0.8175}
for h in horizons:
    r = results.get(h, {})
    mae = r.get("p50_mae", 0)
    cov = r.get("coverage", 0)
    dmae = det_mae.get(h, None)
    ratio = mae / dmae if dmae and mae else None
    dr = f"{ratio:.2f}x" if ratio else "N/A"
    dm = f"{dmae:.4f}" if dmae else "N/A"
    print(f"  {h:>7} | {mae:8.4f} | {cov:9.4f} | {dm:>13} | {dr:>7}")
