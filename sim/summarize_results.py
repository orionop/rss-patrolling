"""Turn results_m1.csv / results_m2.csv into a paper-ready table.

Usage:  python3 summarize_results.py            (reads results_m1.csv, results_m2.csv)
Writes: paper_table.csv  - one row per (map, fleet, variant): mean and std over seeds
        (std is the sample std, ddof=1; blank if only one seed)
"""
import csv, glob, statistics as st
from collections import defaultdict

ORDER = ["tier0_ugv_only", "no_ghost_credit", "fixed_launch_nodes",
         "nearest_ugv_recovery", "full"]
METRICS = [("mean_idleness", "ground_mean_idleness"),
           ("ghost_mean_idleness", "ghost_mean_idleness"),
           ("ghost_max_idleness", "ghost_peak_idleness")]

rows = defaultdict(list)
for f in sorted(glob.glob("results_m*.csv")):
    for r in csv.DictReader(open(f)):
        rows[(r["map"], r["fleet"], r["variant"])].append(r)

out = []
for (m, fl, v), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], ORDER.index(kv[0][2]))):
    row = {"map": m, "fleet": fl, "variant": v, "n_seeds": len(rs)}
    for src, name in METRICS:
        xs = [float(r[src]) for r in rs]
        row[name + "_mean"] = round(st.mean(xs), 1)
        row[name + "_std"] = round(st.stdev(xs), 1) if len(xs) > 1 else ""
    out.append(row)

with open("paper_table.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader(); w.writerows(out)
print(f"wrote paper_table.csv ({len(out)} rows)")
for r in out: print(r)
