"""Regenerate the results tables (markdown) from results/*.csv.
Cells: combined idleness (road mean + air-only mean) / UAV airtime (s, flying distance / 12 m/s).
Margins: paired t over seeds, full minus closest baseline (negative favours full)."""
import os, sys, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
V = ["full", "no_ghost_credit", "fixed_launch_nodes", "nearest_ugv_recovery"]
NM = {"no_ghost_credit": "no air-only credit", "fixed_launch_nodes": "fixed launch nodes", "nearest_ugv_recovery": "nearest-UGV"}
tt = lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12)

def emit(title, f, fleets):
    d = pd.read_csv(os.path.join(R, f)); d["comb"] = d.mean_idleness + d.ghost_mean_idleness; ns = d.seed.nunique()
    print(f"**{title}, {ns} seeds**\n\n| Fleet | Full | No air-only credit | Fixed launch | Nearest-UGV rec. | UGV-only |\n|---|---|---|---|---|---|")
    rows = []
    for fl in fleets:
        x = d[d.fleet == fl]; g = x.groupby("variant")[["comb", "airtime_s"]].mean(); b = g.comb[V].min()
        print(f"| {fl.replace('-', ' / ')} | " + " | ".join((f"**{g.comb[v]:.1f}** / {g.airtime_s[v]:.0f}" if abs(g.comb[v] - b) < 1e-9 else f"{g.comb[v]:.1f} / {g.airtime_s[v]:.0f}") for v in V) + f" | {g.comb['tier0_ugv_only']:.1f} / 0 |")
        P = x.pivot(index="seed", columns="variant", values="comb"); A = x.pivot(index="seed", columns="variant", values="airtime_s")
        mn = P[V].mean(); bs = min(V[1:], key=lambda v: mn[v]); di = (P["full"] - P[bs]).values; da = (A["full"] - A[bs]).values
        rows.append((fl, NM[bs], di.mean(), tt(di), int((di < 0).sum()), da.mean(), tt(da)))
    print("\n| Fleet | Closest baseline | Idleness diff (s) | t | Full lower | Airtime diff (s) | t |\n|---|---|---|---|---|---|---|")
    for r in rows: print(f"| {r[0].replace('-', ' / ')} | {r[1]} | {r[2]:+.1f} | {r[3]:.2f} | {r[4]}/{ns} | {r[5]:+.0f} | {r[6]:.2f} |")
    print()

if __name__ == "__main__":
    emit("M1", "m1_10seeds.csv", ["4-1", "5-1", "4-2", "5-2", "6-3"])
    emit("M2", "m2_50seeds.csv", ["4-1", "5-1", "4-2", "5-2", "6-3"])
    emit("M3 (T = 421 s)", "m3_T421_10seeds.csv", ["8-4", "10-5", "10-7"])
