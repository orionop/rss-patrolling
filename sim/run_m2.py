"""Run all M2 variants (same 5 as M1: Tier 0 floor + full + 3 Tier 1
ablations, no events) at the same two fleet sizes as M1: (4 UGV, 2 UAV)
and (6 UGV, 3 UAV). M2 reuses M1's nodes/priorities/ghosts with edges
thinned to near-tree.

Usage:
    python3 run_m2.py
Reports time-averaged idleness over the whole run (not an end-of-run
snapshot). Prints one line per finished run and writes results_m2.csv.
"""
from runner import run_map

HORIZON_S = 3600 * 2   # 2 simulated hours, placeholder; revisit once first results are in
FLEET_SIZES = [(4, 2), (6, 3)]
SEEDS = list(range(10))       # multiple seeds per condition, per Sec "Statistical protocol"
T_SECONDS = 400   # shared priority time period T (s): walk travel time <= T and overshoot threshold. 400 s = longest M1 priority pair (375 s) + margin; see README
# Valid walks per ordered priority pair: the K shortest simple paths (travel time <= T_SECONDS).
# Exact TPBP enumerates ALL simple paths within T, which is tens of millions
# on this grid. Set to None for the exact (very slow) version. Report K in the paper.
WALKS_PER_PAIR = 500


def run_all():
    return run_map(1, "M2", FLEET_SIZES, SEEDS, HORIZON_S, T_SECONDS, WALKS_PER_PAIR)


if __name__ == "__main__":
    run_all()
