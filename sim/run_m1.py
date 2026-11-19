"""M1 (10x10 grid), all variants, fleets 4/1 5/1 4/2 5/2 6/3, 10 seeds, 2 h horizon.
Final configuration: covering walk set, 500 m recovery radius, nearest-UGV baseline stops once chosen.
Usage: python3 run_m1.py        -> results_m1_cover.csv (idleness, airtime_s per run)"""
from runner import run_map

HORIZON_S = 3600 * 2
FLEET_SIZES = [(4, 1), (5, 1), (4, 2), (5, 2), (6, 3)]
SEEDS = list(range(10))
T_SECONDS = 400
WALKS_PER_PAIR = 500   # only used by the old walk set (WALKSET=present)

if __name__ == "__main__":
    run_map(0, "M1", FLEET_SIZES, SEEDS, HORIZON_S, T_SECONDS, WALKS_PER_PAIR)
