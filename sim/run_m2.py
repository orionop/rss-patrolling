"""Run all M2 variants (same 5 as M1: Tier 0 floor + full + 3 Tier 1
ablations, no events) at the same two fleet sizes as M1: (4,2) and
(6,3). M2 reuses M1's nodes/priorities/ghosts, edges thinned to
near-tree -- this is where we expect ghost-credit and fixed-vs-radius
launch gaps to widen vs M1's dense grid. NOT executed yet -- staged.

Usage once approved:
    python3 run_m2.py
"""
from graph_layouts import make_ladder
from engine import Sim, VARIANTS

HORIZON_S = 3600 * 2   # same horizon as M1, placeholder
FLEET_SIZES = [(4, 2), (6, 3)]
SEEDS = [0, 1, 2]
T_HOPS = 40   # 10x10 grid's thinning pushes the worst priority-pair shortest
              # path to 34 hops; 40 covers it with margin (same reasoning as
              # before -- too small a cutoff makes far priority/ghost pairs
              # structurally unreachable regardless of variant).


def run_all():
    ladder = make_ladder()
    m2 = ladder[1]
    results = []
    for n_ugv, n_uav in FLEET_SIZES:
        for variant_key, variant in VARIANTS.items():
            for seed in SEEDS:
                sim = Sim(m2, n_ugv=n_ugv, n_uav=n_uav, variant=variant, seed=seed,
                          t_hops=T_HOPS)
                log = sim.run(HORIZON_S)
                final = log[-1] if log else {}
                results.append({
                    "map": "M2",
                    "fleet": f"{n_ugv}-{n_uav}",
                    "variant": variant_key,
                    "seed": seed,
                    "mean_idleness": final.get("mean_idleness"),
                    "ghost_mean_idleness": final.get("ghost_mean_idleness"),
                    "ghost_max_idleness": final.get("ghost_max_idleness"),
                })
                print(results[-1])
    return results


if __name__ == "__main__":
    run_all()
