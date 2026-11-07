"""Run all M1 variants (Tier 0 floor + full method + 3 Tier 1 ablations,
no events) at both fleet sizes agreed for M1/M2: (4 UGV, 2 UAV) and
(6 UGV, 3 UAV). NOT executed yet -- staged and ready, per instruction.

Usage once approved:
    python3 run_m1.py
"""
from graph_layouts import make_ladder
from engine import Sim, VARIANTS

HORIZON_S = 3600 * 2   # 2 simulated hours, placeholder; revisit once first results are in
FLEET_SIZES = [(4, 2), (6, 3)]
SEEDS = [0, 1, 2]       # multiple seeds per condition, per Sec "Statistical protocol"
T_HOPS = 20   # 10x10 grid's worst priority-pair shortest path is 15 hops; 20 covers it with margin


def run_all():
    ladder = make_ladder()
    m1 = ladder[0]
    results = []
    for n_ugv, n_uav in FLEET_SIZES:
        for variant_key, variant in VARIANTS.items():
            for seed in SEEDS:
                sim = Sim(m1, n_ugv=n_ugv, n_uav=n_uav, variant=variant, seed=seed,
                          t_hops=T_HOPS)
                log = sim.run(HORIZON_S)
                final = log[-1] if log else {}
                results.append({
                    "map": "M1",
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
