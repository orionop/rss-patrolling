"""Shared runner for run_m1.py / run_m2.py.

- Builds only the map it needs (no OSM download).
- Enumerates the valid-walk set once per process, not once per run.
- Runs the independent (fleet, variant, seed) jobs in parallel on all CPU cores.
- Prints a progress line per finished run and writes a CSV at the end.
"""
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

from graph_layouts import make_ladder
from engine import Sim, VARIANTS, Weights, enumerate_valid_walks, build_cover_walks

# Walk set (final-skeleton Sec 7): "present" = K shortest-by-hops + detours (old),
# "cover" = covering walk set + starvation claims. Select with env WALKSET.
WALKSET = os.environ.get("WALKSET", "present")
COVER_K = int(os.environ.get("COVER_K", "-1"))       # fastest walks per pair; -1 = rule K = |V_g|
CLAIM_CAP = int(os.environ.get("CLAIM_CAP", "-1"))   # N_c; -1 = rule ceil(n_UGV |U| / |V_g|)


# Starvation guard (extension to TPBP, applied to every variant): when a ground node's
# idleness exceeds STARVE_FACTOR * T, the UGV picks among eligible walks that cover it.
# Set to None for the pure TPBP reward.
STARVE_FACTOR = 4

# Same idea for ghost nodes: when a ghost's idleness exceeds GHOST_STARVE_FACTOR * T, UGVs
# prefer walks that pass within r_max of it (so a UAV can be launched). Off (None) by default:
# in testing it changed ghost idleness by < 4 %. Try 2 or 3 to experiment.
GHOST_STARVE_FACTOR = None

# Ghost idleness cap (extension to Sec 3.2): if an unclaimed ghost's idleness exceeds
# GHOST_CAP_FACTOR * T, a docked UAV may launch to it from ANY distance (r_max waived).
# This is what keeps ghosts out of the dark-red zone: peak ghost idleness lands about
# 100 s above the cap. 3 -> 1200 s at T = 400 s. None = pure Sec 3.2 launch rule.
# Only affects variants that launch by proximity (not UGV-only, not fixed launch nodes).
GHOST_CAP_FACTOR = 3

# Overlap discount (extension to TPBP's reward): in Term 1 the idleness of a node that lies on another
# UGV's remaining walk counts with weight (1 - DISCOUNT_OTHERS). 0 = pure TPBP reward.
DISCOUNT_OTHERS = 0.5
import os
GHOST_SEC = os.environ.get('GHOST_SEC','1')=='1'


# Weight presets for c5-c8 (c1-c4 are fixed). Select with the environment variable
# WEIGHT_SET, or edit WEIGHT_SET below.
#   design_rule : satisfies Remarks 2 and 5 of the formulation (default)
#   tuned       : best of a free sweep on tuning seeds 100-102; violates the design rule
import os
WEIGHT_PRESETS = {
    "design_rule": dict(c5=13.0, c6=3.0, c7=9.0, c8=5.0),
    "tuned": dict(c5=3.0, c6=1.5, c7=2.0, c8=5.0),
    # all road weights 1, ghost-credit weight raised: used for the paper tables
    "paper": dict(c1=1.0, c2=1.0, c3=1.0, c4=1.0, c5=6.0, c6=1.0, c7=1.0, c8=1.0),
}
WEIGHT_SET = os.environ.get("WEIGHT_SET", "paper")

# Minimum ghost idleness (s) before a UAV may launch to it. Not in Sec 3.2 as written; see README.
LAUNCH_I_MIN = 20.0


def make_weights(t_seconds):
    return Weights(T=t_seconds, discount_others=DISCOUNT_OTHERS, ghost_units_seconds=GHOST_SEC,
                   claim_cap=CLAIM_CAP if WALKSET == "cover" else None,
                   gate_idleness=float(os.environ["GATE_FACTOR"]) * t_seconds if os.environ.get("GATE_FACTOR") else None,
                   starve_idleness=None if STARVE_FACTOR is None else STARVE_FACTOR * t_seconds,
                   ghost_starve_idleness=None if GHOST_STARVE_FACTOR is None
                   else GHOST_STARVE_FACTOR * t_seconds,
                   ghost_cap_idleness=None if GHOST_CAP_FACTOR is None
                   else GHOST_CAP_FACTOR * t_seconds)


@lru_cache(maxsize=None)
def _layout(map_idx):
    return make_ladder(include_m3=False)[map_idx]


@lru_cache(maxsize=None)
def _walks(map_idx, t_seconds, walks_per_pair):
    lay = _layout(map_idx)
    if WALKSET == "cover":
        walks, st = build_cover_walks(lay.G, lay.priorities, t_seconds, lay.pos, Weights().v_g,
                                      K=COVER_K, ghosts=lay.ghosts, r_max=Weights().r_max)
        return walks, frozenset(st["gated"])
    return enumerate_valid_walks(lay.G, lay.priorities, 0, walks_per_pair,
                                 t_seconds=t_seconds, pos=lay.pos, v_g=Weights().v_g), frozenset()


def summarize(log):
    """Reduce a run's per-step metrics log to time-averaged numbers.

    mean_idleness / ghost_mean_idleness: average over every simulated step
    of the network-mean idleness (road nodes / ghost nodes).
    ghost_max_idleness: the worst single ghost idleness reached at any step
    of the run (peak, not an average).
    The *_final columns are the old end-of-run snapshot, kept for reference.
    """
    if not log:
        return {}
    n = len(log)
    return {
        "mean_idleness": sum(r["mean_idleness"] for r in log) / n,
        "ghost_mean_idleness": sum(r["ghost_mean_idleness"] for r in log) / n,
        "ghost_max_idleness": max(r["ghost_max_idleness"] for r in log),
        "road_max_idleness": max(r.get("road_max_idleness", 0.0) for r in log),
        "priority_max_idleness": max(r.get("priority_max_idleness", 0.0) for r in log),
        "priority_mean_idleness": sum(r.get("priority_mean_idleness", 0.0) for r in log) / n,
        "mean_idleness_final": log[-1]["mean_idleness"],
        "ghost_mean_idleness_final": log[-1]["ghost_mean_idleness"],
    }


def _job(args):
    map_idx, map_name, n_ugv, n_uav, variant_key, seed, horizon, t_hops, k = args
    layout = _layout(map_idx)
    sim = Sim(layout, n_ugv=n_ugv, n_uav=n_uav, variant=VARIANTS[variant_key],
              seed=seed, weights=make_weights(t_hops), valid_walks=_walks(map_idx, t_hops, k)[0],
              gated_walks=_walks(map_idx, t_hops, k)[1])
    log = sim.run(horizon)
    row = {
        "map": map_name,
        "fleet": f"{n_ugv}-{n_uav}",
        "variant": variant_key,
        "seed": seed,
    }
    row.update({k: round(v, 2) for k, v in summarize(log).items()})
    return row


def run_map(map_idx, map_name, fleet_sizes, seeds, horizon, t_hops,
            walks_per_pair, workers=None):
    jobs = [(map_idx, map_name, nu, na, vk, sd, horizon, t_hops, walks_per_pair)
            for (nu, na) in fleet_sizes for vk in VARIANTS for sd in seeds]
    workers = workers or os.cpu_count() or 1
    lay = _layout(map_idx)
    print(f"{map_name}: {lay.G.number_of_nodes()} nodes, {len(jobs)} runs, "
          f"{workers} workers, T={t_hops} s, walks_per_pair={walks_per_pair}", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {r}", flush=True)
    order = {k: i for i, k in enumerate(VARIANTS)}
    results.sort(key=lambda r: (r["fleet"], order[r["variant"]], r["seed"]))
    out = f"results_{map_name.lower()}_{WALKSET}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"done in {time.time() - t0:.0f}s; wrote {out}", flush=True)
    return results
