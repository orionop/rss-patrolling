"""M3 (Boston OSM, cached) with the NEW (covering) walk set + starvation claims.
Usage: python3 m3_cover_run.py "4-2,6-3,8-4,10-5,10-7" [seeds] [T or 'star']
T='star' uses T = T* (smallest T at which every edge lies on a valid walk).
Needs osmnx and the Overpass cache in ../sim/cache (no network)."""
import sys, os, csv, time, math, multiprocessing as mp
os.environ["WALKSET"] = "cover"
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, "..", "sim")
sys.path.insert(0, SIM); os.chdir(SIM)
from graph_layouts import make_ladder
from engine import Sim, VARIANTS, Weights, build_cover_walks
from runner import make_weights, summarize
HORIZON = 7200
LAY = make_ladder(include_m3=True)[2]
KEYS = ["full", "no_ghost_credit", "fixed_launch_nodes", "nearest_ugv_recovery", "tier0_ugv_only"]
WALKS, GATED, T = None, None, 400.0

def job(a):
    nu, na, vk, seed = a
    sim = Sim(LAY, n_ugv=nu, n_uav=na, variant=VARIANTS[vk], seed=seed,
              weights=make_weights(T), valid_walks=WALKS, gated_walks=GATED)
    row = {"map": "M3", "fleet": f"{nu}-{na}", "variant": vk, "seed": seed, "T": round(T, 2)}
    row.update({k: round(v, 2) for k, v in summarize(sim.run(HORIZON)).items()})
    row["claims"] = sim.n_claims_made
    return row

if __name__ == "__main__":
    fleets = [tuple(map(int, f.split("-"))) for f in sys.argv[1].split(",")]
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    targ = sys.argv[3] if len(sys.argv) > 3 else "400"
    t0 = time.time()
    if targ == "star":
        _, st0 = build_cover_walks(LAY.G, LAY.priorities, 400.0, LAY.pos, Weights().v_g, K=1)
        T = math.ceil(st0["T_star"])
    else:
        T = float(targ)
    WALKS, st = build_cover_walks(LAY.G, LAY.priorities, T, LAY.pos, Weights().v_g, K=-1,
                                  ghosts=LAY.ghosts, r_max=Weights().r_max)
    GATED = set(st["gated"])
    print("T", T, {k: v for k, v in st.items() if k != "gated"}, "build s", round(time.time() - t0), flush=True)
    jobs = [(nu, na, vk, s) for nu, na in fleets for vk in KEYS for s in range(nseeds)]
    out = os.path.join(HERE, "..", "results_extra", f"m3_cover_T{int(T)}_{sys.argv[1].replace(',', '_')}.csv")
    rows = []
    with mp.get_context("fork").Pool() as p:
        for r in p.imap_unordered(job, jobs):
            rows.append(r); print(len(rows), "/", len(jobs), r["fleet"], r["variant"], r["seed"], round(time.time() - t0), flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("saved", out, round(time.time() - t0), "s")
