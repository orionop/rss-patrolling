"""M3 (Boston OSM, cached) on the CURRENT engine: legacy credits + overlap discount.
Usage: python3 m3_run.py "4-2,6-3" [seeds] [walks_per_pair]
Needs osmnx and the Overpass cache in ../sim/cache (no network)."""
import sys, os, csv, time, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); SIM = os.path.join(HERE, "..", "sim")
sys.path.insert(0, SIM); os.chdir(SIM)
from graph_layouts import make_ladder
from engine import Sim, VARIANTS, enumerate_valid_walks, Weights
from runner import make_weights, summarize
T = 400; HORIZON = 7200
LAY = make_ladder(include_m3=True)[2]
WALKS = None
KEYS = ["full", "no_ghost_credit", "fixed_launch_nodes", "nearest_ugv_recovery", "tier0_ugv_only"]

def job(a):
    nu, na, vk, seed = a
    sim = Sim(LAY, n_ugv=nu, n_uav=na, variant=VARIANTS[vk], seed=seed,
              weights=make_weights(T), valid_walks=WALKS)
    row = {"map": "M3", "fleet": f"{nu}-{na}", "variant": vk, "seed": seed}
    row.update({k: round(v, 2) for k, v in summarize(sim.run(HORIZON)).items()})
    return row

if __name__ == "__main__":
    fleets = [tuple(map(int, f.split("-"))) for f in sys.argv[1].split(",")]
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    t0 = time.time()
    WALKS = enumerate_valid_walks(LAY.G, LAY.priorities, 0, k, t_seconds=T, pos=LAY.pos, v_g=Weights().v_g)
    print("walks", len(WALKS), "build s", round(time.time() - t0), flush=True)
    jobs = [(nu, na, vk, s) for nu, na in fleets for vk in KEYS for s in range(nseeds)]
    out = os.path.join(HERE, "..", "results_extra", f"m3_{sys.argv[1].replace(',', '_')}.csv")
    rows = []
    with mp.get_context("fork").Pool() as p:
        for r in p.imap_unordered(job, jobs):
            rows.append(r); print(len(rows), "/", len(jobs), r["fleet"], r["variant"], r["seed"], flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("saved", out, round(time.time() - t0), "s")
