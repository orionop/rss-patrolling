"""M3 (Boston OSM, cached road network), fleets 8/4 10/5 10/7, 10 seeds, T = 421 s (>= T* = 420.2 s).
Usage: python3 run_m3.py [fleets e.g. 8-4,10-5,10-7] [seeds] [T] [out.csv]  (about 15 min on 2 cores incl. walk-set build)"""
import sys, os, csv, time, math, multiprocessing as mp
os.environ.setdefault("WALKSET", "cover")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from graph_layouts import make_ladder
from engine import Sim, VARIANTS, Weights, build_cover_walks
from runner import make_weights, summarize
LAY = make_ladder(include_m3=True)[2]
KEYS = ["full", "no_ghost_credit", "fixed_launch_nodes", "nearest_ugv_recovery", "tier0_ugv_only"]
WALKS = GATED = None
T = 421.0

def job(a):
    nu, na, vk, seed = a
    sim = Sim(LAY, n_ugv=nu, n_uav=na, variant=VARIANTS[vk], seed=seed, weights=make_weights(T), valid_walks=WALKS, gated_walks=GATED)
    fd = [0.0]; osu = sim.step_uav
    def su(u):
        b = u.pos_xy; fly = u.state in ("to_ghost", "to_rendezvous"); osu(u)
        if fly and b is not None: fd[0] += math.dist(b, u.pos_xy)
    sim.step_uav = su
    row = {"map": "M3", "fleet": f"{nu}-{na}", "variant": vk, "seed": seed, "T": round(T, 2)}
    row.update({k: round(v, 2) for k, v in summarize(sim.run(7200)).items()})
    row["airtime_s"] = round(fd[0] / 12.0, 1)      # flying time: distance / 12 m/s
    return row

if __name__ == "__main__":
    fleets = [tuple(map(int, f.split("-"))) for f in (sys.argv[1] if len(sys.argv) > 1 else "8-4,10-5,10-7").split(",")]
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 421.0
    out = sys.argv[4] if len(sys.argv) > 4 else "results_m3_cover.csv"
    WALKS, st = build_cover_walks(LAY.G, LAY.priorities, T, LAY.pos, Weights().v_g, K=-1, ghosts=LAY.ghosts, r_max=Weights().r_max)
    GATED = set(st["gated"])
    jobs = [(nu, na, vk, s) for nu, na in fleets for vk in KEYS for s in range(nseeds)]
    rows = []
    with mp.get_context("fork").Pool() as p:
        for r in p.imap_unordered(job, jobs):
            rows.append(r); print(len(rows), "/", len(jobs), r["fleet"], r["variant"], r["seed"], flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("saved", out)
