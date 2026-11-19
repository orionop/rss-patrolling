"""M1 emergency events, final-skeleton Sec 5 protocol.

One event per run at t0 = H/2, severity sigma in {1, 2, 3}, node drawn per seed
(uniform over ground and air-only nodes, same node for every sigma and variant).
Dispatch: full (quota, Alg. respond), V4 (single nearest), V5 (quota filled at random).
Patrolling is the full method in all runs; only dispatch differs.

Metrics, each paired with the same seed's run without the event:
  dJ      combined idleness (road mean + air-only mean) averaged over [t0, H],
          with event minus without
  clear   onset to end of handling (s)
  D_resp  responder travel distance to the event (m)
  first   onset to first responder arrival (s)

Usage: python3 run_m1_events.py [fleets e.g. 4-2,6-3] [seeds] [out.csv]
"""
import csv
import os
import random
import statistics as st
import sys

os.environ.setdefault("WALKSET", "cover")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multiprocessing import get_context

from engine import Sim, VARIANTS, EventConfig, event_quota
from runner import make_weights, _layout, _walks
from run_m1 import T_SECONDS, HORIZON_S, WALKS_PER_PAIR

MAP_IDX = 0
T, H = float(T_SECONDS), HORIZON_S   # same settings as the no-event M1 runs
T0 = H / 2                            # onset: mid-run, after the patrol is in steady state
NODE_SEED_BASE = 1000                 # RNG plumbing: event-node draw independent of the patrol RNG
V5_SEED_MULT = 7919                   # RNG plumbing: V5 draw distinct per (seed, sigma)
DISPATCH = {"full": "quota", "V4_single": "single", "V5_random": "random"}


def event_node(seed):
    lay = _layout(MAP_IDX)
    nodes = sorted(lay.G.nodes()) + sorted(lay.ghosts)
    return random.Random(NODE_SEED_BASE + seed).choice(nodes)


def reduced_fleet(nu, na, seed, sig):
    """fleet left patrolling once the quota responders are pulled (no event, same seed)"""
    air = event_node(seed) in _layout(MAP_IDX).ghosts
    q_g, q_a = event_quota(sig, air, EventConfig())
    return max(nu - q_g, 0), max(na - q_a, 0)


def window_combined(log, key=None):
    rows = [r for r in log if r["t"] >= T0]
    f = (lambda r: r["mean_idleness"] + r["ghost_mean_idleness"]) if key is None else (lambda r: r[key])
    return sum(f(r) for r in rows) / len(rows)


def job(a):
    nu, na, seed, sig, var = a
    lay = _layout(MAP_IDX)
    walks, gated = _walks(MAP_IDX, T, WALKS_PER_PAIR)
    ev = EventConfig(enabled=False)
    if var == "reduced":
        nu, na = reduced_fleet(nu, na, seed, sig)
        sig = 0
    if sig:
        ev = EventConfig(enabled=True, mode="dispatch", onset=T0, node=event_node(seed),
                         severity=sig, dispatch=DISPATCH[var], random_seed=V5_SEED_MULT * seed + sig)
    sim = Sim(lay, n_ugv=nu, n_uav=na, variant=VARIANTS["full"], seed=seed,
              weights=make_weights(T), valid_walks=walks, gated_walks=gated, events=ev)
    log = sim.run(H)
    if sig and sim.ev_stats["clear"] is not None:
        t_back = T0 + sim.ev_stats["clear"] + T      # cleared, plus one time period to rejoin patrol
    else:
        t_back = None
    row = {"fleet": f"{a[0]}-{a[1]}", "seed": seed, "severity": a[3], "variant": var,
           "win_combined": window_combined(log),
           "series": [(r["t"], r["mean_idleness"] + r["ghost_mean_idleness"]) for r in log if r["t"] >= T0][::10],
           "t_back": t_back,
           "win_road": window_combined(log, "mean_idleness"),
           "win_air": window_combined(log, "ghost_mean_idleness"),
           "combined": sum(r["mean_idleness"] + r["ghost_mean_idleness"] for r in log) / len(log)}
    if sig:
        row.update({k: sim.ev_stats.get(k) for k in ("node", "air", "n_ugv", "n_uav", "shortfall",
                                                     "D_resp", "first_arrival", "clear", "late",
                                                     "filled_at", "unfilled")})
    return row


if __name__ == "__main__":
    fleets = [tuple(map(int, f.split("-"))) for f in (sys.argv[1] if len(sys.argv) > 1 else "4-2,6-3").split(",")]
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "results_m1_events.csv")
    jobs = [(nu, na, s, 0, "none") for nu, na in fleets for s in range(ns)]
    jobs += [(nu, na, s, sig, v) for nu, na in fleets for s in range(ns)
             for sig in (1, 2, 3) for v in list(DISPATCH) + ["reduced"]]
    with get_context("fork").Pool() as p:
        rows = p.map(job, jobs)
    series = {(r["fleet"], r["seed"], r["severity"], r["variant"]): r.pop("series") for r in rows}

    def seg(key, t1, t2):
        xs = [c for t, c in series[key] if t1 <= t < t2]
        return st.mean(xs) if xs else None
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    base = {(r["fleet"], r["seed"]): r["win_combined"] for r in rows if r["severity"] == 0}
    broad = {(r["fleet"], r["seed"]): (r["win_road"], r["win_air"]) for r in rows if r["severity"] == 0}
    for nu, na in fleets:
        fl = f"{nu}-{na}"
        print(f"\nM1 {fl}, {ns} seeds, event at t0 = {T0:.0f} s, handling = T = {T:.0f} s")
        b = st.mean(v for (f_, _), v in base.items() if f_ == fl)
        print(f"  no-event window combined = {b:.1f}")
        print(f"  {'sigma':5} {'variant':10} {'dJ':>7} {'dJ%':>6} {'dRoad':>6} {'dAir':>6} {'clear':>7} {'sxclr':>6} {'D_resp':>7} {'first':>6} {'short':>5}"
              f" | full vs this: diff t wins | resp vs no-event, vs reduced fleet")
        for sig in (1, 2, 3):
            dj = {}
            for v in DISPATCH:
                rs = sorted([r for r in rows if r["fleet"] == fl and r["severity"] == sig and r["variant"] == v],
                            key=lambda r: r["seed"])
                dj[v] = [r["win_combined"] - base[(fl, r["seed"])] for r in rs]
                cl = [r["clear"] for r in rs if r["clear"] is not None]
                fa = [r["first_arrival"] for r in rs if r["first_arrival"] is not None]
                cmp = ""
                if v != "full":
                    d = [a - b for a, b in zip(dj["full"], dj[v])]
                    mu, se = st.mean(d), st.stdev(d) / len(d) ** .5
                    cmp = f"{mu:+7.1f} {mu / se if se else 0:6.2f} {sum(x < 0 for x in d)}/{len(d)}"
                dr = st.mean(r["win_road"] - broad[(fl, r["seed"])][0] for r in rs)
                da = st.mean(r["win_air"] - broad[(fl, r["seed"])][1] for r in rs)
                resp, red = [], []
                for r in rs:
                    if r["clear"] is None:
                        continue
                    t1, t2 = T0, T0 + r["clear"]
                    e = seg((fl, r["seed"], sig, v), t1, t2)
                    n0 = seg((fl, r["seed"], 0, "none"), t1, t2)
                    rf = seg((fl, r["seed"], sig, "reduced"), t1, t2)
                    if e is not None and n0:
                        resp.append(100 * (e - n0) / n0)
                    if e is not None and rf:
                        red.append(100 * (e - rf) / rf)
                post = []
                for r in rs:
                    if r["t_back"] is None:
                        continue
                    a = [c for t, c in series[(fl, r["seed"], sig, v)] if t >= r["t_back"]]
                    b0 = [c for t, c in series[(fl, r["seed"], 0, "none")] if t >= r["t_back"]]
                    if a and b0:
                        post.append(100 * (st.mean(a) - st.mean(b0)) / st.mean(b0))
                print(f"  {sig:<5} {v:10} {st.mean(dj[v]):7.1f} {100 * st.mean(dj[v]) / b:5.1f}% {dr:6.1f} {da:6.1f}"
                      f" {st.mean(cl) if cl else float('nan'):7.1f} {sig * st.mean(cl) if cl else float('nan'):6.0f}"
                      f" {st.mean(r['D_resp'] for r in rs):7.0f} {st.mean(fa) if fa else float('nan'):6.1f}"
                      f" {st.mean(r['shortfall'] for r in rs):5.2f} | {cmp:22} | {st.mean(resp):+6.1f}% {st.mean(red):+6.1f}%"
                      f"  served {sum(r['clear'] is not None for r in rs)}/{len(rs)}", flush=True)
                print(f"     post-recovery dJ% (after clear + T): {st.mean(post):+.1f}%  (n={len(post)})")
    print("saved", out)
