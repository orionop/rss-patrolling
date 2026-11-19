"""M1 Table 2: full method vs baselines WITH an emergency event, plus the frozen-fleet check.

Table 1 (no event) and Table 2 (event) use the same seeds, fleets, horizon and T as run_m1.
Event: one per run at t0 = H/2, node drawn per seed (same node for every variant and sigma),
severity sigma in {1, 2, 3}, dispatch = Alg. respond (quota) for every variant. Only the
patrolling method differs between columns.

tier0_ugv_only is left out: it has no UAVs, so it cannot respond to an air-only event.

Frozen check (full method only): same seed, no event; at t0 the quota robots, chosen nearest
first as in Alg. respond, stop where they are for the clear time measured in the paired event run,
then resume. Event run minus frozen run = cost of responding beyond losing those robots.

Usage: python3 run_m1_table2.py [seeds] [out.csv]
       MAP=M3 python3 run_m1_table2.py 30   (M3: run_m3.py layout, T, fleets, walk set)
"""
import csv
import os
import statistics as st
import sys

os.environ.setdefault("WALKSET", "cover")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multiprocessing import get_context

from engine import Sim, VARIANTS, EventConfig
from runner import make_weights, _layout, _walks, summarize
import importlib
import random
import run_m1_events as rme
MAP = os.environ.get("MAP", "M1")
MAP_IDX = {"M1": 0, "M3": 2}[MAP]           # M2 out of scope (archive/2026-11-19_m2-out-of-scope)
if MAP == "M3":
    # M3 setup exactly as run_m3.py: Boston OSM layout, its T, fleets and full covering walk set
    import run_m3
    from engine import Weights, build_cover_walks
    from run_m1 import HORIZON_S                       # same 2 h horizon as run_m3 (sim.run(7200))
    T_SECONDS = run_m3.T
    FLEET_SIZES = [(8, 4), (10, 5), (10, 7)]           # run_m3.py default fleets
    _LAY = run_m3.LAY
    _W, _st = build_cover_walks(_LAY.G, _LAY.priorities, T_SECONDS, _LAY.pos, Weights().v_g, K=-1,
                                ghosts=_LAY.ghosts, r_max=Weights().r_max)
    _WALKS = (_W, set(_st["gated"]))
    get_layout = lambda: _LAY
    get_walks = lambda: _WALKS
else:
    _cfg = importlib.import_module(f"run_{MAP.lower()}")
    T_SECONDS, HORIZON_S, WALKS_PER_PAIR, FLEET_SIZES = (_cfg.T_SECONDS, _cfg.HORIZON_S,
                                                           _cfg.WALKS_PER_PAIR, _cfg.FLEET_SIZES)
    get_layout = lambda: _layout(MAP_IDX)
    get_walks = lambda: _walks(MAP_IDX, float(T_SECONDS), WALKS_PER_PAIR)
T, H = float(T_SECONDS), HORIZON_S
T0 = H / 2


def event_node(seed):
    """same draw as run_m1_events.event_node, on this map"""
    lay = get_layout()
    nodes = sorted(lay.G.nodes()) + sorted(lay.ghosts)
    return random.Random(rme.NODE_SEED_BASE + seed).choice(nodes)


METHODS = [v for v in VARIANTS if not VARIANTS[v].ugv_only]   # full + baselines that have UAVs
SIGMAS = (1, 2, 3)


def combined(r):
    return r["mean_idleness"] + r["ghost_mean_idleness"]


def job(a):
    nu, na, var, seed, sig, freeze = a
    lay = get_layout()
    walks, gated = get_walks()
    ev = EventConfig(enabled=False)
    if sig:
        ev = EventConfig(enabled=True, mode="dispatch", onset=T0, node=event_node(seed),
                         severity=sig, dispatch="quota", freeze_for=freeze)
    sim = Sim(lay, n_ugv=nu, n_uav=na, variant=VARIANTS[var], seed=seed,
              weights=make_weights(T), valid_walks=walks, gated_walks=gated, events=ev)
    log = sim.run(H)
    clear = sim.ev_stats["clear"] if sig else None
    row = {"fleet": f"{nu}-{na}", "variant": var, "seed": seed, "severity": sig,
           "frozen": freeze is not None, "clear": clear,
           "combined": st.mean(combined(r) for r in log)}
    if clear is not None:
        w = [combined(r) for r in log if T0 <= r["t"] < T0 + clear]
        row["resp_window"] = st.mean(w) if w else None
    if sig and freeze is None:
        row["n_ugv"], row["n_uav"] = sim.ev_stats["n_ugv"], sim.ev_stats["n_uav"]
    if not sig:
        row.update(summarize(log))                # Table 1 metrics: road, air mean/peak, road/priority peak
    return row


def paired_t(d):
    mu = st.mean(d)
    se = st.stdev(d) / len(d) ** .5 if len(d) > 1 else 0
    return mu, (mu / se if se else 0.0), sum(x < 0 for x in d)


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, f"results_{MAP.lower()}_table2.csv")
    seeds = range(ns)
    jobs = [(nu, na, v, s, sig, None) for nu, na in FLEET_SIZES for v in METHODS
            for s in seeds for sig in (0,) + SIGMAS]
    jobs += [(nu, na, v, s, 0, None) for nu, na in FLEET_SIZES for v in VARIANTS
             if v not in METHODS for s in seeds]        # UGV-only floor: no-event table only
    with get_context("fork").Pool() as p:
        rows = p.map(job, jobs)
        R = {(r["fleet"], r["variant"], r["seed"], r["severity"]): r for r in rows}
        fz_jobs = [(nu, na, "full", s, sig, R[(f"{nu}-{na}", "full", s, sig)]["clear"])
                   for nu, na in FLEET_SIZES for s in seeds for sig in SIGMAS]
        fz_rows = p.map(job, fz_jobs)
    F = {(r["fleet"], r["seed"], r["severity"]): r for r in fz_rows}
    allrows = rows + fz_rows
    keys = sorted({k for r in allrows for k in r}, key=lambda k: list(allrows[-1]).index(k) if k in allrows[-1] else 99)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)

    def m(fl, v, sig):
        return st.mean(R[(fl, v, s, sig)]["combined"] for s in seeds)

    print(f"{MAP} Table 1 vs Table 2, {ns} seeds, H = {H:.0f} s, event at t0 = {T0:.0f} s, whole-run combined idleness")
    for nu, na in FLEET_SIZES:
        fl = f"{nu}-{na}"
        print(f"\n  fleet {fl}")
        print(f"    {'method':22} {'no event':>9} " + " ".join(f"{'sigma' + str(s):>16}" for s in SIGMAS))
        for v in METHODS:
            b = m(fl, v, 0)
            cells = " ".join(f"{m(fl, v, s):8.1f} ({100 * (m(fl, v, s) / b - 1):+5.1f}%)" for s in SIGMAS)
            print(f"    {v:22} {b:9.1f} {cells}")
        for sig in (0,) + SIGMAS:
            best, bt = None, None
            for v in METHODS[1:] if METHODS[0] == "full" else [x for x in METHODS if x != "full"]:
                d = [R[(fl, "full", s, sig)]["combined"] - R[(fl, v, s, sig)]["combined"] for s in seeds]
                mu, t, wins = paired_t(d)
                if best is None or mu > bt[0]:
                    best, bt = v, (mu, t, wins)
            verdict = "WIN" if bt[0] < 0 else "LOSS"
            print(f"    sigma {sig}: {verdict:4} closest {best} diff {bt[0]:+.1f} t {bt[1]:.2f} wins {bt[2]}/{ns}")
        for sig in SIGMAS:
            dw = [R[(fl, "full", s, sig)]["resp_window"] - F[(fl, s, sig)]["resp_window"] for s in seeds
                  if R[(fl, "full", s, sig)].get("resp_window") is not None and F[(fl, s, sig)].get("resp_window") is not None]
            dr = [R[(fl, "full", s, sig)]["combined"] - F[(fl, s, sig)]["combined"] for s in seeds]
            fw = st.mean(F[(fl, s, sig)]["resp_window"] for s in seeds if F[(fl, s, sig)].get("resp_window") is not None)
            mu_w, t_w, _ = paired_t(dw)
            mu_r, t_r, _ = paired_t(dr)
            print(f"    frozen check sigma {sig}: event - frozen, response window {mu_w:+.1f} ({100 * mu_w / fw:+.1f}%, t {t_w:.2f}),"
                  f" whole run {mu_r:+.1f} (t {t_r:.2f})")
    print("saved", out)
