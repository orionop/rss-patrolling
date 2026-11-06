"""Run M3 (Boston OSM, 204 nodes) in two conditions, full method only:
  (a) no_events  -- Sec 1-3 only, same as M1/M2's "full" variant
  (b) with_events -- (a) plus the emergency-event layer (engine.py
      EventConfig), severity-scaled responder diversion

Fleet sizes per the agreed M3 scaling: (8,4), (10,5), (10,7) -- larger
than M1/M2 since M3's graph is ~4x the nodes. NOT executed yet -- staged.

Usage once approved:
    python3 run_m3.py
"""
from graph_layouts import make_ladder
from engine import Sim, VARIANTS, EventConfig

HORIZON_S = 3600 * 2
FLEET_SIZES = [(8, 4), (10, 5), (10, 7)]
SEEDS = [0, 1, 2]

CONDITIONS = {
    "no_events": EventConfig(enabled=False),
    "with_events": EventConfig(enabled=True, rate_per_hour=3.0,
                                severity_min=1, severity_max=5,
                                dwell_per_severity=15.0),
}


def run_all():
    ladder = make_ladder()
    m3 = ladder[2]
    full = VARIANTS["full"]
    results = []
    for n_ugv, n_uav in FLEET_SIZES:
        for cond_key, events in CONDITIONS.items():
            for seed in SEEDS:
                sim = Sim(m3, n_ugv=n_ugv, n_uav=n_uav, variant=full,
                          seed=seed, events=events)
                log = sim.run(HORIZON_S)
                final = log[-1] if log else {}
                results.append({
                    "map": "M3",
                    "fleet": f"{n_ugv}-{n_uav}",
                    "condition": cond_key,
                    "seed": seed,
                    "mean_idleness": final.get("mean_idleness"),
                    "ghost_mean_idleness": final.get("ghost_mean_idleness"),
                    "ghost_max_idleness": final.get("ghost_max_idleness"),
                    "event_mean_response": final.get("event_mean_response"),
                    "event_max_response": final.get("event_max_response"),
                    "events_resolved": final.get("events_resolved"),
                    "events_pending": final.get("events_pending"),
                })
                print(results[-1])
    return results


if __name__ == "__main__":
    run_all()
