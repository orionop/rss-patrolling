# rss-patrolling

Simulation code for a TPBP (Time-Period-Based Patrolling) extension to
heterogeneous ground-air robot teams: UGVs run the original TPBP
walk-selection rule unchanged; UAVs launch from UGVs to service
air-only ("ghost") nodes and are recovered by a UGV afterward. This
repo holds only the code behind the paper's results — no stale
experiments, renders, or planning scaffolding.

## Files

- `sim/engine.py` — the simulation engine. Implements the unmodified
  TPBP walk enumeration and reward (`enumerate_valid_walks`,
  `walk_reward`), plus the extension's ghost-credit and
  recovery-credit reward terms, the UAV launch/recovery state
  machine, and the `Sim` class that steps UGVs and UAVs together.
  `VARIANTS` defines the 5 configurations compared in the paper:
  - `tier0_ugv_only` — floor: original TPBP, no UAVs at all.
  - `full` — proposed method, all mechanisms on.
  - `no_ghost_credit` — ablation: drop the ghost-credit reward term.
  - `fixed_launch_nodes` — ablation: UAVs launch from fixed nodes
    instead of the nearest-unclaimed-ghost rule.
  - `nearest_ugv_recovery` — ablation: recover a flown UAV with the
    nearest UGV instead of the coordinated recovery-credit rule.
- `sim/graph_layouts.py` — map construction. Builds the M1 (7x7 dense
  grid) and M2 (same nodes/priorities/ghosts, edges thinned toward a
  near-spanning-tree, with a connectivity guard so thinning can't
  orphan a ghost) layouts, plus the M3 loader for a real Boston OSM
  road network.
- `sim/run_m1.py`, `sim/run_m2.py` — run all 5 variants on M1/M2 at
  two fleet sizes (4 UGV/2 UAV and 6 UGV/3 UAV), 3 seeds each.
- `sim/run_m3.py` — runs the proposed method on the real-road M3 map
  across fleet sizes and emergency-event conditions.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
cd sim
python3 run_m1.py   # dense grid
python3 run_m2.py   # sparse grid (same nodes as M1, thinned edges)
python3 run_m3.py   # real Boston road network
```

Each script prints per-seed rows of `mean_idleness`,
`ghost_mean_idleness`, and `ghost_max_idleness` for every
variant/fleet-size combination.
