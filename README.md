# rss-patrolling

Simulation code for a TPBP (Time-Period-Based Patrolling) extension to
heterogeneous ground-air robot teams: UGVs run the original TPBP
walk-selection rule unchanged; UAVs launch from UGVs to service
air-only ("ghost") nodes and are recovered by a UGV afterward. This
repo holds only the code behind the paper's results: no stale
experiments, renders, or planning scaffolding.

## Current state (2026-11-19)

- Maps: **M1** (10x10 grid) and **M3** (Boston OSM). M2 (thinned near-tree
  grid, 4 independent cycles) is out of scope: the credits act through walk and
  landing-point choice, which a near-tree does not offer. M2 code and results
  are archived outside this repo.
- UAV flight limits: launch within r_max = 70 m; recovery UGV and landing node
  within r_rec = 500 m of the air-only node (nearest-UGV baseline: the chosen
  UGV stops until the UAV lands); air-only cap launch only from a carrier
  within r_rec, closest docked UAV only; flight to an event only if the
  carrier is within r_rec.
- Emergency events (`EventConfig(mode="dispatch")` in `sim/engine.py`): one
  event per run at t0 = H/2, severity 1-3, quota
  q = ceil(n_ref (sigma-1)/(sigma_max-1)) with n_ref = 2 UGV / 2 UAV, work
  W = (q_g + q_a) T, response mandatory (open slots refilled every step,
  courier UGV if no UAV is in range). `freeze_for` gives the frozen-fleet
  counterfactual.

```bash
cd sim
python3 run_m1_table2.py 30            # M1: Table 1 (no event) + Table 2 (event) + frozen check
MAP=M3 python3 run_m1_table2.py 30     # M3, same
python3 run_m1_events.py 4-2,6-3 30 > results_m1_events.txt   # responder selection (V4, V5)
python3 make_event_tables.py results_m1_events.txt > tables.tex  # LaTeX tables for the paper
python3 plot_m1_map.py                 # M1 figure
```

Results (30 seeds): `sim/results_m1_table2.csv`, `sim/results_m3_table2.csv`,
`sim/results_m1_events.csv`, `sim/results_m1_events.txt`.

Sections below are the development history; M2 mentions there are historical.

## Files

- `sim/engine.py`: the simulation engine. Implements the unmodified
  TPBP walk enumeration and reward (`enumerate_valid_walks`,
  `walk_reward`), plus the extension's ghost-credit and
  recovery-credit reward terms, the UAV launch/recovery state
  machine, and the `Sim` class that steps UGVs and UAVs together.
  `VARIANTS` defines the 5 configurations compared in the paper:
  - `tier0_ugv_only`: floor: original TPBP, no UAVs at all.
  - `full`: proposed method, all mechanisms on.
  - `no_ghost_credit`: ablation: drop the ghost-credit reward term.
  - `fixed_launch_nodes`: ablation: UAVs launch from fixed nodes
    instead of the nearest-unclaimed-ghost rule.
  - `nearest_ugv_recovery`: ablation: recover a flown UAV with the
    nearest UGV instead of the coordinated recovery-credit rule.
- `sim/graph_layouts.py`: map construction (M1 grid, M3 loader for the
  Boston OSM road network; the ladder also builds the archived M2 layout).
- `sim/run_m1.py`: runs all 5 variants on M1.
- `sim/run_m3.py`: runs the proposed method on the real-road M3 map
  across fleet sizes and emergency-event conditions.

- `sim/runner.py`: shared batch runner used by `run_m1.py`
  (parallel runs, progress lines, CSV output, time-averaged idleness).
- `sim/run_m1_table2.py`, `sim/run_m1_events.py`, `sim/make_event_tables.py`
 : event experiments and paper tables (see "Current state").
- `sim/plot_m1_map.py`: draws the M1 map to PDF.
- `sim/visualize.py`: live visual simulation with per-node idleness
  (see "Visual simulation" below).

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
cd sim
python3 run_m1.py   # dense grid
python3 run_m3.py   # real Boston road network
```

Each script prints per-seed rows of `mean_idleness`,
`ghost_mean_idleness`, and `ghost_max_idleness` for every
variant/fleet-size combination.

## Visual simulation

```bash
cd sim
python3 visualize.py                                   # M1, full method, fleet 4-2, seed 0
python3 visualize.py --variant fixed_launch_nodes --fleet 6-3 --seed 1
python3 visualize.py --save m1.gif --no-show           # render to a file, no window
python3 visualize.py --no-show --node-csv node_idleness.csv   # headless, per-node CSV
```

Nodes are coloured by current idleness; hover a node for its id and idleness,
click it to plot its idleness history. Keys: SPACE pause, UP/DOWN speed, `s`
writes the per-node CSV. In code, `Sim.node_idleness()` returns the current
idleness of every node, and `Sim(..., track_node_idleness=True)` plus
`sim.node_idleness_stats()` gives each node's running average and peak.

## TPBP conformance (what the UGV rule implements)

`select_walk` / `walk_reward` follow Mallya et al. (ICRA 2021), Sec. III:
- agents start on distinct priority nodes;
- an agent may only pick walks ending in Y(t) = S \ X(t), where X(t) is the
  set of other agents' destination priority nodes (so two agents never head
  for the same priority node); the agent's own node stays in Y;
- Term 2 sums idleness over the priority nodes in the walk that are in Y,
  Term 3 uses the shortest distance to each s_j in Y from any node of the walk;
- valid walks may be closed (end node = start node).

Deliberate simplifications that are NOT in the paper: valid walks are simple
paths bounded by a hop count (`T_HOPS`) rather than by travel time T, at most
`WALKS_PER_PAIR` shortest walks per start/end pair are kept, the overshoot
threshold T in Term 2 is 0, and Terms 3/4 are measured in hops/nodes, not metres.

## Synchronized rendezvous (Sec 3.3 of the formulation)

- Recovery is assigned once per service cycle, after the service dwell, among
  empty, unreserved UGVs; the reservation is released only on docking.
- Landing candidates are the ground nodes at path distances l/3, 2l/3, l along
  the UGV's remaining planned path (duplicates removed); t_u is the path
  travel time, t_a = ||p - q_h|| / s_a, TL = max(t_u, t_a), t_dep = TL - t_a.
- The reserved UGV follows its walk to the landing node and waits there; the
  UAV departs at t_dep. A vehicle that arrives first waits, and the node's
  idleness is held at 0 while it waits (Definition 1). Docking is by
  co-location within eps_l.
- A ghost's idleness restarts from 0 when the UAV leaves it.
- `nearest_ugv_recovery` ablation: the UAV leaves immediately and pursues the
  nearest free UGV's current position; the UGV keeps walking and does not wait.
  (The formulation leaves this UGV behaviour to be fixed; this is the choice made.)

## Walk set and Term 1 (latest changes)
- Valid walks now also include non-simple "detour" walks (s -> v -> f) within the hop bound, so every node is reachable on sparse maps (M2: 44/100 nodes were unreachable under simple paths only).
- Reward Term 1 counts each distinct node in a walk once.
- T = 400 s (Weights.T, set by T_SECONDS in run_m1.py/run_m2.py): a walk is valid iff its travel time (edge length / v_g) <= T, and overshoot is [I_s - T]+. Chosen because the longest M1 priority pair is 375 s, so every priority pair is connected, and larger T (500-800 s) gave worse idleness and coverage in a 1 h sweep. T = 100 s allows <= 4 hops, shorter than any priority pair, so UGVs could only loop.
- With |UGVs| = |priority nodes| the claim rule leaves each UGV only its own priority node, so it repeats closed walks. Use fewer UGVs than priority nodes to see roaming.

## Starvation guard (extension, not in TPBP)
TPBP's reward is a sum of idleness over a walk, so a stale node that only short walks reach (e.g. a leaf next to a priority node on M2) can lose to long walks for hours. `STARVE_FACTOR = 4` in `sim/runner.py` makes a UGV, whenever a ground node's idleness exceeds 4T (1600 s at T = 400 s), choose among its eligible walks only those that cover the stalest such node. It applies to every variant equally. Set `STARVE_FACTOR = None` for the pure TPBP reward. On M2 (2 h, 3 seeds) it took never-visited nodes from 0-1 per run to 0 and capped the worst node idleness near 2000-3400 s instead of 7200 s.

## Ghost starvation guard (optional extension)
`GHOST_STARVE_FACTOR` in `sim/runner.py` (default `None` = off). When a ghost's idleness exceeds GHOST_STARVE_FACTOR * T, a UGV chooses among its eligible walks only those passing within r_max (70 m) of the stalest such ghost, so a UAV can launch toward it. It does nothing for the UGV-only and fixed-launch-node variants. In a 2 h, 3-seed test on M1 and M2 at 2T, 3T and 4T it changed mean ghost idleness by under 4 % and left the worst ghost idleness essentially unchanged (ghost staleness is limited by UAV availability and the launch rule, not by UGVs never passing near ghosts), so it is off by default.

## Ghost idleness cap (extension to Sec 3.2)
`GHOST_CAP_FACTOR = 3` in `sim/runner.py`: if an unclaimed ghost's idleness exceeds 3T (1200 s at T = 400 s), a docked UAV may launch to it from any distance (r_max is waived for that launch). Peak ghost idleness then stays about 100 s above the cap (flight time) instead of reaching 2000-3500 s. Ground idleness is unchanged (within noise). It applies only to variants that launch by proximity (full, no_ghost_credit, nearest_ugv_recovery); the fixed-launch-node and UGV-only variants do not get it, so comparisons against them are no longer like for like. Set `GHOST_CAP_FACTOR = None` for the pure Sec 3.2 launch rule. The earlier walk-level ghost guard (`GHOST_STARVE_FACTOR`) stays off: the UGV carrying a free UAV often cannot reach the stale ghost at all.

## Paper results (CSV)
1. `cd sim && python3 run_m1.py && python3 run_m2.py` writes `results_m1.csv`, `results_m2.csv` (one row per fleet x variant x seed, all 5 variants).
2. `python3 summarize_results.py` writes `paper_table.csv` (mean and std over seeds). Raise `SEEDS` in the run scripts (e.g. `list(range(10))`) before the final run.

## Weights and formulation conformance (latest)
Ghost credit and recovery credit now follow Eq. (6) and Eq. (7) of the formulation exactly (`Sim.walk_reward`):
- Eq. (6): c5*Ihat + c6*[Ihat - T_h]+ , with Ihat = ghost idleness predicted at the walk's first node in L_h, T_h = the walk's remaining TIME (s) at that node (before: c6 was multiplied by c5, T_h was a hop count, Ihat was the current idleness). Conditions 1-4 as in the text; the launch target at that node follows Sec 3.2 (nearest unclaimed ghost, ties to highest idleness); one credited ghost per walk (one UAV per UGV).
- Eq. (7): c7/trem + c8*[T_h - trem]+ for each occupied ghost the walk passes within r_max of, free UGVs only (before: the c8 term was missing, which made the recovery credit negligible).

Weights: c1=1, c2=2, c3=0.5, c4=0.1 (hand-set, untuned; TPBP always uses c1=1 and tunes the rest empirically).
c5-c8 presets in `sim/runner.py` (`WEIGHT_SET`, or env var `WEIGHT_SET=tuned`):
- design_rule (default): c5=13, c6=3, c7=9, c8=5. Satisfies Remark 2 (13*4 = 52 > 51) and Remark 5 (9*6 = 54 > 51), Wmax = 17 nodes, (c1+c2)*Wmax = 51. Best of a sweep over rule-satisfying values on tuning seeds 100-102.
- tuned: c5=3, c6=1.5, c7=2, c8=5. Best of a free sweep on the same tuning seeds; violates the design rule (7.5 and 12 < 51).

## Investigating M2 6-3 (tools/)
Run from the project root (`python tools/<script>.py`); each prints Combined idleness (s) on tuning seeds 100-104.
- `tools/sweep_thresholds.py`: sweeps r_max, the air-only cap and the service dwell, full vs nearest_ugv_recovery, on M2 6-3, M2 4-2, M1 6-3. Edit `tests` / `cases` / `SEEDS` at the bottom.
- `tools/sweep_hybrid.py`: sweeps the optional hybrid-recovery bound `Weights.recovery_tau_max` (seconds; None = off, the default).
- `tools/diag_split.py`: ground / air-only mean / air-only peak split per variant.
Weights used in these scripts are the "tuned" set (c5=3, c6=1.5, c7=2, c8=5); change them in the `replace(...)` line.

## Current version: legacy credits + overlap discount (delta = 0.5)
With DISCOUNT_OTHERS = 0 this is the code that produced the original M1/M2 tables: full 856.9 / 536.6 / 1191.4 / 877.6 (combined, 10 seeds). `sim/engine.py` is the version before the Eq. 6/7 rewrite. Deviations from the formulation, to state in the paper:
- the ghost credit uses `c5 * (I_hat + c6 * max(I_hat - t_h, 0))` with t_h = hops left on the walk (not seconds);
- the recovery credit uses only `c7 / t_rem`; c8 is not used;
- the launch rule has no I_min (I_min = 20 s applies to the credit only);
- weights c1..c8 = 1, 2, 0.5, 0.1, 3, 1.5, 2, 1 (untuned placeholders; they do not satisfy Remarks 2 and 5).
Reproduce: `cd sim && python3 run_m1.py && python3 run_m2.py` (about 2 minutes). Newer experimental options (launch I_min, gateway, hybrid recovery, extra landing candidates) are NOT in this version.

## Extra experiments (results_extra/, scripts in tools/)
- `results_reduced_fleets.csv` (tools/reduced.py): M1 and M2, fleets 4-1, 5-1, 4-2, 5-2, 10 seeds. Full is lowest in all 8 rows; used in the paper's Reduced Fleets subsection.
- `results_20x20_fleet8-4.csv`, `results_20x20_fleet6-3.csv` (tools/big*.py): 20x20 grid (400 nodes, 24 priorities, 20 air-only nodes, 100 walks per pair; building the walk set takes about 3 minutes). Full is behind nearest-UGV recovery by 18-64.
- `tools/coverage.py`, `coverage2.py`: walk coverage and rarely-visited nodes on M2 versus T and the walk cap.
- Scripts that write `results_m2.csv` overwrite the main results; copy first.

## Overlap discount (added last)
`DISCOUNT_OTHERS = 0.5` in `sim/runner.py` (`Weights.discount_others` in `sim/engine.py`): in Term 1 of the walk reward, a node on another UGV's remaining walk counts with weight (1 - delta). 0 reproduces the original numbers exactly (full 856.9 / 536.6 / 1191.4 / 877.6 on M1 4-2, 6-3, M2 4-2, 6-3). It applies to every variant including the UGV-only floor. delta was compared at 0, 0.5, 1 on seeds 0-9 and 0 vs 0.5 on seeds 100-109.
All CSVs in `sim/` and `results_extra/` are from the code with delta = 0.5 (M2 8-4 is `sim/results_m2_fleet8-4.csv`; the 'Legacy-credit' numbers quoted earlier in this README refer to delta = 0).

## Covering walk set (final)

All reported results use the covering walk set (`build_cover_walks` in `sim/engine.py`):

- K = |corridor(s, f)| fastest walks per priority pair, searched inside the corridor
  {v : d(s,v) + d(v,f) <= T} (exact; ranked by travel time, not hops).
- Off-path nodes get greedy detours from their nearest priority node. Detours are gated:
  they enter the walk choice only once a node on them passes the starvation threshold.
- Starvation claims for off-path nodes, cap N_c = ceil(n_UGV * |U+| / |V_g|).
- T must satisfy T >= T* = max over edges of d(u,S) + w(u,w) + d(w,S); M3 runs at T = 421 s.

Reward, selection, guard and weights are unchanged.

```
cd sim
WALKSET=cover python3 run_m1.py      # -> results_m1_cover.csv
WALKSET=cover python3 run_m2.py      # -> results_m2_cover.csv
WALKSET=present python3 run_m1.py    # old walk set, for comparison
cd ../tools
python3 m3_cover_run.py "4-2,6-3,8-4,10-5,10-7" 10 star   # M3 at T = T*
```
