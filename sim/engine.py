"""Discrete-time simulation engine for persistent patrolling with
heterogeneous UGV-UAV teams, implementing genswarm-extension.tex
Sections 1-3 (no emergency events yet -- that's a separate layer for
M3, not needed for M1/M2). Each mechanism is gated by a VariantConfig
flag, so the same engine runs the full method and every Tier 0/1
ablation just by flipping flags -- nothing is duplicated per variant.

Time is discretized into fixed steps of DT seconds; UGVs move at
constant speed v_g along their current walk's edges (real Euclidean
length, not hop count), UAVs fly at constant speed s_a in a straight
line. This is an approximation of the continuous-time formulation
(Sec 3.2's launch check becomes "within r_max at this discrete step"
rather than truly continuous), accurate to within one timestep's
travel distance -- acceptable at DT small relative to r_max / v_g.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from graph_layouts import Layout, Coord


# ---------------------------------------------------------------------------
# Variant configuration: toggles exactly the mechanisms that distinguish
# Tier 0 / Tier 1 ablations from the full method. Nothing else changes.
# ---------------------------------------------------------------------------

@dataclass
class VariantConfig:
    name: str
    ugv_only: bool = False          # Tier 0 floor: no UAVs at all
    ghost_credit: bool = True       # Sec 2.3 c5/c6 term
    recovery_credit: bool = True    # Sec 2.4 c7/c8 term
    radius_launch: bool = True      # Sec 3.2 live proximity check; False = fixed launch nodes
    landing_candidate_recovery: bool = True  # Sec 3.3 step 3; False = nearest-free-UGV, no coordination


VARIANTS = {
    "tier0_ugv_only": VariantConfig("UGV-only (floor)", ugv_only=True,
                                     ghost_credit=False, recovery_credit=False,
                                     radius_launch=False, landing_candidate_recovery=False),
    "full": VariantConfig("Full method"),
    "no_ghost_credit": VariantConfig("No ghost credit", ghost_credit=False),
    "fixed_launch_nodes": VariantConfig("Fixed launch nodes", radius_launch=False),
    "nearest_ugv_recovery": VariantConfig("Nearest-UGV recovery",
                                           recovery_credit=False,
                                           landing_candidate_recovery=False),
}


# ---------------------------------------------------------------------------
# Reward weights (TPBP base: c1-c4; ours: c5-c8). Not tuned yet -- these are
# placeholder magnitudes satisfying the starvation design rules of
# genswarm-extension.tex Sec 2.3/2.4 (c5(1+c6) > (c1+c2)*W_max, and the
# analogous inequality for c7/c8), not claimed optimal.
# ---------------------------------------------------------------------------

@dataclass
class Weights:
    c1: float = 1.0
    c2: float = 2.0
    c3: float = 0.5
    c4: float = 0.1
    c5: float = 3.0
    c6: float = 1.5
    c7: float = 2.0
    c8: float = 1.0
    ghost_units_seconds: bool = False  # ghost credit hinge uses walk travel time in seconds instead of node count
    discount_others: float = 0.0  # EXTENSION (overlap discount): scale the idleness credit of nodes on other UGVs' remaining walks by (1 - this)
    c9: float = 5.0           # emergency diversion weight, severity-scaled (M3 events layer only)
    r_max: float = 70.0       # launch/recovery-credit proximity radius (meters)
    i_min: float = 20.0       # minimum ghost idleness before credit fires
    T: float = 400.0          # shared priority time period T (seconds): walk travel-time bound and overshoot threshold
    starve_idleness: Optional[float] = None  # EXTENSION (not in TPBP): if a ground node's idleness exceeds this (s), restrict the choice to eligible walks that cover the stalest such node. None = pure TPBP reward
    ghost_starve_idleness: Optional[float] = None  # EXTENSION (not in TPBP): if a ghost's idleness exceeds this (s), restrict the choice to eligible walks that pass within r_max of the stalest such ghost (only for variants that launch UAVs by proximity). None = off
    ghost_cap_idleness: Optional[float] = None  # EXTENSION (not in Sec 3.2): if an unclaimed ghost's idleness exceeds this (s), a docked UAV may launch to it from ANY distance (r_max waived). None = off
    v_g: float = 4.0          # UGV speed, m/s
    s_a: float = 12.0         # UAV speed, m/s (s_a >> v_g, per formulation)
    eps_l: float = 2.0        # rendezvous co-location tolerance (meters)
    landing_candidates: int = 3  # C in Sec 3.3 step 3
    gate_idleness: Optional[float] = None  # detour gate threshold (s); None = starve_idleness
    recovery_hops: Optional[int] = None  # EXTENSION: UAV recovered only by a UGV (and landing node) within this many graph hops of the air-only node's nearest ground node; None = unlimited
    recovery_radius_m: Optional[float] = None  # EXTENSION: recovering UGV (current position) and landing node must be within this straight-line distance (m) of the air-only node
    claim_cap: Optional[int] = None  # COVERING SET (final-skeleton Sec 7.4): N_c, most UGVs holding a starvation claim at once. None = old starvation guard


# ---------------------------------------------------------------------------
# Emergency-event layer (M3 only, "with events" condition). NOT part of
# genswarm-extension.tex Sec 1-3 formulation text seen this session --
# this is a minimal, explicitly-placeholder severity-scaled responder
# model built to exercise the title's "emergency response" claim before
# the formal formulation section exists in the paper. Flag this to the
# user if/when the real spec diverges: current design is Poisson arrivals
# at random non-ghost nodes, severity in [1,5] drawn uniform, nearest
# free UGV (by travel time) diverts and "resolves" the event after a
# severity-proportional dwell, same idleness/occupied-since bookkeeping
# as a ghost. Reward diversion term c9*severity added to any walk whose
# node set includes an active unclaimed event.
# ---------------------------------------------------------------------------

@dataclass
class EventConfig:
    enabled: bool = False
    rate_per_hour: float = 3.0     # Poisson arrival rate, placeholder
    severity_min: int = 1
    severity_max: int = 5
    dwell_per_severity: float = 15.0   # seconds of on-site service per severity point
    # --- final-skeleton Sec 5 protocol (mode="dispatch"): one event per run ---
    mode: str = "poisson"          # "poisson" = old placeholder above; "dispatch" = Sec 5
    onset: float = 3600.0          # event time (s)
    node: Optional[str] = None     # event node (ground or air-only)
    severity: int = 1              # sigma in {1, 2, 3}
    handling: Optional[float] = None   # on-site handling time (s); None = T (one time period)
    dispatch: str = "quota"        # "quota" (full, Alg. respond), "single" (V4), "random" (V5)
    random_seed: int = 0           # V5 draw, separate from the patrol RNG
    sigma_max: int = 3             # highest severity level
    ref_ugv: int = 2               # reference fleet the quota is designed against (Sec 5)
    ref_uav: int = 2
    freeze_for: Optional[float] = None   # counterfactual: quota robots stop patrolling for this long, no event


def event_quota(sigma: int, air: bool, cfg: "EventConfig") -> Tuple[int, int]:
    """Sec 5 quota rule. Severity scales linearly from one robot (sigma = 1) to the
    whole reference fleet (sigma = sigma_max):
        q_g = ceil(ref_ugv * (sigma - 1) / (sigma_max - 1)),  same for q_a with ref_uav,
    then at least one robot of a type that can reach the node. An air-only node
    takes UAVs only (q_g = 0). Holds for any sigma_max >= 1 and any reference fleet."""
    s = min(max(int(sigma), 1), max(int(cfg.sigma_max), 1))
    frac = 0.0 if cfg.sigma_max <= 1 else (s - 1) / (cfg.sigma_max - 1)
    q_g = math.ceil(max(cfg.ref_ugv, 0) * frac - 1e-9)
    q_a = math.ceil(max(cfg.ref_uav, 0) * frac - 1e-9)
    if air:
        return 0, max(q_a, 1)
    if q_g + q_a == 0:                     # one robot, of a type the reference fleet has
        return (1, 0) if cfg.ref_ugv > 0 or cfg.ref_uav <= 0 else (0, 1)
    return q_g, q_a


@dataclass
class EmergencyEvent:
    eid: str
    node: str
    severity: int
    spawn_t: float
    claimed_by: Optional[str] = None   # UGV id
    resolved_t: Optional[float] = None
    service_start: Optional[float] = None


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

@dataclass
class UGV:
    uid: str
    pos_node: str                 # current/most-recent node
    edge_progress: float = 0.0    # meters traveled along current edge
    walk: List[str] = field(default_factory=list)
    walk_idx: int = 0             # index into walk of the NEXT node to reach
    docked_uav: Optional[str] = None
    reserved_for: Optional[str] = None   # UAV id this UGV is reserved to recover
    reserved_point: Optional[Coord] = None

    @property
    def is_free(self) -> bool:
        return self.docked_uav is None and self.reserved_for is None

    @property
    def xy_hint(self) -> Optional[Coord]:
        return None  # resolved by Sim.ugv_position(), needs graph positions


class UAVState:
    DOCKED = "docked"
    TO_GHOST = "to_ghost"
    SERVICING = "servicing"
    TO_RENDEZVOUS = "to_rendezvous"
    WAITING_AT_RENDEZVOUS = "waiting_at_rendezvous"


@dataclass
class UAV:
    uid: str
    carrier: Optional[str]         # UGV id, when docked
    state: str = UAVState.DOCKED
    target_ghost: Optional[str] = None
    pos_xy: Optional[Coord] = None      # only meaningful mid-flight
    recovery_ugv: Optional[str] = None
    recovery_point: Optional[Coord] = None
    depart_time: Optional[float] = None
    service_start: Optional[float] = None
    expected_service_time: float = 25.0  # placeholder fixed dwell, not battery-modeled


# ---------------------------------------------------------------------------
# Valid walks: enumerated once, offline, between priority-node pairs,
# matching TPBP exactly (Sec 2.1 / Background). Simple paths up to a hop
# bound T_hops standing in for the shared Time Period T.
# ---------------------------------------------------------------------------

def enumerate_valid_walks(G: nx.Graph, priorities: List[str], t_hops: int,
                          walks_per_pair: Optional[int] = None,
                          detour_walks: bool = True,
                          t_seconds: Optional[float] = None,
                          pos: Optional[Dict[str, Coord]] = None,
                          v_g: Optional[float] = None) -> List[List[str]]:
    """Valid walks between priority nodes. As in TPBP (Sec III-A) the end node
    may equal the start node, so closed walks s -> ... -> s are included.

    walks_per_pair=None -> every simple path (every simple cycle when s == f)
    of <= t_hops hops (exact, but the count grows ~5x for every 2 extra hops,
    far too large on a dense grid with t_hops ~ 20).
    walks_per_pair=K -> only the K shortest per ordered pair (still capped at
    t_hops hops). This approximates the walk set: report K in the paper.

    detour_walks=True additionally adds, for every ordered priority pair (s, f)
    and every node v, the walk shortest(s, v) + shortest(v, f) whenever it has
    <= t_hops hops. TPBP valid walks (Sec III-A) need not be simple paths, only
    shorter than the time period, and without these detours the nodes that lie
    off the simple paths between priority nodes (44 of 100 on M2) can never be
    visited at all.

    t_seconds (with pos and v_g) imposes the real time period T: a walk is valid
    only if its travel time (sum of Euclidean edge lengths / v_g) is <= T, which
    is L(p) <= T of Sec 2.1. t_hops is then only an enumeration cap, derived
    from T and the shortest edge."""
    if t_seconds is not None:
        min_edge = min(math.dist(pos[u], pos[v]) for u, v in G.edges())
        t_hops = int(t_seconds * v_g / min_edge + 1e-9)
    walks: List[List[str]] = []
    for s in priorities:
        for f in priorities:
            if s == f:
                walks.extend(_closed_walks(G, s, t_hops, walks_per_pair))
            elif walks_per_pair is None:
                for path in nx.all_simple_paths(G, s, f, cutoff=t_hops):
                    walks.append(path)
            else:
                # shortest_simple_paths yields in increasing hop count, so we
                # can stop as soon as a path exceeds the cutoff or K is reached
                for path in itertools.islice(nx.shortest_simple_paths(G, s, f), walks_per_pair):
                    if len(path) - 1 > t_hops:
                        break
                    walks.append(path)
    if detour_walks:
        sp = dict(nx.all_pairs_shortest_path(G))
        seen = {tuple(w) for w in walks}
        for s in priorities:
            for f in priorities:
                for v in G.nodes():
                    a, b = sp[s][v], sp[v][f]
                    if len(a) + len(b) - 2 > t_hops:
                        continue
                    w = a + b[1:]
                    if tuple(w) not in seen:
                        seen.add(tuple(w))
                        walks.append(w)
    if t_seconds is not None:
        def _time(w):
            return sum(math.dist(pos[a], pos[b]) for a, b in zip(w, w[1:])) / v_g
        walks = [w for w in walks if _time(w) <= t_seconds + 1e-9]
    return walks


def _closed_walks(G: nx.Graph, s: str, t_hops: int,
                  walks_per_pair: Optional[int]) -> List[List[str]]:
    """Closed walks s -> n -> ... -> s: a first step to a neighbour n, then a
    simple path from n back to s (so s -> n -> s out-and-backs are included)."""
    out: List[List[str]] = []
    for n in G.neighbors(s):
        if walks_per_pair is None:
            for p in nx.all_simple_paths(G, n, s, cutoff=t_hops - 1):
                out.append([s] + p)
        else:
            for p in itertools.islice(nx.shortest_simple_paths(G, n, s), walks_per_pair):
                if len(p) > t_hops:     # s + p has len(p) hops
                    break
                out.append([s] + p)
    if walks_per_pair is not None:
        out.sort(key=len)
        out = out[:walks_per_pair]
    return out


# ---------------------------------------------------------------------------
# Covering walk set (final-skeleton.tex Sec 7, Algorithm alg:cover).
# Direct walks: K fastest per priority pair within T (ranked by travel time).
# Off-path nodes U: ground nodes on no direct walk (tails, parallel roads).
# Detours: greedy from each off-path node's owner (nearest priority node).
# ---------------------------------------------------------------------------

def _travel_time(w: List[str], pos: Dict[str, Coord], v_g: float) -> float:
    return sum(math.dist(pos[a], pos[b]) for a, b in zip(w, w[1:])) / v_g


def build_cover_walks(G: nx.Graph, priorities: List[str], t_seconds: float,
                      pos: Dict[str, Coord], v_g: float, K: int = 10,
                      ghosts: Optional[Dict[str, Coord]] = None,
                      r_max: float = 70.0, m: int = 3,
                      pair_cover: int = 0,
                      ghost_per_start: bool = True,
                      ghost_per_pair: bool = False,
                      ghost_sweeps: bool = False,
                      all_fastest: bool = False,
                      disjoint: bool = False,
                      disjoint_covers: bool = False) -> Tuple[List[List[str]], dict]:
    """Returns (walks, stats). stats: T_star, n_direct, U, n_detours,
    n_ghost_walks, no_walk (ground nodes on no stored walk).

    pair_cover=r > 0: corridor cover. For every ordered pair (s, f) with
    d(s, f) <= T, greedily add sweeping walks s -> ... -> f until every node of
    the corridor {v : d(s, v) + d(v, f) <= T} lies on at least r of them."""
    T = t_seconds
    # K <= 0: rule K = |corridor(s, f)| per pair (local, set by the map)
    Gt = G.copy()
    for u, v in Gt.edges():
        Gt[u][v]["tt"] = math.dist(pos[u], pos[v]) / v_g
    # step 1: one Dijkstra per priority node
    dist, path = {}, {}
    for s in priorities:
        dist[s], path[s] = nx.single_source_dijkstra(Gt, s, cutoff=T, weight="tt")
    dS = {v: min(dist[s].get(v, math.inf) for s in priorities) for v in G.nodes()}
    owner = {v: min(priorities, key=lambda s: dist[s].get(v, math.inf)) for v in G.nodes()}
    T_star = max(dS[u] + Gt[u][v]["tt"] + dS[v] for u, v in Gt.edges())   # inf if some node is beyond T of every priority node
    tt = lambda w: _travel_time(w, pos, v_g)

    walks: List[List[str]] = []
    # step 2: direct walks, K fastest per ordered pair with d(s, f) <= T,
    # searched inside the pair's corridor {v : d(s, v) + d(v, f) <= T}, which
    # holds every node of every valid s -> f walk (so the search is exact)
    INF = math.inf
    for s in priorities:
        for f in priorities:
            if dist[s].get(f, INF) > T + 1e-9:
                continue
            corridor = [v for v in dist[s] if dist[s][v] + dist[f].get(v, INF) <= T + 1e-9]
            H = Gt.subgraph(corridor)
            Kp = K if K > 0 else len(corridor)
            if s == f:
                closed = []
                for n in H.neighbors(s):
                    for p in itertools.islice(nx.shortest_simple_paths(H, n, s, weight="tt"), Kp):
                        w = [s] + p
                        if tt(w) > T + 1e-9:
                            break
                        closed.append(w)
                closed.sort(key=tt)
                walks.extend(closed[:Kp])
            else:
                for p in itertools.islice(nx.shortest_simple_paths(H, s, f, weight="tt"), Kp):
                    if tt(p) > T + 1e-9:
                        break
                    walks.append(p)
    # step 2a (optional): every equally fastest route per ordered pair s != f
    if all_fastest:
        have = {tuple(w) for w in walks}
        for s in priorities:
            for f in priorities:
                if s != f and dist[s].get(f, math.inf) <= T + 1e-9:
                    for p in nx.all_shortest_paths(Gt, s, f, weight="tt"):
                        if tuple(p) not in have:
                            have.add(tuple(p)); walks.append(p)
    # step 2a' (optional): edge-disjoint fastest routes per ordered pair s != f:
    # fastest route, remove its edges, repeat while one fits in T (<= degree many)
    if disjoint:
        have = {tuple(w) for w in walks}
        for s in priorities:
            for f in priorities:
                if s == f or dist[s].get(f, math.inf) > T + 1e-9:
                    continue
                H = Gt.copy()
                while True:
                    try:
                        t_sf, p = nx.single_source_dijkstra(H, s, f, weight="tt")
                    except nx.NetworkXNoPath:
                        break
                    if t_sf > T + 1e-9:
                        break
                    if tuple(p) not in have:
                        have.add(tuple(p)); walks.append(p)
                    H.remove_edges_from(zip(p, p[1:]))
    # step 2c (optional): repeated corridor covers on disjoint streets. Per pair:
    # cover the corridor, remove the edges used, cover again on what is left,
    # until no walk fits in T (at most max-degree rounds).
    if disjoint_covers:
        for s in priorities:
            for f in priorities:
                if dist[s].get(f, math.inf) > T + 1e-9:
                    continue
                H = Gt.copy()
                while True:
                    ds, ps = nx.single_source_dijkstra(H, s, cutoff=T, weight="tt")
                    df, pf = nx.single_source_dijkstra(H, f, cutoff=T, weight="tt")
                    corridor = sorted(v for v in ds if v in df and ds[v] + df[v] <= T + 1e-9)
                    need = set(v for v in corridor if v not in (s, f))
                    round_walks = []
                    while need:
                        u = max(sorted(need), key=lambda x: ds[x] + df[x])
                        p = list(ps[u]); t_p = ds[u]; cur = u
                        while True:
                            left = need - set(p)
                            if not left:
                                break
                            dc, pc = nx.single_source_dijkstra(H, cur, cutoff=T - t_p, weight="tt")
                            best, best_score = None, 0.0
                            for x in sorted(left):
                                if x not in dc or dc[x] <= 0 or x not in df or t_p + dc[x] + df[x] > T + 1e-9:
                                    continue
                                sc = len(set(pc[x]) & left) / dc[x]
                                if sc > best_score:
                                    best, best_score = x, sc
                            if best is None:
                                break
                            p += pc[best][1:]; t_p += dc[best]; cur = best
                        p += list(reversed(pf[cur]))[1:]
                        round_walks.append(p)
                        need -= set(p)
                    if not round_walks:
                        break
                    walks.extend(round_walks)
                    for p in round_walks:
                        H.remove_edges_from(zip(p, p[1:]))
    # step 2b (optional): corridor cover per ordered pair
    n_corridor = 0
    if pair_cover > 0:
        for s in priorities:
            for f in priorities:
                if dist[s].get(f, math.inf) > T + 1e-9:
                    continue
                corridor = [v for v in G.nodes()
                            if dist[s].get(v, math.inf) + dist[f].get(v, math.inf) <= T + 1e-9]
                count = {v: 0 for v in corridor}
                for _ in range(len(corridor) * pair_cover):
                    need = {v for v in corridor if count[v] < pair_cover}
                    if not need:
                        break
                    u = max(sorted(need), key=lambda x: dist[s][x] + dist[f][x])
                    p = list(path[s][u])
                    t_p = dist[s][u]
                    cur = u
                    while True:
                        left = need - set(p)
                        if not left:
                            break
                        dc, pc = nx.single_source_dijkstra(Gt, cur, cutoff=T - t_p, weight="tt")
                        best, best_score = None, 0.0
                        for x in sorted(left):
                            if x not in dc or dc[x] <= 0 or t_p + dc[x] + dist[f][x] > T + 1e-9:
                                continue
                            score = len(set(pc[x]) & left) / dc[x]
                            if score > best_score:
                                best, best_score = x, score
                        if best is None:
                            break
                        p += pc[best][1:]
                        t_p += dc[best]
                        cur = best
                    p += list(reversed(path[f][cur]))[1:]
                    if len(p) < 2:
                        count[u] = pair_cover   # s == f == u: trivial, skip
                        continue
                    walks.append(p)
                    n_corridor += 1
                    for v in set(p):
                        if v in count:
                            count[v] += 1
    n_direct = len(walks)
    on_direct = set(n for w in walks for n in w)
    U = [v for v in G.nodes() if v not in on_direct]

    # step 4: detours, greedy
    uncovered = set(U)
    n_detours = 0
    gated: List[tuple] = []
    while uncovered:
        u = max(sorted(uncovered), key=lambda x: dS[x])
        A = owner[u]
        if 2 * dS[u] > T + 1e-9:   # T < T*: this node cannot be on any valid walk
            uncovered.discard(u)
            continue
        p = list(path[A][u])
        t_p = dist[A][u]
        cur = u
        while True:
            left = set(uncovered) - set(p)
            if not left:
                break
            dc, pc = nx.single_source_dijkstra(Gt, cur, cutoff=T - t_p, weight="tt")
            best, best_score = None, 0.0
            for x in sorted(left):
                if x not in dc or dc[x] <= 0:
                    continue
                if t_p + dc[x] + dS[x] > T + 1e-9:
                    continue
                new = len(set(pc[x]) & left)
                score = new / dc[x]
                if score > best_score:
                    best, best_score = x, score
            if best is None:
                break
            p += pc[best][1:]
            t_p += dc[best]
            cur = best
        # close, preferring the owner A
        if t_p + dist[A].get(cur, math.inf) <= T + 1e-9:
            back = list(reversed(path[A][cur]))
        else:
            B = owner[cur]
            back = list(reversed(path[B][cur]))
        p += back[1:]
        walks.append(p)
        gated.append(tuple(p))
        n_detours += 1
        uncovered -= set(p)

    # step 5: at least m walks through every launch set
    n_ghost = 0
    if ghosts and ghost_sweeps:
        # for every pair (s, f) and launch set L_h meeting its corridor: one sweep
        # s -> v (cheapest launch node) -> greedily most new corridor nodes per
        # second -> f, so the launch walk also does road work
        for h, hp in ghosts.items():
            L = sorted(v for v in G.nodes() if math.dist(pos[v], hp) <= r_max)
            for s in priorities:
                for f in priorities:
                    c = min(((dist[s].get(v, math.inf) + dist[f].get(v, math.inf), v) for v in L),
                            default=(math.inf, None))
                    if c[0] > T + 1e-9:
                        continue
                    corridor = {v for v in G.nodes()
                                if dist[s].get(v, math.inf) + dist[f].get(v, math.inf) <= T + 1e-9}
                    v0 = c[1]
                    p = list(path[s][v0]); t_p = dist[s][v0]; cur = v0
                    while True:
                        left = corridor - set(p)
                        if not left:
                            break
                        dc, pc = nx.single_source_dijkstra(Gt, cur, cutoff=T - t_p, weight="tt")
                        best, best_score = None, 0.0
                        for x in sorted(left):
                            if x not in dc or dc[x] <= 0 or t_p + dc[x] + dist[f][x] > T + 1e-9:
                                continue
                            sc = len(set(pc[x]) - set(p)) / dc[x]
                            if sc > best_score:
                                best, best_score = x, sc
                        if best is None:
                            break
                        p += pc[best][1:]; t_p += dc[best]; cur = best
                    p += list(reversed(path[f][cur]))[1:]
                    if len(p) > 1:
                        walks.append(p)
                        n_ghost += 1
    if ghosts and ghost_per_pair:
        # for every pair (s, f) whose corridor meets launch set L_h, one walk
        # s -> v -> f through L_h (cheapest v), unless the pair already has one
        for h, hp in ghosts.items():
            L = [v for v in G.nodes() if math.dist(pos[v], hp) <= r_max]
            Lset = set(L)
            for s in priorities:
                for f in priorities:
                    if any(w[0] == s and w[-1] == f and Lset.intersection(w) for w in walks):
                        continue
                    best = min(((dist[s].get(v, math.inf) + dist[f].get(v, math.inf), v) for v in sorted(L)),
                               default=(math.inf, None))
                    if best[0] > T + 1e-9:
                        continue
                    v = best[1]
                    w = list(path[s][v]) + list(reversed(path[f][v]))[1:]
                    if len(w) > 1:
                        walks.append(w)
                        n_ghost += 1
    elif ghosts and ghost_per_start:
        # m walks through every launch set FROM EVERY priority node s:
        # s -> v -> f for launch nodes v, ends f by time, cheapest first
        for h, hp in ghosts.items():
            L = [v for v in G.nodes() if math.dist(pos[v], hp) <= r_max]
            for s in priorities:
                have = sum(1 for w in walks if w[0] == s and any(v in L for v in w))
                cands = []
                for v in L:
                    for f in priorities:
                        t = dist[s].get(v, math.inf) + dist[f].get(v, math.inf)
                        if t <= T + 1e-9:
                            cands.append((t, v, f))
                cands.sort()
                used_v = set()
                for t, v, f in cands:
                    if have >= m:
                        break
                    if (v, f) in used_v:
                        continue
                    w = list(path[s][v]) + list(reversed(path[f][v]))[1:]
                    if len(w) > 1:
                        walks.append(w)
                        used_v.add((v, f))
                        have += 1
                        n_ghost += 1
    elif ghosts:
        for h, hp in ghosts.items():
            L = [v for v in G.nodes() if math.dist(pos[v], hp) <= r_max]
            have = sum(1 for w in walks if any(v in L for v in w))
            for v in sorted(L, key=lambda x: dS[x]):
                if have >= m:
                    break
                if 2 * dS[v] > T + 1e-9:
                    continue
                A = owner[v]
                w = list(path[A][v]) + list(reversed(path[A][v]))[1:]
                if len(w) > 1:
                    walks.append(w)
                    have += 1
                    n_ghost += 1

    # step 6: dedupe, keep order
    seen, out = set(), []
    for w in walks:
        if tuple(w) not in seen:
            seen.add(tuple(w))
            out.append(w)
    covered = set(n for w in out for n in w)
    stats = dict(T_star=T_star, n_direct=n_direct, n_corridor=n_corridor, U=len(U), n_detours=n_detours,
                 n_ghost_walks=n_ghost, walks=len(out),
                 no_walk=sum(1 for v in G.nodes() if v not in covered),
                 gated=gated)
    return out, stats


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class Sim:
    def __init__(self, layout: Layout, n_ugv: int, n_uav: int,
                 variant: VariantConfig, weights: Weights = None,
                 t_hops: int = 8, seed: int = 0, dt: float = 1.0,
                 events: EventConfig = None, record_traj: bool = False,
                 traj_stride: int = 5,
                 valid_walks: Optional[List[List[str]]] = None,
                 walks_per_pair: Optional[int] = None,
                 track_node_idleness: bool = False,
                 t_seconds: Optional[float] = None,
                 gated_walks: Optional[set] = None):
        self.layout = layout
        self.G = layout.G
        self.pos = layout.pos
        self.ghosts = dict(layout.ghosts)
        self.priorities = list(layout.priorities)
        self.variant = variant
        self.w = weights or Weights()
        self.dt = dt
        self.rng = random.Random(seed)
        self.events_cfg = events or EventConfig()
        self.active_events: Dict[str, EmergencyEvent] = {}
        self.resolved_events: List[EmergencyEvent] = []
        self._event_seq = 0

        self.edge_len = {(u, v): math.dist(self.pos[u], self.pos[v])
                          for u, v in self.G.edges()}
        for (u, v), l in list(self.edge_len.items()):
            self.edge_len[(v, u)] = l

        # Pass valid_walks in to reuse one precomputed list across many runs
        # (the list depends only on the graph, priorities and t_hops).
        if valid_walks is None:
            valid_walks = enumerate_valid_walks(self.G, self.priorities, t_hops,
                                                walks_per_pair=walks_per_pair,
                                                t_seconds=t_seconds, pos=self.pos,
                                                v_g=self.w.v_g)
        self.valid_walks = valid_walks
        # Gated walks (covering-set detours, final-skeleton Sec 7): not in the
        # normal argmax; only a starvation claim can send a UGV on one.
        self._gated = set(gated_walks or ())
        # index by start node (keeps the original order within each start)
        self._walks_by_start: Dict[str, List[List[str]]] = {}
        self._gated_by_start: Dict[str, List[List[str]]] = {}
        for w in self.valid_walks:
            if tuple(w) in self._gated:
                self._gated_by_start.setdefault(w[0], []).append(w)
                continue
            self._walks_by_start.setdefault(w[0], []).append(w)
        # the graph is static: all-pairs hop distances once, not a BFS per walk
        self._sp = dict(nx.all_pairs_shortest_path_length(self.G))

        # Starvation claims (final-skeleton Sec 7.4), only when claim_cap is set
        self.claim_queue: List[str] = []
        self.claims: Dict[str, dict] = {}   # UGV id -> {node, walk, target}
        self.n_claims_made = 0
        if self.w.claim_cap is not None:
            Gt = self.G.copy()
            for u, v in Gt.edges():
                Gt[u][v]["tt"] = self.edge_len[(u, v)] / self.w.v_g
            self._ptime = {s: nx.single_source_dijkstra_path_length(Gt, s, weight="tt")
                           for s in self.priorities}
            self._GS = nx.Graph()
            self._GS.add_nodes_from(self.priorities)
            for s in self.priorities:
                for f in self.priorities:
                    if s != f and self._ptime[s].get(f, math.inf) <= self.w.T + 1e-9:
                        self._GS.add_edge(s, f)
            self._cover_walk_of: Dict[str, List[str]] = {}
            # claims only for off-path nodes: those on no ungated walk
            on_free = set(v for wk in self.valid_walks if tuple(wk) not in self._gated for v in wk)
            self._claim_nodes = set(v for wk in self.valid_walks for v in wk) - on_free
            # claim_cap = -1: rule N_c = ceil(n_UGV * |U| / |V_g|), fleet share = load share
            if self.w.claim_cap == -1:
                self.w.claim_cap = math.ceil(n_ugv * len(self._claim_nodes) / self.G.number_of_nodes())
            self._hop_walk: Dict[Tuple[str, str], List[str]] = {}
            for wk in self.valid_walks:
                t = self._walk_time(wk)
                for v in set(wk):
                    cur = self._cover_walk_of.get(v)
                    if cur is None or t < self._walk_time(cur):
                        self._cover_walk_of[v] = wk
                key = (wk[0], wk[-1])
                cur = self._hop_walk.get(key)
                if wk[0] != wk[-1] and tuple(wk) not in self._gated and (cur is None or t < self._walk_time(cur)):
                    self._hop_walk[key] = wk

        self.t = 0.0
        self.last_visit: Dict[str, float] = {n: 0.0 for n in self.G.nodes()}
        self.last_visit.update({h: 0.0 for h in self.ghosts})
        self.occupied_since: Dict[str, Optional[float]] = {n: None for n in
                                                             list(self.G.nodes()) + list(self.ghosts)}
        self.claimed_ghosts: Dict[str, str] = {}      # ghost -> UAV id (launch claim, Q-)
        self.recovery_claims: Dict[str, str] = {}      # ghost -> UGV id (Q+, credit-only, see Sec 2.4

        self.ugvs: List[UGV] = []
        self.uavs: List[UAV] = []
        self._init_fleet(n_ugv, n_uav)

        # fixed-launch-node ablation: pick a static subset of ground nodes
        # once, used instead of the live radius check
        self.fixed_launch_nodes = self._pick_fixed_launch_nodes()

        # Optional per-node bookkeeping (off by default so batch runs cost
        # nothing extra): running time-average and peak idleness of every node.
        self._track_nodes = track_node_idleness
        self._idle_nodes: List[str] = list(self.G.nodes()) + list(self.ghosts)
        self._idle_sum: Dict[str, float] = {n: 0.0 for n in self._idle_nodes}
        self._idle_max: Dict[str, float] = {n: 0.0 for n in self._idle_nodes}
        self._idle_steps = 0

        self.metrics_log: List[dict] = []
        self.record_traj = record_traj
        self.traj_stride = traj_stride
        self.traj_log: List[dict] = []

    # -- setup ---------------------------------------------------------

    def _init_fleet(self, n_ugv: int, n_uav: int):
        # TPBP (Sec III-A): agents start on distinct Priority Nodes, no two on
        # the same node until every Priority Node is covered.
        pri = list(self.priorities)
        if n_ugv <= len(pri):
            start_nodes = self.rng.sample(pri, n_ugv)
        else:
            start_nodes = self.rng.sample(pri, len(pri)) + \
                [self.rng.choice(pri) for _ in range(n_ugv - len(pri))]
        for i, n0 in enumerate(start_nodes):
            self.ugvs.append(UGV(uid=f"g{i}", pos_node=n0))
        if not self.variant.ugv_only:
            for i in range(n_uav):
                carrier = self.ugvs[i % len(self.ugvs)]
                uav = UAV(uid=f"a{i}", carrier=carrier.uid)
                carrier.docked_uav = uav.uid if carrier.docked_uav is None else carrier.docked_uav
                self.uavs.append(uav)

    def _pick_fixed_launch_nodes(self) -> List[str]:
        # one fixed launch node per ghost: its nearest ground node, chosen
        # once, offline -- the thing the radius check is compared against
        out = []
        for h, hp in self.ghosts.items():
            nearest = min(self.G.nodes(), key=lambda n: math.dist(self.pos[n], hp))
            out.append(nearest)
        return out

    # -- geometry / idleness --------------------------------------------

    def ugv_xy(self, g: UGV) -> Coord:
        if g.walk_idx == 0 or g.walk_idx >= len(g.walk):
            return self.pos[g.pos_node]
        u = g.walk[g.walk_idx - 1]
        v = g.walk[g.walk_idx]
        total = self.edge_len[(u, v)]
        frac = min(g.edge_progress / total, 1.0) if total > 0 else 1.0
        ux, uy = self.pos[u]
        vx, vy = self.pos[v]
        return (ux + frac * (vx - ux), uy + frac * (vy - uy))

    def uav_xy(self, uav: UAV) -> Coord:
        if uav.state == UAVState.DOCKED:
            carrier = self._ugv(uav.carrier)
            return self.ugv_xy(carrier) if carrier else (0.0, 0.0)
        return uav.pos_xy if uav.pos_xy is not None else (0.0, 0.0)

    def idleness(self, node: str) -> float:
        if self.occupied_since.get(node) is not None:
            return 0.0
        return self.t - self.last_visit[node]

    def node_idleness(self) -> Dict[str, float]:
        """Current idleness (seconds since last visit) of every node: ground
        nodes (including priority nodes) and ghost nodes, keyed by node id."""
        return {n: self.idleness(n) for n in self._idle_nodes}

    def node_idleness_stats(self) -> Dict[str, Dict[str, float]]:
        """Per-node time-average and peak idleness so far. Needs
        Sim(..., track_node_idleness=True)."""
        if not self._track_nodes:
            raise RuntimeError("construct Sim with track_node_idleness=True")
        steps = max(self._idle_steps, 1)
        return {n: {"avg": self._idle_sum[n] / steps, "max": self._idle_max[n]}
                for n in self._idle_nodes}

    def visit(self, node: str):
        self.last_visit[node] = self.t

    # -- reward: base TPBP + ghost credit + recovery credit --------------

    def _others_of(self, g: UGV) -> set:
        others = set()
        for og in self.ugvs:
            if og is not g and og.walk:
                others.update(og.walk[og.walk_idx:])
        return others

    def walk_reward(self, g: UGV, walk: List[str],
                    unclaimed_priorities: Optional[List[str]] = None) -> Tuple[float, Optional[str], Optional[str]]:
        """Returns (reward, credited_launch_ghost_or_None, credited_recovery_ghost_or_None)."""
        w = self.w
        # TPBP Term 1: idleness of the nodes COVERED by the walk, so a node that a
        # (non-simple) walk passes twice counts once
        covered = set(walk)
        if getattr(self.w, "discount_others", 0.0) > 0:
            others = getattr(self, "_others_cache", None)
            if others is None:  # outside a decision: build on demand
                others = self._others_of(g)
            term1 = sum(self.idleness(v) * (1.0 - self.w.discount_others if v in others else 1.0) for v in covered) * w.c1
        else:
            term1 = sum(self.idleness(v) for v in covered) * w.c1
        # Y(t) = S \ X(t): priority nodes not claimed by other agents
        if unclaimed_priorities is None:
            in_use = self._destinations_in_use(exclude=g)
            unclaimed_priorities = [s for s in self.priorities if s not in in_use]
        Y = set(unclaimed_priorities)
        # TPBP: Term 2 sums over s_j in p ∩ Y; Term 3 uses dist(p, s_j), the
        # shortest distance to s_j from ANY node of the walk
        term2 = sum(max(self.idleness(s) - w.T, 0.0) for s in covered if s in Y) * w.c2   # [I_s - T]+, T in seconds
        term3 = sum(min(self._sp[v][sj] for v in walk) for sj in unclaimed_priorities) * w.c3
        term4 = len(walk) * w.c4
        reward = term1 + term2 - term3 - term4

        credited_launch = None
        credited_recovery = None

        if self.variant.ghost_credit and g.docked_uav is not None:
            for h, hp in self.ghosts.items():
                if h in self.claimed_ghosts:
                    continue
                if self.idleness(h) < w.i_min:
                    continue
                launch_set_hit = any(math.dist(self.pos[n], hp) <= w.r_max for n in walk)
                if launch_set_hit:
                    ihat = self.idleness(h)
                    if getattr(w, "ghost_units_seconds", False):
                        t_h = sum(self.edge_len[(walk[i], walk[i + 1])] for i in range(len(walk) - 1)) / w.v_g
                    else:
                        t_h = len(walk)  # remaining walk time proxy, hop-based
                    reward += w.c5 * (ihat + w.c6 * max(ihat - t_h, 0.0))
                    credited_launch = h
                    break  # credit at most one ghost per walk, nearest match first

        if self.variant.recovery_credit and g.is_free:
            for uav in self.uavs:
                if uav.state != UAVState.SERVICING or uav.target_ghost is None:
                    continue
                h = uav.target_ghost
                if h in self.recovery_claims and self.recovery_claims[h] != g.uid:
                    continue
                hp = self.ghosts[h] if h in self.ghosts else self.pos[h]
                near = any(math.dist(self.pos[n], hp) <= self.w.r_max for n in walk)
                if near:
                    t_rem = max(uav.expected_service_time -
                                (self.t - (uav.service_start or self.t)), 0.1)
                    reward += w.c7 * (1.0 / t_rem)
                    credited_recovery = h
                    break

        if self.events_cfg.enabled:
            for ev in self.active_events.values():
                if ev.claimed_by is not None:
                    continue
                if ev.node in walk:
                    reward += w.c9 * ev.severity
                    break  # at most one event credited per walk

        return reward, credited_launch, credited_recovery

    def _destinations_in_use(self, exclude: Optional[UGV] = None) -> set:
        """X(t): destination priority nodes of the OTHER agents (TPBP Sec III-B)."""
        return {g.walk[-1] for g in self.ugvs if g.walk and g is not exclude}

    def _near_ghost_nodes(self, h: str) -> set:
        """Ground nodes within r_max of ghost h (cached; the graph is static)."""
        cache = self.__dict__.setdefault("_near_cache", {})
        if h not in cache:
            hp = self.ghosts[h]
            cache[h] = {n for n in self.G.nodes() if math.dist(self.pos[n], hp) <= self.w.r_max}
        return cache[h]

    # -- starvation claims (final-skeleton Sec 7.4) ------------------------

    def _walk_time(self, w: List[str]) -> float:
        return sum(self.edge_len[(a, b)] for a, b in zip(w, w[1:])) / self.w.v_g

    def _remaining_time(self, g: UGV) -> float:
        if not g.walk or g.walk_idx >= len(g.walk):
            return 0.0
        rest = self._walk_time(g.walk[g.walk_idx - 1:])
        return max(rest - g.edge_progress / self.w.v_g, 0.0)

    def _update_claims(self):
        thr = self.w.starve_idleness
        if self.w.claim_cap is None or thr is None:
            return
        on_claimed = set()
        for c in self.claims.values():
            on_claimed.update(c["walk"])
        # drop queued nodes that were visited meanwhile
        self.claim_queue = [v for v in self.claim_queue if self.idleness(v) > thr]
        queued = set(self.claim_queue)
        for v in sorted(self.G.nodes(), key=self.idleness, reverse=True):
            if self.idleness(v) <= thr:
                break
            if v not in queued and v not in on_claimed and v in self._claim_nodes:
                self.claim_queue.append(v)
                queued.add(v)
        while len(self.claims) < self.w.claim_cap and self.claim_queue:
            v = self.claim_queue.pop(0)
            pv = self._cover_walk_of[v]
            sv = pv[0]
            best, best_t = None, math.inf
            for g in self.ugvs:
                if g.uid in self.claims or g.reserved_for is not None or getattr(g, "_ev", None):
                    continue
                end = g.walk[-1] if g.walk and g.walk_idx < len(g.walk) else g.pos_node
                t = self._remaining_time(g) + self._ptime[sv].get(end, math.inf)
                if t < best_t:
                    best, best_t = g, t
            if best is None:
                self.claim_queue.insert(0, v)
                break
            self.claims[best.uid] = {"node": v, "walk": pv, "target": sv}
            self.n_claims_made += 1

    def _claim_walk(self, g: UGV) -> Optional[List[str]]:
        """Next walk for a UGV holding a claim, ignoring Y(t); None = release."""
        c = self.claims.get(g.uid)
        if c is None:
            return None
        here = g.pos_node
        if here == c["target"]:
            del self.claims[g.uid]
            return c["walk"]
        if here not in self._GS or not nx.has_path(self._GS, here, c["target"]):
            del self.claims[g.uid]
            return None
        nxt = nx.shortest_path(self._GS, here, c["target"])[1]
        hop = self._hop_walk.get((here, nxt))
        if hop is None:
            del self.claims[g.uid]
            return None
        return hop

    def select_walk(self, g: UGV):
        if self.w.claim_cap is not None and g.uid in self.claims:
            cw = self._claim_walk(g)
            if cw is not None:
                g.walk = cw
                g.walk_idx = 1
                g.edge_progress = 0.0
                return
        candidates = self._walks_by_start.get(g.pos_node)
        # gate opens: a gated detour from here becomes a candidate once one of
        # its nodes is past the starvation threshold
        thr_g = self.w.gate_idleness if self.w.gate_idleness is not None else self.w.starve_idleness
        if thr_g is not None and self._gated_by_start.get(g.pos_node):
            opened = [w for w in self._gated_by_start[g.pos_node]
                      if any(self.idleness(v) > thr_g for v in w)]
            if opened:
                candidates = (candidates or []) + opened
        if not candidates:
            candidates = self.valid_walks  # fallback: any walk, teleport-free not guaranteed
        # same for every candidate walk in this call, so compute once
        in_use = self._destinations_in_use(exclude=g)      # X(t)
        unclaimed = [s for s in self.priorities if s not in in_use]   # Y(t)
        # TPBP: only walks ending in Y(t) are eligible (so two agents never
        # head for the same priority node); the agent's own node is in Y.
        eligible = [w for w in candidates if w[-1] in unclaimed]
        if eligible:
            candidates = eligible
        # Starvation guard (extension, off by default): TPBP's sum-of-idleness
        # reward can ignore a stale node that only short walks reach.
        thr = self.w.starve_idleness
        if thr is not None:   # guard for nodes on normal walks; claims handle off-path nodes
            for v in sorted(self.G.nodes(), key=self.idleness, reverse=True):
                if self.idleness(v) <= thr:
                    break
                hit = [w for w in candidates if v in w]
                if hit:
                    candidates = hit
                    break
        # Ghost starvation guard (extension, off by default): a ghost is only
        # serviced when a UGV passes within r_max of it, so when one has gone
        # stale, prefer walks that pass within r_max of the stalest such ghost.
        gthr = self.w.ghost_starve_idleness
        if (gthr is not None and not self.variant.ugv_only and self.variant.radius_launch
                and g.docked_uav is not None):
            for h in sorted(self.ghosts, key=self.idleness, reverse=True):
                if self.idleness(h) <= gthr:
                    break
                near = self._near_ghost_nodes(h)
                hit = [w for w in candidates if near.intersection(w)]
                if hit:
                    candidates = hit
                    break
        best, best_r, best_launch, best_recovery = None, -1e18, None, None
        # O_g(t) is built once per decision, not once per candidate walk
        self._others_cache = self._others_of(g) if getattr(self.w, "discount_others", 0.0) > 0 else None
        for w in candidates:
            r, launch_h, recovery_h = self.walk_reward(g, w, unclaimed)
            if r > best_r:
                best, best_r, best_launch, best_recovery = w, r, launch_h, recovery_h
        self._others_cache = None
        if best is None:
            return
        g.walk = best
        g.walk_idx = 1
        g.edge_progress = 0.0
        # NOTE: best_launch/best_recovery are reward-shaping signals only.
        # Do not register persistent claims here -- a walk can pick up a
        # closer launch opportunity mid-route and never reach this ghost,
        # which would orphan a "pending" claim forever (bug: see M1 run,
        # ghost h2 stuck "pending" permanently every seed). Real launch
        # claims are registered in check_launch(); recovery claims aren't
        # needed as committed state since _assign_recovery doesn't gate on
        # them.

    # -- launch condition (Sec 3.2) --------------------------------------

    def check_launch(self, g: UGV):
        if g.docked_uav is None or getattr(g, "_ev", None) in ("to_event", "handling", "to_launch"):
            return
        uav = self._uav(g.docked_uav)
        if uav is None or uav.state != UAVState.DOCKED or getattr(uav, "_frozen", False):
            return

        eligible = [h for h in self.ghosts if h not in self.claimed_ghosts
                    or self.claimed_ghosts[h] == "pending"]
        gxy = self.ugv_xy(g)

        if self.variant.radius_launch:
            in_range = [h for h in eligible if math.dist(gxy, self.ghosts[h]) <= self.w.r_max]
        else:
            # fixed launch nodes: only fires if the UGV is AT one of the
            # static launch nodes (Tier 1 ablation, Sec "Scope"/5.1 comparison)
            in_range = []
            if g.pos_node in self.fixed_launch_nodes:
                idx = self.fixed_launch_nodes.index(g.pos_node)
                h = list(self.ghosts.keys())[idx] if idx < len(self.ghosts) else None
                if h in eligible:
                    in_range = [h]

        if not in_range and self.w.ghost_cap_idleness is not None and self.variant.radius_launch:
            over = [h for h in eligible if self.idleness(h) > self.w.ghost_cap_idleness]
            # cap launch flight limit (same bound as recovery): only from a carrier within r_rec
            # of the node, and only by the docked UAV closest to it
            R = self.w.recovery_radius_m
            over = [h for h in over if (R is None or math.dist(gxy, self.ghosts[h]) <= R)
                    and self._closest_docked_carrier(h) is g]
            if over:
                in_range = [max(over, key=self.idleness)]
        if not in_range:
            return

        # tie-break: nearest, then highest idleness (genswarm-extension.tex Sec 3.2)
        h_star = min(in_range, key=lambda h: (math.dist(gxy, self.ghosts[h]), -self.idleness(h)))
        self.claimed_ghosts[h_star] = uav.uid
        uav.state = UAVState.TO_GHOST
        uav.target_ghost = h_star
        uav.pos_xy = gxy
        g.docked_uav = None

    def _closest_docked_carrier(self, h: str):
        """UGV carrying a launchable docked UAV, closest (straight line) to air-only node h"""
        best, bd = None, math.inf
        for c in self.ugvs:
            if c.docked_uav is None or getattr(c, "_ev", None) in ("to_event", "handling", "to_launch"):
                continue
            u = self._uav(c.docked_uav)
            if u is None or u.state != UAVState.DOCKED or getattr(u, "_frozen", False):
                continue
            d = math.dist(self.ugv_xy(c), self.ghosts[h])
            if d < bd:
                best, bd = c, d
        return best

    # -- UAV service cycle (Sec 3.3) --------------------------------------

    def step_uav(self, uav: UAV):
        w = self.w
        if uav.state == UAVState.TO_GHOST:
            target = self.ghosts[uav.target_ghost]
            uav.pos_xy = self._move_toward(uav.pos_xy, target, w.s_a)
            if math.dist(uav.pos_xy, target) <= w.eps_l:
                uav.state = UAVState.SERVICING
                uav.service_start = self.t
                self.occupied_since[uav.target_ghost] = self.t
                self.visit(uav.target_ghost)

        elif uav.state == UAVState.SERVICING:
            # assign a recovery UGV exactly once per cycle, after the service dwell
            if uav.recovery_ugv is None and \
                    self.t - uav.service_start >= uav.expected_service_time:
                self._assign_recovery(uav)

        elif uav.state == UAVState.TO_RENDEZVOUS:
            rg = self._ugv(uav.recovery_ugv)
            if rg is None:
                return
            if not self.variant.landing_candidate_recovery:
                # nearest-UGV ablation: no landing point, no UGV waiting; the
                # UAV pursues the UGV's current position until co-located
                uav.pos_xy = self._move_toward(uav.pos_xy, self.ugv_xy(rg), w.s_a)
                if math.dist(uav.pos_xy, self.ugv_xy(rg)) <= w.eps_l:
                    self._dock(uav, rg)
                return
            uav.pos_xy = self._move_toward(uav.pos_xy, uav.recovery_point, w.s_a)
            if math.dist(uav.pos_xy, uav.recovery_point) <= w.eps_l:
                if self._ugv_at_landing(rg, uav):
                    self._dock(uav, rg)
                else:
                    # UAV got there first: wait, holding the node at idleness 0
                    # (Definition 1, continuous occupancy)
                    uav.state = UAVState.WAITING_AT_RENDEZVOUS
                    node = self._node_at_point(uav.recovery_point)
                    if node is not None and self.occupied_since.get(node) is None:
                        self.occupied_since[node] = self.t

        elif uav.state == "to_event":
            tgt = self._ev_xy
            old = uav.pos_xy
            uav.pos_xy = self._move_toward(uav.pos_xy, tgt, w.s_a)
            self.ev_stats["D_resp"] += math.dist(old, uav.pos_xy)
            if math.dist(uav.pos_xy, tgt) <= w.eps_l:
                if self._ev_state["phase"] == "done":
                    self._ev_uav_release(uav)     # arrived after clearing: recover as usual
                else:
                    uav.state = "event_handling"
                    self._ev_arrive(uav.uid)

        elif uav.state == UAVState.WAITING_AT_RENDEZVOUS:
            rg = self._ugv(uav.recovery_ugv)
            if rg is not None and self._ugv_at_landing(rg, uav):
                self._dock(uav, rg)

    def _ugv_at_landing(self, g: UGV, uav: UAV) -> bool:
        """Co-location at the landing node: the UGV stands on it and the UAV
        is within eps_l (Sec 3.3 step 5)."""
        return (uav.recovery_point is not None
                and g.edge_progress == 0.0
                and self.pos[g.pos_node] == uav.recovery_point
                and math.dist(uav.pos_xy, self.ugv_xy(g)) <= self.w.eps_l)

    def _assign_recovery(self, uav: UAV):
        """Sec 3.3 step 3: pick (UGV, landing node) minimising TL = max(t_u, t_a)
        over empty, unreserved UGVs and their C landing candidates."""
        w = self.w
        candidates = [g for g in self.ugvs if g.is_free and getattr(g, "_ev", None) is None]
        if not candidates:
            return  # retry next step; the UAV keeps servicing (idleness held at 0)

        h = self.ghosts[uav.target_ghost] if uav.target_ghost in self.ghosts else self.pos[uav.target_ghost]
        best = None   # (TL, g, point, t_a)
        R = w.recovery_radius_m
        for g in candidates:
            if R is not None and math.dist(self.ugv_xy(g), h) > R:
                continue
            if self.variant.landing_candidate_recovery:
                points = [(self.pos[node], d / w.v_g)
                          for node, d in self._landing_points(g, w.landing_candidates)
                          if R is None or math.dist(self.pos[node], h) <= R]
            else:
                # nearest-UGV ablation: the UGV's position right now
                points = [(self.ugv_xy(g), 0.0)]
            for p, t_u in points:
                t_a = math.dist(h, p) / w.s_a
                TL = max(t_u, t_a)
                if best is None or TL < best[0]:
                    best = (TL, g, p, t_a)
        if best is None:
            return
        TL, g, p, t_a = best

        g.reserved_for = uav.uid
        g.reserved_point = p
        uav.recovery_ugv = g.uid
        uav.recovery_point = p
        if self.variant.landing_candidate_recovery:
            uav.depart_time = self.t + max(TL - t_a, 0.0)      # t_dep = TL - t_a
        else:
            uav.depart_time = self.t                           # leaves at once
        uav._depart_at = uav.depart_time
        if uav.target_ghost in self.recovery_claims:
            del self.recovery_claims[uav.target_ghost]

    def _hops_from_ghost(self, h):
        cache = self.__dict__.setdefault("_hop_cache", {})
        if h not in cache:
            hp = self.ghosts[h]
            n0 = min(self.G.nodes(), key=lambda n: math.dist(self.pos[n], hp))
            cache[h] = dict(nx.single_source_shortest_path_length(self.G, n0))
        return cache[h]

    def _landing_points(self, g: UGV, c: int) -> List[Tuple[str, float]]:
        """Landing candidates for UGV g as (ground node, path distance from the
        UGV's current position). Targets are at path distances l/3, 2l/3, l of
        the remaining planned path (Sec 3.3 step 3), snapped to the first node
        at or beyond each; duplicates removed. l = 0 -> the UGV's own node."""
        if not g.walk or g.walk_idx >= len(g.walk):
            return [(g.pos_node, 0.0)]
        nxt = g.walk[g.walk_idx:]
        prev = g.walk[g.walk_idx - 1]
        d0 = max(self.edge_len[(prev, nxt[0])] - g.edge_progress, 0.0)
        cum = [d0]
        for a, b in zip(nxt[:-1], nxt[1:]):
            cum.append(cum[-1] + self.edge_len[(a, b)])
        total = cum[-1]
        targets = [total / 3, 2 * total / 3, total][:c]
        out, seen = [], set()
        for td in targets:
            idx = next((i for i, x in enumerate(cum) if x >= td - 1e-9), len(cum) - 1)
            node = nxt[idx]
            if node not in seen:
                seen.add(node)
                out.append((node, cum[idx]))
        return out

    def _dock(self, uav: UAV, g: UGV):
        # docking ends any waiting at the landing node: idleness resumes from 0
        node = self._node_at_point(uav.recovery_point) if uav.recovery_point else None
        if node is not None and self.occupied_since.get(node) is not None:
            self.occupied_since[node] = None
            self.visit(node)
        if uav.target_ghost in self.claimed_ghosts:
            del self.claimed_ghosts[uav.target_ghost]
        uav.state = UAVState.DOCKED
        uav.carrier = g.uid
        uav.target_ghost = None
        uav.recovery_ugv = None
        uav.recovery_point = None
        uav._depart_at = None
        g.docked_uav = uav.uid
        g.reserved_for = None
        g.reserved_point = None

    def _move_toward(self, cur: Coord, target: Coord, speed: float) -> Coord:
        d = math.dist(cur, target)
        if d <= speed * self.dt:
            return target
        frac = (speed * self.dt) / d
        return (cur[0] + frac * (target[0] - cur[0]), cur[1] + frac * (target[1] - cur[1]))

    def _node_at_point(self, p: Coord) -> Optional[str]:
        for n, xy in self.pos.items():
            if math.dist(xy, p) < 1e-6:
                return n
        return None

    def _uav(self, uid: Optional[str]) -> Optional[UAV]:
        return next((a for a in self.uavs if a.uid == uid), None) if uid else None

    def _ugv(self, uid: Optional[str]) -> Optional[UGV]:
        return next((g for g in self.ugvs if g.uid == uid), None) if uid else None

    # -- per-step UGV motion ----------------------------------------------

    def step_ugv(self, g: UGV):
        st = getattr(g, "_ev", None)
        if st == "handling":
            return
        if st in ("to_event", "to_prio", "to_launch") and (not g.walk or g.walk_idx >= len(g.walk)):
            self._ev_ugv_arrived(g)
            return
        # nearest-UGV baseline: once chosen to recover a UAV the UGV STOPS where it is
        # until the UAV has landed on it (so the UAV's flight stays within the radius)
        if (not self.variant.landing_candidate_recovery) and g.reserved_for is not None:
            return
        # Sec 3.3 steps 3-4: a UGV reserved for a rendezvous follows its walk to
        # the landing node and waits there until the UAV arrives; the node is
        # held at idleness 0 while it waits (Definition 1).
        if (g.reserved_point is not None and self.variant.landing_candidate_recovery
                and g.edge_progress == 0.0 and self.pos[g.pos_node] == g.reserved_point):
            if self.occupied_since.get(g.pos_node) is None:
                self.occupied_since[g.pos_node] = self.t
            return
        if not g.walk or g.walk_idx >= len(g.walk):
            self.select_walk(g)
            return
        u = g.walk[g.walk_idx - 1]
        v = g.walk[g.walk_idx]
        edge_total = self.edge_len[(u, v)]
        g.edge_progress += self.w.v_g * self.dt
        if g.edge_progress >= edge_total:
            g.pos_node = v
            g.edge_progress = 0.0
            self.visit(v)
            g.walk_idx += 1
            self._maybe_claim_event(g, v)

    def _maybe_claim_event(self, g: UGV, node: str):
        for ev in self.active_events.values():
            if ev.node == node and ev.claimed_by is None:
                ev.claimed_by = g.uid
                ev.service_start = self.t
                return

    def _spawn_events(self):
        if not self.events_cfg.enabled:
            return
        if self.events_cfg.mode == "dispatch":
            self._dispatch_step()
            return
        lam = self.events_cfg.rate_per_hour * (self.dt / 3600.0)
        if self.rng.random() >= lam:
            return
        candidates = [n for n in self.G.nodes() if n not in self.ghosts]
        if not candidates:
            return
        node = self.rng.choice(candidates)
        severity = self.rng.randint(self.events_cfg.severity_min, self.events_cfg.severity_max)
        self._event_seq += 1
        eid = f"e{self._event_seq}"
        self.active_events[eid] = EmergencyEvent(eid=eid, node=node, severity=severity, spawn_t=self.t)

    def _resolve_events(self):
        if not self.events_cfg.enabled or self.events_cfg.mode == "dispatch":
            return
        done = []
        for eid, ev in self.active_events.items():
            if ev.claimed_by is None or ev.service_start is None:
                continue
            dwell = self.events_cfg.dwell_per_severity * ev.severity
            if self.t - ev.service_start >= dwell:
                ev.resolved_t = self.t
                done.append(eid)
        for eid in done:
            ev = self.active_events.pop(eid)
            self.resolved_events.append(ev)

    # -- emergency events, final-skeleton Sec 5 (mode="dispatch") ----------

    def _ugv_dist(self, g: UGV, target: str) -> Tuple[float, List[str]]:
        """Path length (m) and node path from g's current position to target."""
        if g.walk and 0 < g.walk_idx < len(g.walk) and g.edge_progress > 0.0:
            prev, nxt = g.walk[g.walk_idx - 1], g.walk[g.walk_idx]
            d0 = self.edge_len[(prev, nxt)] - g.edge_progress
            start = nxt
        else:
            prev, d0, start = None, 0.0, g.pos_node
        dist, path = nx.single_source_dijkstra(self.G, start, target,
                                               weight=lambda a, b, _: self.edge_len[(a, b)])
        return d0 + dist, ([prev] + path if prev is not None else path)

    def _ev_route(self, g: UGV, target: str) -> float:
        d, path = self._ugv_dist(g, target)
        if path[0] != g.pos_node and g.edge_progress == 0.0:
            path = [g.pos_node] + path
        g.walk = path
        g.walk_idx = 1
        if g.edge_progress == 0.0 and len(path) == 1:
            g.walk_idx = 1   # already there: walk exhausted
        return d

    def _dispatch_step(self):
        cfg = self.events_cfg
        ev = getattr(self, "_ev_state", None)
        if ev is None:
            if self.t < cfg.onset:
                return
            if cfg.freeze_for is not None:
                self._fz_start()
            else:
                self._ev_start()
            return
        if ev.get("freeze"):
            self._fz_step()
            return
        if ev["phase"] != "done":
            self._ev_fill()                       # response is mandatory: keep filling open slots
        if ev["phase"] == "active":
            # work rule: W(sigma) = (q_g + q_a) * h_e robot-seconds; every responder
            # on site does one robot-second per second
            ev["work"] += ev["present"] * self.dt
            if ev["work"] >= ev["W"] - 1e-9:
                self._ev_finish()

    def _ev_start(self):
        cfg = self.events_cfg
        v, sig = cfg.node, cfg.severity
        air = v in self.ghosts
        self._ev_xy = self.ghosts[v] if air else self.pos[v]
        q_g, q_a = event_quota(sig, air, cfg)
        Ge = [g for g in self.ugvs if g.reserved_for is None]
        Ae = [a for a in self.uavs if a.state == UAVState.DOCKED]
        ugv_t = {} if air else {g.uid: self._ugv_dist(g, v)[0] / self.w.v_g for g in Ge}
        def uav_t(a):
            c = self._ugv(a.carrier)
            return math.dist(self.ugv_xy(c), self._ev_xy) / self.w.s_a
        R = self.w.recovery_radius_m
        # deadhead limit: a UAV flies to the event only within the recovery radius of its carrier
        # (same bound as the recovery flight); farther UAVs can only ride a responding UGV
        Af = [a for a in Ae if R is None or uav_t(a) * self.w.s_a <= R]
        sent_g, sent_a = [], []
        mode = cfg.dispatch
        if mode == "single":
            if air:
                if Af:
                    sent_a = [min(Af, key=uav_t)]
            elif Ge:
                sent_g = [min(Ge, key=lambda g: ugv_t[g.uid])]
        elif mode == "random":
            r = random.Random(cfg.random_seed)
            sent_g = r.sample(sorted(Ge, key=lambda g: g.uid), min(q_g, len(Ge))) if not air else []
            riding = {g.docked_uav for g in sent_g if g.docked_uav}
            left = [a for a in sorted(Af, key=lambda a: a.uid) if a.uid not in riding]
            need = max(q_a - len(riding), 0)
            sent_a = [self._uav(u) for u in riding] + r.sample(left, min(need, len(left)))
        else:   # quota: Alg. respond
            Gl = sorted(Ge, key=lambda g: ugv_t.get(g.uid, math.inf))
            k = min(q_g, q_a)
            for g in list(Gl):
                if k <= 0:
                    break
                if g.docked_uav is not None:
                    sent_g.append(g); sent_a.append(self._uav(g.docked_uav)); Gl.remove(g)
                    k -= 1; q_g -= 1; q_a -= 1
            while q_g > 0 and Gl:
                sent_g.append(Gl.pop(0)); q_g -= 1
            riding = {a.uid for a in sent_a}
            Al = sorted([a for a in Af if a.uid not in riding and
                         not any(g.docked_uav == a.uid for g in sent_g)], key=uav_t)
            while q_a > 0 and Al:
                sent_a.append(Al.pop(0)); q_a -= 1
        q_g0, q_a0 = event_quota(sig, air, cfg)
        riding = {g.docked_uav for g in sent_g if g.docked_uav}
        self.ev_stats = {"node": v, "air": air, "severity": sig, "onset": self.t,
                         "n_ugv": len(sent_g), "n_uav": len({a.uid for a in sent_a} | riding),
                         "shortfall": max(q_g0 - len(sent_g), 0) + max(q_a0 - len({a.uid for a in sent_a}), 0),
                         "D_resp": 0.0, "first_arrival": None, "clear": None}
        waiting = set()
        for g in sent_g:
            if g.uid in self.claims:             # drop its starvation claim; node re-queues
                c = self.claims.pop(g.uid)
                self.claim_queue.insert(0, c["node"])
            g._ev = "to_event"
            self.ev_stats["D_resp"] += self._ev_route(g, v)
            waiting.add(g.uid)
        for a in sent_a:
            if a.uid in riding:
                continue                          # rides on its responding carrier
            c = self._ugv(a.carrier)
            a.pos_xy = self.ugv_xy(c)
            c.docked_uav = None
            a.carrier = None
            a.state = "to_event"
            waiting.add(a.uid)
        self._ev_resp_g = [g.uid for g in sent_g]
        self._ev_resp_a = [a.uid for a in sent_a if a.uid not in riding]
        h_e = self.events_cfg.handling if self.events_cfg.handling is not None else self.w.T
        # slots the variant must fill: quota for full / V5, one robot for V4 (single nearest)
        if mode == "single":
            t_g, t_a = (0, 1) if air else (1, 0)
        else:
            t_g, t_a = q_g0, q_a0
        n_a = len({a.uid for a in sent_a} | riding)
        self._ev_state = {"phase": "travel", "waiting": waiting, "h_e": h_e, "work": 0.0,
                          "present": 0, "W": (q_g0 + q_a0) * h_e, "riding": riding,
                          "open_g": max(t_g - len(sent_g), 0), "open_a": max(t_a - n_a, 0),
                          "couriers": {}, "rng": random.Random(cfg.random_seed + 1)}
        self.ev_stats["late"] = 0                 # slots filled after onset
        self.ev_stats["filled_at"] = 0.0 if not (self._ev_state["open_g"] or self._ev_state["open_a"]) else None

    # -- frozen-fleet counterfactual (no event): the quota robots, chosen nearest first as in
    #    Alg. respond, stop patrolling where they are for freeze_for seconds, then resume.
    def _fz_start(self):
        cfg = self.events_cfg
        v = cfg.node
        air = v in self.ghosts
        self._ev_xy = self.ghosts[v] if air else self.pos[v]
        q_g, q_a = event_quota(cfg.severity, air, cfg)
        self.ev_stats = {"node": v, "air": air, "severity": cfg.severity, "onset": self.t,
                         "n_ugv": 0, "n_uav": 0, "clear": cfg.freeze_for}
        self._ev_state = {"freeze": True, "phase": "frozen", "open_g": q_g, "open_a": q_a,
                          "g": [], "a": [], "until": self.t + cfg.freeze_for}
        self._fz_step()

    def _fz_step(self):
        ev, node = self._ev_state, self.events_cfg.node
        if ev["phase"] == "done":
            return
        if self.t >= ev["until"]:
            for g in ev["g"]:
                g._ev = None                      # resumes its walk where it stopped
            for a in ev["a"]:
                a._frozen = False
            ev["phase"] = "done"
            return
        while ev["open_g"] > 0:
            free = [g for g in self.ugvs if g.reserved_for is None and getattr(g, "_ev", None) is None]
            if not free:
                break
            g = min(free, key=lambda g: self._ugv_dist(g, node)[0])
            if g.uid in self.claims:
                c = self.claims.pop(g.uid)
                self.claim_queue.insert(0, c["node"])
            g._ev = "handling"
            ev["g"].append(g); ev["open_g"] -= 1; self.ev_stats["n_ugv"] += 1
            u = self._uav(g.docked_uav) if g.docked_uav else None
            if u is not None and ev["open_a"] > 0 and not getattr(u, "_frozen", False):
                u._frozen = True
                ev["a"].append(u); ev["open_a"] -= 1; self.ev_stats["n_uav"] += 1
        while ev["open_a"] > 0:
            docked = [a for a in self.uavs if a.state == UAVState.DOCKED and not getattr(a, "_frozen", False)]
            if not docked:
                break
            a = min(docked, key=lambda a: math.dist(self.ugv_xy(self._ugv(a.carrier)), self._ev_xy))
            a._frozen = True
            ev["a"].append(a); ev["open_a"] -= 1; self.ev_stats["n_uav"] += 1

    def _ev_pick(self, cands, key):
        """nearest first (full, V4); uniform random for V5"""
        if self.events_cfg.dispatch == "random":
            return self._ev_state["rng"].choice(sorted(cands, key=lambda x: x.uid))
        return min(cands, key=key)

    def _ev_slot_filled(self):
        ev = self._ev_state
        self.ev_stats["late"] += 1
        if ev["open_g"] == 0 and ev["open_a"] == 0:
            self.ev_stats["filled_at"] = self.t - self.ev_stats["onset"]

    def _ev_fly(self, a):
        """docked UAV leaves its carrier and flies straight to the event"""
        c = self._ugv(a.carrier)
        a.pos_xy = self.ugv_xy(c)
        c.docked_uav = None
        a.carrier = None
        a.state = "to_event"
        self._ev_state["waiting"].add(a.uid)
        self._ev_resp_a.append(a.uid)
        self.ev_stats["n_uav"] += 1

    def _ev_release_courier(self, g):
        """courier stops at its next node and resumes TPBP there"""
        self._ev_state["couriers"].pop(g.uid, None)
        g._ev = "to_prio"
        if g.walk and g.walk_idx < len(g.walk):
            g.walk = g.walk[:g.walk_idx + 1]

    def _ev_fill(self):
        """Response is mandatory. Slots not filled at onset stay open and are filled as robots
        become available, by the same rules as at onset:
          UGV slot: nearest free UGV (shortest path); it brings its docked UAV if a UAV slot is open.
          UAV slot: nearest docked UAV whose carrier is within r_rec of the event flies there.
          No docked UAV within r_rec: the UGV carrying a UAV that can bring it within r_rec
          soonest drives toward the event (courier); its UAV flies once in range."""
        ev, node, R = self._ev_state, self.events_cfg.node, self.w.recovery_radius_m
        if ev["open_g"] > 0:
            free = [g for g in self.ugvs if g.reserved_for is None and getattr(g, "_ev", None) is None]
            if free:
                g = self._ev_pick(free, lambda g: self._ugv_dist(g, node)[0])
                if g.uid in self.claims:
                    c = self.claims.pop(g.uid)
                    self.claim_queue.insert(0, c["node"])
                g._ev = "to_event"
                self.ev_stats["D_resp"] += self._ev_route(g, node)
                ev["waiting"].add(g.uid)
                self._ev_resp_g.append(g.uid)
                self.ev_stats["n_ugv"] += 1
                ev["open_g"] -= 1
                if g.docked_uav is not None and ev["open_a"] > 0:
                    ev["riding"].add(g.docked_uav)
                    self.ev_stats["n_uav"] += 1
                    ev["open_a"] -= 1
                    self._ev_slot_filled()
                self._ev_slot_filled()
        if ev["open_a"] <= 0:
            for uid in list(ev["couriers"]):
                self._ev_release_courier(self._ugv(uid))
            return
        def free_carrier(a):
            c = self._ugv(a.carrier) if a.carrier else None
            return c is not None and getattr(c, "_ev", None) in (None, "to_launch")
        def d_ev(a):
            return math.dist(self.ugv_xy(self._ugv(a.carrier)), self._ev_xy)
        docked = [a for a in self.uavs if a.state == UAVState.DOCKED and free_carrier(a)]
        near = [a for a in docked if R is None or d_ev(a) <= R]
        if near:
            a = self._ev_pick(near, d_ev)
            c = self._ugv(a.carrier)
            self._ev_fly(a)
            ev["open_a"] -= 1
            self._ev_slot_filled()
            if c.uid in ev["couriers"]:
                self._ev_release_courier(c)
            while len(ev["couriers"]) > ev["open_a"]:
                self._ev_release_courier(self._ugv(next(iter(ev["couriers"]))))
            return
        if len(ev["couriers"]) >= ev["open_a"]:
            return
        cands = [g for g in self.ugvs if g.docked_uav is not None and g.reserved_for is None
                 and getattr(g, "_ev", None) is None
                 and self._uav(g.docked_uav).state == UAVState.DOCKED]
        if not cands:
            return                                # every UAV airborne: first to land is picked up above
        Ln = [n for n in self.G.nodes if R is None or math.dist(self.pos[n], self._ev_xy) <= R]
        if not Ln:                                # no ground node within r_rec: closest one
            Ln = [min(self.G.nodes, key=lambda n: math.dist(self.pos[n], self._ev_xy))]
        def best_launch(g):
            return min((self._ugv_dist(g, n)[0] / self.w.v_g + math.dist(self.pos[n], self._ev_xy) / self.w.s_a, n)
                       for n in Ln)
        g = self._ev_pick(cands, lambda g: best_launch(g)[0])
        _, L = best_launch(g)
        if g.uid in self.claims:
            c = self.claims.pop(g.uid)
            self.claim_queue.insert(0, c["node"])
        g._ev = "to_launch"
        self.ev_stats["D_resp"] += self._ev_route(g, L)
        ev["couriers"][g.uid] = L

    def _ev_uav_release(self, a):
        """UAV done at the event: normal recovery from the event node"""
        node = self.events_cfg.node
        a.state = UAVState.SERVICING
        a.target_ghost = node
        a.service_start = self.t - a.expected_service_time
        if node in self.ghosts:
            self.claimed_ghosts[node] = a.uid
            self.occupied_since[node] = self.t

    def _ev_arrive(self, rid: str):
        ev = self._ev_state
        if self.ev_stats["first_arrival"] is None:
            self.ev_stats["first_arrival"] = self.t - self.ev_stats["onset"]
            node = self.events_cfg.node
            if self.occupied_since.get(node) is None:
                self.occupied_since[node] = self.t
            self.visit(node)
        ev["waiting"].discard(rid)
        g = self._ugv(rid)
        ev["present"] += 1 + (1 if g is not None and g.docked_uav in ev["riding"] else 0)
        ev["phase"] = "active"

    def _ev_ugv_arrived(self, g: UGV):
        if g._ev == "to_launch":                  # courier at its launch node: UAV flies even if beyond r_rec
            ev = self._ev_state
            ev["couriers"].pop(g.uid, None)
            if ev["phase"] != "done" and ev["open_a"] > 0 and g.docked_uav is not None:
                self._ev_fly(self._uav(g.docked_uav))
                ev["open_a"] -= 1
                self._ev_slot_filled()
            g._ev = None
            self.select_walk(g)
            return
        if g._ev == "to_event":
            g._ev = "handling"
            self._ev_arrive(g.uid)
        else:                                     # reached its nearest priority node: resume TPBP
            g._ev = None
            self.select_walk(g)

    def _ev_finish(self):
        node = self.events_cfg.node
        self._ev_state["phase"] = "done"
        self.ev_stats["clear"] = self.t - self.ev_stats["onset"]
        if self.occupied_since.get(node) is not None:
            self.occupied_since[node] = None
        self.visit(node)
        for uid in self._ev_resp_g:
            g = self._ugv(uid)
            p = min(self.priorities, key=lambda s: self._ugv_dist(g, s)[0])
            g._ev = "to_prio"
            g.edge_progress = 0.0
            self._ev_route(g, p)
            if len(g.walk) <= 1:
                g._ev = None
                self.select_walk(g)
        ev = self._ev_state
        self.ev_stats["unfilled"] = ev["open_g"] + ev["open_a"]
        for uid in list(ev["couriers"]):
            self._ev_release_courier(self._ugv(uid))
        for uid in self._ev_resp_a:               # UAVs go through normal recovery
            a = self._uav(uid)
            if a.state == "event_handling":
                self._ev_uav_release(a)

    # -- top-level loop -----------------------------------------------------

    def step(self):
        self._spawn_events()
        self._update_claims()
        for g in self.ugvs:
            self.check_launch(g)
            self.step_ugv(g)
        self._resolve_events()
        for uav in self.uavs:
            if uav.state == UAVState.SERVICING and getattr(uav, "_depart_at", None) is not None:
                if self.t >= uav._depart_at:
                    uav.state = UAVState.TO_RENDEZVOUS
                    self.occupied_since[uav.target_ghost] = None  # left the ghost, resumes idling
                    self.visit(uav.target_ghost)                   # ...from 0, not from arrival
                    uav._depart_at = None
            self.step_uav(uav)
        self._log()
        if self._track_nodes:
            self._idle_steps += 1
            for n in self._idle_nodes:
                i = self.idleness(n)
                self._idle_sum[n] += i
                if i > self._idle_max[n]:
                    self._idle_max[n] = i
        if self.record_traj and int(self.t / self.dt) % self.traj_stride == 0:
            self._log_traj()
        self.t += self.dt

    def run(self, horizon: float):
        steps = int(horizon / self.dt)
        for _ in range(steps):
            self.step()
        return self.metrics_log

    # -- metrics -------------------------------------------------------

    def _log(self):
        road_idle = [self.idleness(n) for n in self.G.nodes()]
        ghost_idle = [self.idleness(h) for h in self.ghosts]
        response_times = [ev.service_start - ev.spawn_t for ev in self.resolved_events
                           if ev.service_start is not None]
        self.metrics_log.append({
            "t": self.t,
            "mean_idleness": sum(road_idle) / len(road_idle) if road_idle else 0.0,
            "road_max_idleness": max(road_idle) if road_idle else 0.0,
            "priority_max_idleness": max(self.idleness(s) for s in self.priorities),
            "priority_mean_idleness": sum(self.idleness(s) for s in self.priorities) / len(self.priorities),
            "ghost_mean_idleness": sum(ghost_idle) / len(ghost_idle) if ghost_idle else 0.0,
            "ghost_max_idleness": max(ghost_idle) if ghost_idle else 0.0,
            "event_mean_response": sum(response_times) / len(response_times) if response_times else None,
            "event_max_response": max(response_times) if response_times else None,
            "events_resolved": len(self.resolved_events),
            "events_pending": len(self.active_events),
        })

    def _log_traj(self):
        self.traj_log.append({
            "t": self.t,
            "ugv": [self.ugv_xy(g) for g in self.ugvs],
            "uav": [(self.uav_xy(a), a.state) for a in self.uavs],
            "ghost_occupied": {h: self.occupied_since.get(h) is not None for h in self.ghosts},
        })
