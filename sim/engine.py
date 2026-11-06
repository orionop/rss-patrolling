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
    c9: float = 5.0           # emergency diversion weight, severity-scaled (M3 events layer only)
    r_max: float = 70.0       # launch/recovery-credit proximity radius (meters)
    i_min: float = 20.0       # minimum ghost idleness before credit fires
    v_g: float = 4.0          # UGV speed, m/s
    s_a: float = 12.0         # UAV speed, m/s (s_a >> v_g, per formulation)
    eps_l: float = 2.0        # rendezvous co-location tolerance (meters)
    landing_candidates: int = 3  # C in Sec 3.3 step 3


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

def enumerate_valid_walks(G: nx.Graph, priorities: List[str], t_hops: int) -> List[List[str]]:
    walks: List[List[str]] = []
    for s in priorities:
        for f in priorities:
            if s == f:
                continue
            for path in nx.all_simple_paths(G, s, f, cutoff=t_hops):
                walks.append(path)
    return walks


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class Sim:
    def __init__(self, layout: Layout, n_ugv: int, n_uav: int,
                 variant: VariantConfig, weights: Weights = None,
                 t_hops: int = 8, seed: int = 0, dt: float = 1.0,
                 events: EventConfig = None, record_traj: bool = False,
                 traj_stride: int = 5):
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

        self.valid_walks = enumerate_valid_walks(self.G, self.priorities, t_hops)

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

        self.metrics_log: List[dict] = []
        self.record_traj = record_traj
        self.traj_stride = traj_stride
        self.traj_log: List[dict] = []

    # -- setup ---------------------------------------------------------

    def _init_fleet(self, n_ugv: int, n_uav: int):
        nodes = list(self.G.nodes())
        start_nodes = self.rng.sample(nodes, min(n_ugv, len(nodes)))
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

    def visit(self, node: str):
        self.last_visit[node] = self.t

    # -- reward: base TPBP + ghost credit + recovery credit --------------

    def walk_reward(self, g: UGV, walk: List[str]) -> Tuple[float, Optional[str], Optional[str]]:
        """Returns (reward, credited_launch_ghost_or_None, credited_recovery_ghost_or_None)."""
        w = self.w
        term1 = sum(self.idleness(v) for v in walk) * w.c1
        term2 = sum(max(self.idleness(s) - 0.0, 0.0) for s in walk if s in self.priorities) * w.c2
        unclaimed_priorities = [s for s in self.priorities if s not in self._destinations_in_use()]
        term3 = sum(nx.shortest_path_length(self.G, walk[-1], s) for s in unclaimed_priorities) * w.c3
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
                hp = self.ghosts[h]
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

    def _destinations_in_use(self) -> set:
        return {g.walk[-1] for g in self.ugvs if g.walk}

    def select_walk(self, g: UGV):
        candidates = [w for w in self.valid_walks if w[0] == g.pos_node]
        if not candidates:
            candidates = [w for w in self.valid_walks]  # fallback: any walk, teleport-free not guaranteed
        best, best_r, best_launch, best_recovery = None, -1e18, None, None
        for w in candidates:
            r, launch_h, recovery_h = self.walk_reward(g, w)
            if r > best_r:
                best, best_r, best_launch, best_recovery = w, r, launch_h, recovery_h
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
        if g.docked_uav is None:
            return
        uav = self._uav(g.docked_uav)
        if uav is None or uav.state != UAVState.DOCKED:
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

        if not in_range:
            return

        # tie-break: nearest, then highest idleness (genswarm-extension.tex Sec 3.2)
        h_star = min(in_range, key=lambda h: (math.dist(gxy, self.ghosts[h]), -self.idleness(h)))
        self.claimed_ghosts[h_star] = uav.uid
        uav.state = UAVState.TO_GHOST
        uav.target_ghost = h_star
        uav.pos_xy = gxy
        g.docked_uav = None

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
            if self.t - uav.service_start >= uav.expected_service_time:
                self._assign_recovery(uav)

        elif uav.state == UAVState.TO_RENDEZVOUS:
            uav.pos_xy = self._move_toward(uav.pos_xy, uav.recovery_point, w.s_a)
            rg = self._ugv(uav.recovery_ugv)
            if rg is not None and math.dist(uav.pos_xy, self.ugv_xy(rg)) <= w.eps_l \
               and rg.pos_node == self._node_at_point(uav.recovery_point):
                self._dock(uav, rg)

        elif uav.state == UAVState.WAITING_AT_RENDEZVOUS:
            rg = self._ugv(uav.recovery_ugv)
            if rg is not None and rg.pos_node == self._node_at_point(uav.recovery_point):
                self._dock(uav, rg)

    def _assign_recovery(self, uav: UAV):
        w = self.w
        candidates = [g for g in self.ugvs if g.is_free]
        if not candidates:
            return  # wait another step; stays SERVICING, continuous-occupancy holds idleness at 0

        best = None
        best_TL = 1e18
        best_point = None
        best_g = None

        for g in candidates:
            if self.variant.landing_candidate_recovery:
                points = self._landing_points(g, w.landing_candidates)
            else:
                # nearest-UGV-recovery ablation: single point, the UGV's
                # current position, no coordination with its planned path
                points = [(g.pos_node, self.pos[g.pos_node])]
            for node_id, p in points:
                t_u = self._ugv_time_to_point(g, p)
                t_a = math.dist(self.ghosts[uav.target_ghost], p) / w.s_a
                TL = max(t_u, t_a)
                if TL < best_TL:
                    best_TL, best_point, best_g = TL, p, g
        if best_g is None:
            return

        best_g.reserved_for = uav.uid
        best_g.reserved_point = best_point
        t_a = math.dist(self.ghosts[uav.target_ghost], best_point) / w.s_a
        uav.recovery_ugv = best_g.uid
        uav.recovery_point = best_point
        uav.depart_time = self.t + max(best_TL - t_a, 0.0)
        if uav.target_ghost in self.recovery_claims:
            del self.recovery_claims[uav.target_ghost]
        uav.state = UAVState.SERVICING  # keep servicing until depart_time
        uav._depart_at = uav.depart_time  # scheduled check in step_uav caller

    def _landing_points(self, g: UGV, c: int) -> List[Tuple[str, Coord]]:
        """Up to C distinct points along g's remaining planned path,
        deduped -- genswarm-extension.tex Sec 3.3 step 3 (dedup version)."""
        remaining = g.walk[g.walk_idx - 1:] if g.walk else [g.pos_node]
        if len(remaining) < 2:
            return [(g.pos_node, self.pos[g.pos_node])]
        cum = [0.0]
        for i in range(len(remaining) - 1):
            cum.append(cum[-1] + self.edge_len[(remaining[i], remaining[i + 1])])
        total = cum[-1]
        if total == 0:
            return [(remaining[0], self.pos[remaining[0]])]
        targets = [total / 3, 2 * total / 3, total][:c]
        points = []
        seen = set()
        for td in targets:
            idx = 0
            while idx < len(cum) - 1 and cum[idx + 1] < td:
                idx += 1
            node = remaining[idx + 1] if idx + 1 < len(remaining) else remaining[-1]
            if node not in seen:
                seen.add(node)
                points.append((node, self.pos[node]))
        return points

    def _ugv_time_to_point(self, g: UGV, p: Coord) -> float:
        d_to_pos = math.dist(self.ugv_xy(g), p)
        return d_to_pos / self.w.v_g

    def _dock(self, uav: UAV, g: UGV):
        self.occupied_since[uav.target_ghost] = None
        if uav.target_ghost in self.claimed_ghosts:
            del self.claimed_ghosts[uav.target_ghost]
        uav.state = UAVState.DOCKED
        uav.carrier = g.uid
        uav.target_ghost = None
        uav.recovery_ugv = None
        uav.recovery_point = None
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
        if not self.events_cfg.enabled:
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

    # -- top-level loop -----------------------------------------------------

    def step(self):
        self._spawn_events()
        for g in self.ugvs:
            self.check_launch(g)
            self.step_ugv(g)
        self._resolve_events()
        for uav in self.uavs:
            if uav.state == UAVState.SERVICING and getattr(uav, "_depart_at", None) is not None:
                if self.t >= uav._depart_at:
                    uav.state = UAVState.TO_RENDEZVOUS
                    self.occupied_since[uav.target_ghost] = None  # left the ghost, resumes idling
                    uav._depart_at = None
            self.step_uav(uav)
        self._log()
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
