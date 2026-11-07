"""Graph-complexity ladder for the RSS evaluation: G1 (full grid), G2 (some
edges removed), G3 (more edges removed). Ground nodes sit on a square
lattice; ghost (air-only) nodes sit inside grid cells, each a jittered
point near a cell centroid, with no graph edge at all -- a structural
property, not a label. Freshly generated here; no existing campus graph
or other pre-built dataset is reused, per project convention.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import networkx as nx

Coord = Tuple[float, float]


@dataclass
class Layout:
    name: str
    G: nx.Graph               # ground graph: nodes = V_g, edges = E
    ghosts: Dict[str, Coord]  # V_a: id -> (x, y), no edge, ever
    priorities: List[str]     # S subset of V_g
    pos: Dict[str, Coord]     # V_g node positions


def build_grid(rows: int, cols: int, spacing: float = 100.0) -> Tuple[nx.Graph, Dict[str, Coord]]:
    """Full square-lattice ground graph, 4-connected (no diagonals)."""
    G = nx.Graph()
    pos: Dict[str, Coord] = {}
    for r in range(rows):
        for c in range(cols):
            nid = f"g{r}_{c}"
            pos[nid] = (c * spacing, r * spacing)
            G.add_node(nid)
    for r in range(rows):
        for c in range(cols):
            nid = f"g{r}_{c}"
            if c + 1 < cols:
                G.add_edge(nid, f"g{r}_{c+1}")
            if r + 1 < rows:
                G.add_edge(nid, f"g{r+1}_{c}")
    return G, pos


def place_ghosts(rows: int, cols: int, pos: Dict[str, Coord], n_ghosts: int,
                  priority_pos: List[Coord], spacing: float = 100.0,
                  jitter: float = 18.0, seed: int = 0) -> Dict[str, Coord]:
    """Pick n_ghosts grid cells by farthest-point sampling over cell
    centroids, biased away from priority nodes, so ghosts are deliberately
    spread out and include at least one far from every priority -- the
    case the ghost-credit starvation rule (genswarm-extension.tex Sec 2.3)
    exists to handle. Not cosmetic: a baseline with no ghost credit or a
    fixed launch node should visibly struggle on exactly this placement."""
    rng = random.Random(seed)
    cells = [(r, c) for r in range(rows - 1) for c in range(cols - 1)]
    centroids = {(r, c): ((c + 0.5) * spacing, (r + 0.5) * spacing) for r, c in cells}

    def dist2(a: Coord, b: Coord) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    # start from the cell farthest from the nearest priority node, so the
    # hardest-to-reach ghost is guaranteed to be in the set
    start = max(cells, key=lambda rc: min(dist2(centroids[rc], p) for p in priority_pos))
    chosen = [start]
    while len(chosen) < n_ghosts:
        best, best_d = None, -1.0
        for rc in cells:
            if rc in chosen:
                continue
            d = min(dist2(centroids[rc], centroids[c]) for c in chosen)
            if d > best_d:
                best, best_d = rc, d
        chosen.append(best)

    ghosts: Dict[str, Coord] = {}
    for i, (r, c) in enumerate(chosen):
        cx, cy = centroids[(r, c)]
        cx += rng.uniform(-jitter, jitter)
        cy += rng.uniform(-jitter, jitter)
        ghosts[f"h{i}"] = (cx, cy)
    return ghosts


def pick_priorities(G: nx.Graph, k: int, pos: Dict[str, Coord],
                     seed: int = 0) -> List[str]:
    """Farthest-point sampling over node positions, so priorities spread
    across corners and edges rather than clustering near the center."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    first = rng.choice(nodes)
    chosen = [first]
    while len(chosen) < k:
        best, best_d = None, -1.0
        for n in nodes:
            if n in chosen:
                continue
            d = min((pos[n][0] - pos[c][0]) ** 2 + (pos[n][1] - pos[c][1]) ** 2
                     for c in chosen)
            if d > best_d:
                best, best_d = n, d
        chosen.append(best)
    return sorted(chosen)


def thin_edges(G: nx.Graph, pos: Dict[str, Coord], drop_fraction: float,
                seed: int = 0, protect_nodes: Optional[List[str]] = None,
                priorities: Optional[List[str]] = None) -> nx.Graph:
    """Remove a fraction of edges while keeping the graph connected.
    Never removes an edge if doing so would disconnect the graph.

    If protect_nodes/priorities are given, also never removes an edge
    if doing so would take a protected node off every shortest path
    between every priority pair -- i.e. make it genuinely unreachable
    by the TPBP valid-walk scheme (priority-to-priority routes only),
    regardless of hop budget. Without this guard, thinning can strand a
    ghost's nearest node on a dead-end spur that no priority-to-priority
    path ever passes through, making that ghost unservicable by any
    variant -- a map-construction flaw, not a method result. Applied
    identically before any variant runs, so it is a map fix, not a bias
    toward any one condition."""
    rng = random.Random(seed)
    H = G.copy()
    candidates = list(H.edges())
    rng.shuffle(candidates)
    target_removals = int(len(candidates) * drop_fraction)

    def reachable(graph):
        if not protect_nodes or not priorities:
            return True
        for n in protect_nodes:
            ok = False
            for i, pa in enumerate(priorities):
                for pb in priorities[i + 1:]:
                    if not (nx.has_path(graph, pa, n) and nx.has_path(graph, n, pb)):
                        continue
                    d_direct = nx.shortest_path_length(graph, pa, pb)
                    d_via = (nx.shortest_path_length(graph, pa, n)
                             + nx.shortest_path_length(graph, n, pb))
                    if d_via == d_direct:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                return False
        return True

    removed = 0
    for u, v in candidates:
        if removed >= target_removals:
            break
        H.remove_edge(u, v)
        if nx.is_connected(H) and reachable(H):
            removed += 1
        else:
            H.add_edge(u, v)
    return H


def build_irregular_hub(n_arms: int = 5, arm_len: int = 4,
                         seed: int = 0) -> Tuple[nx.Graph, Dict[str, Coord]]:
    """A non-grid, hub-and-spoke ground graph with genuinely varied edge
    lengths (unlike the lattice, where every edge is identical): a central
    hub with n_arms radiating arms of arm_len nodes each, plus cross-links
    between adjacent arms at matching rings for redundancy and loops, in
    the spirit of Deepak's fan-shaped Layout B/C but larger. Varied
    geometric spacing matters here because UGV/UAV recovery timing
    (Sec 3.3 of genswarm-extension.tex) depends on real Euclidean
    distance, not graph-hop count, so a uniform grid can't exercise it as
    meaningfully as this can."""
    rng = random.Random(seed)
    G = nx.Graph()
    pos: Dict[str, Coord] = {"hub": (0.0, 0.0)}
    G.add_node("hub")

    arms: List[List[str]] = []
    for a in range(n_arms):
        angle = 2 * math.pi * a / n_arms + rng.uniform(-0.12, 0.12)
        chain = []
        prev = "hub"
        for i in range(1, arm_len + 1):
            radius = i * (85 + rng.uniform(-10, 15))
            nid = f"arm{a}_{i}"
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            pos[nid] = (x, y)
            G.add_node(nid)
            G.add_edge(prev, nid)
            chain.append(nid)
            prev = nid
        arms.append(chain)

    # cross-links between adjacent arms at matching rings, for loops and
    # shortcuts (not a pure tree) -- skip the outermost ring sometimes so
    # the graph isn't perfectly uniform either
    for a in range(n_arms):
        other = (a + 1) % n_arms
        for ring in range(arm_len - 1):
            if rng.random() < 0.7:
                G.add_edge(arms[a][ring], arms[other][ring])

    return G, pos


def place_ghosts_irregular(pos: Dict[str, Coord], priority_pos: List[Coord],
                            seed: int = 0) -> Dict[str, Coord]:
    """Deliberately mixed ghost placement for the irregular map: one
    isolated far ghost (starvation test), one close clustered pair near a
    priority node (crowding / claim-set test, Q+ and Q- both), and two
    moderate-distance ghosts near different arms (general recovery-credit
    timing diversity, since arm distances vary)."""
    rng = random.Random(seed)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span = max(max(xs) - min(xs), max(ys) - min(ys))

    def dist(a: Coord, b: Coord) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # isolated: beyond the farthest node from every priority
    far_node = max(pos.items(), key=lambda kv: min(dist(kv[1], p) for p in priority_pos))
    fx, fy = far_node[1]
    norm = math.hypot(fx, fy) or 1.0
    isolated = (fx * 1.55, fy * 1.55)

    # clustered pair: near whichever priority node is closest to the hub,
    # offset in two nearby but distinct directions
    near_priority = min(priority_pos, key=lambda p: math.hypot(*p))
    base_angle = rng.uniform(0, 2 * math.pi)
    clustered_a = (near_priority[0] + 40 * math.cos(base_angle),
                   near_priority[1] + 40 * math.sin(base_angle))
    clustered_b = (near_priority[0] + 40 * math.cos(base_angle + 0.9),
                   near_priority[1] + 40 * math.sin(base_angle + 0.9))

    # two moderate ghosts on different arms, offset from the arm line
    nodes = list(pos.items())
    rng.shuffle(nodes)
    moderate = []
    picks = [n for n in nodes if n[0] != "hub"][:6]
    for nid, (x, y) in picks[:2]:
        angle = rng.uniform(0, 2 * math.pi)
        moderate.append((x + 35 * math.cos(angle), y + 35 * math.sin(angle)))

    coords = [isolated, clustered_a, clustered_b] + moderate
    return {f"h{i}": c for i, c in enumerate(coords)}


def build_campus(n_loop: int = 18, radius: float = 420.0, n_subdiv: int = 2,
                  n_chords: int = 6, n_spurs: int = 6, spur_len: int = 3,
                  seed: int = 0) -> Tuple[nx.Graph, Dict[str, Coord], List[Coord]]:
    """Large-scale synthetic campus-style road network: an irregular
    perimeter loop (lumpy, not a rectangle), interior cross-street
    chords, several dead-end spurs, and one small interior sub-block --
    the same topological species as a real campus road network (organic
    loop + chords + dead ends, no grid regularity), built fresh from
    scratch with random irregular geometry. No existing campus graph
    file (iitb.graphml or similar) is read or reused anywhere in this
    function, per project convention; only the *style* is modeled after
    one, not the data. Returns the road graph, its positions, and ghost
    placement candidates (interior points with no road access)."""
    rng = random.Random(seed)
    G = nx.Graph()
    pos: Dict[str, Coord] = {}

    # irregular closed perimeter: near-evenly-spaced angles with jitter
    # (not pure random, which can bunch several vertices together) and a
    # randomized radius per vertex for the lumpy, non-rectangular outline
    step = 2 * math.pi / n_loop
    angles = sorted((i * step + rng.uniform(-step * 0.3, step * 0.3)) % (2 * math.pi)
                     for i in range(n_loop))
    verts = []
    for i, ang in enumerate(angles):
        r = radius * rng.uniform(0.65, 1.0)
        verts.append((r * math.cos(ang), r * math.sin(ang)))

    loop_ids = [f"p{i}" for i in range(n_loop)]
    for nid, xy in zip(loop_ids, verts):
        pos[nid] = xy
        G.add_node(nid)

    def add_subdivided_edge(a: str, b: str, k: int, prefix: str):
        ax, ay = pos[a]
        bx, by = pos[b]
        prev = a
        for j in range(1, k + 1):
            t = j / (k + 1)
            nid = f"{prefix}{j}"
            pos[nid] = (ax + t * (bx - ax), ay + t * (by - ay))
            G.add_node(nid)
            G.add_edge(prev, nid)
            prev = nid
        G.add_edge(prev, b)

    # perimeter, subdivided for finer routing granularity
    for i in range(n_loop):
        a, b = loop_ids[i], loop_ids[(i + 1) % n_loop]
        add_subdivided_edge(a, b, n_subdiv, f"e{i}_")

    # interior chords across the loop, mimicking the real network's
    # cross-streets; each also subdivided
    chord_count = 0
    attempts = 0
    while chord_count < n_chords and attempts < 200:
        attempts += 1
        i, j = rng.sample(range(n_loop), 2)
        if abs(i - j) % n_loop < 3 or abs(i - j) % n_loop > n_loop - 3:
            continue  # skip near-adjacent pairs, those are basically perimeter
        a, b = loop_ids[i], loop_ids[j]
        add_subdivided_edge(a, b, n_subdiv, f"c{chord_count}_")
        chord_count += 1

    # dead-end spurs off random loop nodes, pointing outward (away from
    # the shape's center) so they never fold back into the interior and
    # overlap other nodes, matching the real network's clean stubs
    spur_bases = rng.sample(loop_ids, min(n_spurs, n_loop))
    for s, base in enumerate(spur_bases):
        bx, by = pos[base]
        out_ang = math.atan2(by, bx) + rng.uniform(-0.25, 0.25)
        prev = base
        for k in range(1, spur_len + 1):
            nid = f"spur{s}_{k}"
            dist = 90 * k
            pos[nid] = (bx + dist * math.cos(out_ang),
                        by + dist * math.sin(out_ang))
            G.add_node(nid)
            G.add_edge(prev, nid)
            prev = nid

    # one small interior sub-block (a tiny rectangle), like the real
    # network's interior building near its edge cluster
    cx, cy = rng.uniform(-80, 80), rng.uniform(-80, 80)
    rect = [f"blk{k}" for k in range(4)]
    offs = [(-40, -30), (40, -30), (40, 30), (-40, 30)]
    for nid, (dx, dy) in zip(rect, offs):
        pos[nid] = (cx + dx, cy + dy)
        G.add_node(nid)
    for k in range(4):
        G.add_edge(rect[k], rect[(k + 1) % 4])
    # connect the sub-block to its nearest existing node
    nearest = min((n for n in G.nodes() if n not in rect),
                  key=lambda n: (pos[n][0] - cx) ** 2 + (pos[n][1] - cy) ** 2)
    G.add_edge(nearest, rect[0])

    assert nx.is_connected(G), "campus generation produced a disconnected road graph"

    # ghost candidates: interior points away from every road node, found
    # by sampling the interior and keeping points far from the nearest road
    all_xy = list(pos.values())
    minx, maxx = min(p[0] for p in all_xy), max(p[0] for p in all_xy)
    miny, maxy = min(p[1] for p in all_xy), max(p[1] for p in all_xy)
    candidates: List[Coord] = []
    tries = 0
    while len(candidates) < 40 and tries < 4000:
        tries += 1
        x = rng.uniform(minx * 0.7, maxx * 0.7)
        y = rng.uniform(miny * 0.7, maxy * 0.7)
        nearest_d = min((x - p[0]) ** 2 + (y - p[1]) ** 2 for p in all_xy)
        if nearest_d > 55 ** 2:
            candidates.append((x, y))

    return G, pos, candidates


def place_ghosts_campus(candidates: List[Coord], priority_pos: List[Coord],
                         n_ghosts: int = 9, seed: int = 0) -> Dict[str, Coord]:
    """Farthest-point sampling over interior candidate points (already
    guaranteed far from any road node), starting from the point farthest
    from every priority node, so the set is spread and includes a
    genuinely hard-to-reach ghost for the starvation test, same spirit as
    place_ghosts for the grid maps."""
    def dist2(a: Coord, b: Coord) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    start = max(candidates, key=lambda c: min(dist2(c, p) for p in priority_pos))
    chosen = [start]
    while len(chosen) < n_ghosts and len(chosen) < len(candidates):
        best, best_d = None, -1.0
        for c in candidates:
            if c in chosen:
                continue
            d = min(dist2(c, ch) for ch in chosen)
            if d > best_d:
                best, best_d = c, d
        chosen.append(best)
    return {f"h{i}": c for i, c in enumerate(chosen)}


def place_ghosts_campus_with_crowding(candidates: List[Coord], priority_pos: List[Coord],
                                       n_ghosts: int = 9, seed: int = 0) -> Dict[str, Coord]:
    """Same spread placement as place_ghosts_campus, plus one deliberate
    close pair near a priority node, so the crowding / claim-set
    mechanism (Q- and Q+, genswarm-extension.tex Sec 2.3-2.4) actually
    has something to prevent. Used only for M4; M1/M2/M3 keep their
    existing placement unchanged."""
    base = place_ghosts_campus(candidates, priority_pos, n_ghosts=max(n_ghosts - 1, 1), seed=seed)
    rng = random.Random(seed)
    near_priority = min(priority_pos, key=lambda p: math.hypot(*p))
    pick_key = rng.choice(list(base.keys()))
    px, py = base[pick_key]
    angle = rng.uniform(0, 2 * math.pi)
    extra = (px + 45 * math.cos(angle), py + 45 * math.sin(angle))
    base[f"h{len(base)}"] = extra
    return base


def load_osm_city(place: tuple, dist: float = 500.0, n_ghosts: int = 9,
                   n_priorities: int = 13, seed: int = 0) -> Layout:
    """Load a real road network from OpenStreetMap via OSMnx, same method
    as the Line Coverage dataset (Agarwal & Akella, IEEE T-RO 2024) that
    both Nguyen & Akella RSS papers (DUCPP, Optimal Partitioning)
    evaluate on. `place` is a (lat, lon) center point; `dist` is the
    extraction radius in meters. Real, independently-sourced, organic
    non-grid topology -- not hand-designed, not any campus graph."""
    import osmnx as ox

    ox.settings.log_console = False
    ox.settings.use_cache = True
    raw = ox.graph_from_point(place, dist=dist, network_type="drive", simplify=True)
    raw = ox.convert.to_undirected(raw)
    raw_proj = ox.projection.project_graph(raw)

    G = nx.Graph()
    pos: Dict[str, Coord] = {}
    for n, data in raw_proj.nodes(data=True):
        pos[str(n)] = (float(data["x"]), float(data["y"]))
        G.add_node(str(n))
    for u, v in raw_proj.edges():
        if u != v:
            G.add_edge(str(u), str(v))
    # keep only the largest connected component (OSM extracts can have
    # small disconnected slivers at the boundary of the query radius)
    largest = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest).copy()
    pos = {n: pos[n] for n in G.nodes()}
    assert nx.is_connected(G)

    priorities = pick_priorities(G, n_priorities, pos, seed=seed)
    priority_pos = [pos[p] for p in priorities]

    all_xy = list(pos.values())
    minx, maxx = min(p[0] for p in all_xy), max(p[0] for p in all_xy)
    miny, maxy = min(p[1] for p in all_xy), max(p[1] for p in all_xy)
    rng = random.Random(seed)
    candidates: List[Coord] = []
    tries = 0
    while len(candidates) < 60 and tries < 8000:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        nearest_d = min((x - p[0]) ** 2 + (y - p[1]) ** 2 for p in all_xy)
        if nearest_d > (dist * 0.05) ** 2:
            candidates.append((x, y))
    ghosts = place_ghosts_campus(candidates, priority_pos,
                                                n_ghosts=n_ghosts, seed=seed)

    return Layout("M3 (Boston, real OSM road network)", G, ghosts, priorities, pos)


def load_iitb_graphml(path: str, n_ghosts: int = 6,
                       n_priorities: int = 5, seed: int = 0) -> Layout:
    """Load Deepak's actual iitb.graphml exactly as-is: real node
    coordinates, real road connectivity, no modification to the graph
    itself. Only priorities and ghost placement (new concepts, not in
    the original TPBP file) are added on top."""
    raw = nx.read_graphml(path)
    G = nx.Graph()
    pos: Dict[str, Coord] = {}
    for n, data in raw.nodes(data=True):
        pos[n] = (float(data["x"]), float(data["y"]))
        G.add_node(n)
    for u, v in raw.edges():
        G.add_edge(u, v)
    assert nx.is_connected(G), "iitb.graphml is not connected as loaded"

    priorities = pick_priorities(G, n_priorities, pos, seed=seed)
    priority_pos = [pos[p] for p in priorities]

    # ghost candidates: interior points far from every road node (air-only
    # by construction), same farthest-point spread as the other maps
    all_xy = list(pos.values())
    minx, maxx = min(p[0] for p in all_xy), max(p[0] for p in all_xy)
    miny, maxy = min(p[1] for p in all_xy), max(p[1] for p in all_xy)
    rng = random.Random(seed)
    candidates: List[Coord] = []
    tries = 0
    while len(candidates) < 60 and tries < 6000:
        tries += 1
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        nearest_d = min((x - p[0]) ** 2 + (y - p[1]) ** 2 for p in all_xy)
        if nearest_d > 40 ** 2:
            candidates.append((x, y))
    ghosts = place_ghosts_campus(candidates, priority_pos, n_ghosts=n_ghosts, seed=seed)

    return Layout("M3 (iitb.graphml, Deepak's design)", G, ghosts, priorities, pos)


IITB_GRAPHML = "../upstream/mrpp_sumo/graph_ml/iitb.graphml"


def make_ladder(rows: int = 10, cols: int = 10, n_ghosts: int = 5,
                 n_priorities: int = 6, seed: int = 0) -> List[Layout]:
    G1, pos1 = build_grid(rows, cols)
    priorities1 = pick_priorities(G1, n_priorities, pos1, seed=seed)
    priority_pos1 = [pos1[p] for p in priorities1]
    ghosts1 = place_ghosts(rows, cols, pos1, n_ghosts, priority_pos1, seed=seed)

    ghost_nearest_nodes = [min(G1.nodes(), key=lambda n: math.dist(pos1[n], hp))
                           for hp in ghosts1.values()]
    G2 = thin_edges(G1, pos1, drop_fraction=0.43, seed=seed + 1,
                     protect_nodes=ghost_nearest_nodes, priorities=priorities1)

    m3 = load_osm_city((42.3601, -71.0589), seed=seed)  # Boston, USA

    return [
        Layout("M1 (full grid)", G1, ghosts1, priorities1, pos1),
        Layout("M2 (grid, near-tree)", G2, ghosts1, priorities1, pos1),
        m3,
    ]


def plot_layout(layout: Layout, ax, scale: float = 1.0, edge_color: str = "#78828C"):
    nx.draw_networkx_edges(layout.G, layout.pos, ax=ax, edge_color=edge_color,
                            width=1.4 * scale)
    ground_only = [n for n in layout.G.nodes() if n not in layout.priorities]
    nx.draw_networkx_nodes(layout.G, layout.pos, nodelist=ground_only, ax=ax,
                            node_color="white", edgecolors="#546E7A",
                            node_size=120 * scale, linewidths=1.0 * scale)
    nx.draw_networkx_nodes(layout.G, layout.pos, nodelist=layout.priorities, ax=ax,
                            node_color="#FFC107", edgecolors="#8D6E00",
                            node_size=170 * scale, linewidths=1.2 * scale)
    gx = [p[0] for p in layout.ghosts.values()]
    gy = [p[1] for p in layout.ghosts.values()]
    ax.scatter(gx, gy, marker="D", s=130 * scale, facecolor="#1565C0", edgecolor="white",
               linewidths=1.2 * scale, zorder=5)
    ax.set_title(f"{layout.name}\n|V_g|={layout.G.number_of_nodes()}, "
                 f"|E|={layout.G.number_of_edges()}, |V_a|={len(layout.ghosts)}, "
                 f"|S|={len(layout.priorities)}", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    ladder = make_ladder()
    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="#546E7A", markersize=9, label="ground node $V_g$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFC107",
                   markeredgecolor="#8D6E00", markersize=10, label="priority node $S$"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#1565C0",
                   markeredgecolor="white", markersize=10, label="ghost node $V_a$"),
    ]

    # M1 + M2, locked, side by side
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for layout, ax in zip(ladder[:2], axes):
        plot_layout(layout, ax)
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = "m1_m2_locked.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")

    # M3, standalone, very large canvas, small markers -- real OSM data
    # has dense intersection clusters that need room to breathe
    m3 = ladder[2]
    fig3, ax3 = plt.subplots(figsize=(28, 24))
    plot_layout(m3, ax3, scale=0.55, edge_color="#5E6973")
    fig3.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
                fontsize=16)
    fig3.tight_layout(rect=(0, 0.03, 1, 1))
    out3 = "m3_boston_osm.pdf"
    fig3.savefig(out3, bbox_inches="tight")
    print(f"saved {out3}")

    for layout in ladder:
        print(layout.name, "nodes:", layout.G.number_of_nodes(),
              "edges:", layout.G.number_of_edges(),
              "connected:", nx.is_connected(layout.G))


if __name__ == "__main__":
    main()
