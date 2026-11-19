"""Live visual simulation of the patrolling sim on the M1 / M2 maps.

Run from the sim/ folder:

    python3 visualize.py                              # M1, full method, fleet 4-2, seed 0
    python3 visualize.py --map m2 --variant fixed_launch_nodes --fleet 6-3 --seed 1
    python3 visualize.py --speed 40                   # faster playback
    python3 visualize.py --save m1.gif --no-show      # render to a file, no window
    python3 visualize.py --node-csv node_idleness.csv # per-node avg/max idleness at the end

What you see
  * every node is coloured by its CURRENT idleness (green = just visited,
    red = stale); ghost nodes are the diamonds
  * priority nodes have a thick gold outline
  * black squares = UGVs (dashed line = the walk it is following)
    triangles = UAVs, coloured by state (legend on the right)
  * HOVER a node: its id and idleness. CLICK a node: pin it, and its idleness
    history is plotted at the bottom right.
Keys: SPACE pause/resume | r restart the run | UP / DOWN faster / slower | s write node CSV now
"""
import argparse
import csv
import math
import sys

import matplotlib

# headless render when saving without a window
if "--no-show" in sys.argv or "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from engine import Sim, VARIANTS, UAVState, Weights
from runner import _layout, _walks, make_weights

MAPS = {"m1": (0, 400), "m2": (1, 400)}   # (index in make_ladder, default T_HOPS)

UAV_COLORS = {
    UAVState.DOCKED: "#9E9E9E",
    UAVState.TO_GHOST: "#1E88E5",
    UAVState.SERVICING: "#2E7D32",
    UAVState.TO_RENDEZVOUS: "#FB8C00",
    UAVState.WAITING_AT_RENDEZVOUS: "#D81B60",
}


def hhmmss(t):
    t = int(t)
    return f"{t // 3600}:{(t % 3600) // 60:02d}:{t % 60:02d}"


class Viewer:
    def __init__(self, sim, layout, variant_key, fleet, vmax, steps_per_frame,
                 horizon, node_csv, make_sim=None):
        self.sim = sim
        self.make_sim = make_sim   # callable returning a fresh Sim (for the r key)
        self.layout = layout
        self.variant_key = variant_key
        self.fleet = fleet
        self.vmax = vmax
        self.steps_per_frame = max(1, int(steps_per_frame))
        self.horizon = horizon
        self.node_csv = node_csv
        self.paused = False
        self.done = False
        self.pinned = None

        # fixed node order for colouring / lookup
        self.ground = list(sim.G.nodes())
        self.ghost_ids = list(sim.ghosts)
        self.all_ids = self.ground + self.ghost_ids
        self.xy = {n: sim.pos[n] for n in self.ground}
        self.xy.update({h: sim.ghosts[h] for h in self.ghost_ids})

        self.times = []
        self.hist = {n: [] for n in self.all_ids}   # idleness history per node
        self.road_sum = 0.0
        self.ghost_sum = 0.0
        self.frames_logged = 0

        self._build_figure()

    # ----- figure ------------------------------------------------------

    def _build_figure(self):
        sim = self.sim
        self.fig = plt.figure(figsize=(15, 8))
        gs = self.fig.add_gridspec(3, 3, width_ratios=[3.0, 0.09, 1.35],
                                   height_ratios=[1.25, 0.9, 1.0], wspace=0.14, hspace=0.3)
        self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax_cbar = self.fig.add_subplot(gs[:, 1])
        self.ax_txt = self.fig.add_subplot(gs[0, 2])
        self.ax_leg = self.fig.add_subplot(gs[1, 2])
        self.ax_hist = self.fig.add_subplot(gs[2, 2])
        self.ax_txt.axis("off")
        self.ax_leg.axis("off")

        ax = self.ax
        for u, v in sim.G.edges():
            (x1, y1), (x2, y2) = sim.pos[u], sim.pos[v]
            ax.plot([x1, x2], [y1, y2], color="#9AA5AD", lw=1.4, zorder=1)

        gxy = [self.xy[n] for n in self.ground]
        self.sc_ground = ax.scatter([p[0] for p in gxy], [p[1] for p in gxy], s=170,
                                    c=[0.0] * len(gxy), cmap="RdYlGn_r",
                                    vmin=0, vmax=self.vmax, edgecolors="#37474F",
                                    linewidths=1.0, zorder=3)
        pr = [self.xy[n] for n in sim.priorities]
        ax.scatter([p[0] for p in pr], [p[1] for p in pr], s=330, facecolors="none",
                   edgecolors="#F9A825", linewidths=3.0, zorder=4)
        hxy = [self.xy[h] for h in self.ghost_ids]
        self.sc_ghost = ax.scatter([p[0] for p in hxy], [p[1] for p in hxy], s=300,
                                   c=[0.0] * len(hxy), cmap="RdYlGn_r", marker="D",
                                   vmin=0, vmax=self.vmax, edgecolors="#0D47A1",
                                   linewidths=2.0, zorder=4)
        for h in self.ghost_ids:
            ax.annotate(h, self.xy[h], textcoords="offset points", xytext=(9, 9),
                        fontsize=8, color="#0D47A1", zorder=6)

        # agents
        self.ugv_pts = ax.scatter([0] * len(sim.ugvs), [0] * len(sim.ugvs), s=170,
                                  marker="s", c="black", zorder=7)
        self.uav_pts = ax.scatter([0] * len(sim.uavs), [0] * len(sim.uavs), s=210,
                                  marker="^", c="#9E9E9E", edgecolors="black",
                                  linewidths=1.0, zorder=8)
        self.ugv_labels = [ax.text(0, 0, g.uid, fontsize=8, color="black", zorder=9,
                                   ha="left", va="bottom") for g in sim.ugvs]
        self.uav_labels = [ax.text(0, 0, a.uid, fontsize=8, color="#1A237E", zorder=9,
                                   ha="left", va="top") for a in sim.uavs]
        self.walk_lines = [ax.plot([], [], ls="--", lw=1.6, color="#1565C0", alpha=0.7,
                                   zorder=2)[0] for _ in sim.ugvs]
        self.uav_lines = [ax.plot([], [], ls=":", lw=1.4, color="#6A1B9A", alpha=0.8,
                                  zorder=2)[0] for _ in sim.uavs]
        self.pin_ring = ax.scatter([], [], s=520, facecolors="none", edgecolors="#6A1B9A",
                                   linewidths=2.5, zorder=10)
        self.tip = ax.annotate("", xy=(0, 0), xytext=(14, 14), textcoords="offset points",
                               bbox=dict(boxstyle="round", fc="white", ec="#455A64", alpha=0.95),
                               fontsize=9, zorder=20)
        self.tip.set_visible(False)

        allx = [p[0] for p in self.xy.values()]
        ally = [p[1] for p in self.xy.values()]
        pad = 60
        ax.set_xlim(min(allx) - pad, max(allx) + pad)
        ax.set_ylim(min(ally) - pad, max(ally) + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        self.title = ax.set_title("", fontsize=11)

        cb = self.fig.colorbar(self.sc_ground, cax=self.ax_cbar)
        cb.set_label("idleness (s since last visit)")
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.yaxis.set_label_position("left")

        legend = [Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
                         markersize=9, label="UGV")]
        names = {UAVState.DOCKED: "UAV docked", UAVState.TO_GHOST: "UAV to ghost",
                 UAVState.SERVICING: "UAV servicing", UAVState.TO_RENDEZVOUS: "UAV to rendezvous",
                 UAVState.WAITING_AT_RENDEZVOUS: "UAV waiting at rendezvous"}
        for st, col in UAV_COLORS.items():
            legend.append(Line2D([0], [0], marker="^", color="w", markerfacecolor=col,
                                 markeredgecolor="black", markersize=9, label=names[st]))
        legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                             markeredgecolor="#F9A825", markeredgewidth=2.5, markersize=10,
                             label="priority node"))
        self.ax_leg.legend(handles=legend, loc="center left", fontsize=8, frameon=False)
        self.txt = self.ax_txt.text(0.0, 1.0, "", va="top", ha="left", fontsize=9,
                                    family="monospace", transform=self.ax_txt.transAxes)

        self.ax_hist.set_xlabel("time (s)", fontsize=8)
        self.ax_hist.set_ylabel("idleness (s)", fontsize=8)
        self.ax_hist.tick_params(labelsize=8)
        self.ax_hist.set_title("click a node to pin its history", fontsize=9)
        (self.hist_line,) = self.ax_hist.plot([], [], color="#6A1B9A", lw=1.5)

        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ----- lookup / interaction ----------------------------------------

    def nearest_node(self, x, y, radius=30.0):
        """Node id whose position is within `radius` metres of (x, y), else None."""
        best, bd = None, radius
        for n, (nx_, ny_) in self.xy.items():
            d = math.hypot(nx_ - x, ny_ - y)
            if d <= bd:
                best, bd = n, d
        return best

    def _node_label(self, n):
        kind = "ghost" if n in self.sim.ghosts else (
            "priority" if n in self.sim.priorities else "ground")
        return f"{n} ({kind})\nidleness {self.sim.idleness(n):.0f} s"

    def _on_move(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            if self.tip.get_visible():
                self.tip.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        n = self.nearest_node(event.xdata, event.ydata)
        if n is None:
            if self.tip.get_visible():
                self.tip.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        self.tip.xy = self.xy[n]
        self.tip.set_text(self._node_label(n))
        self.tip.set_visible(True)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        n = self.nearest_node(event.xdata, event.ydata)
        self.pinned = n
        if n is None:
            self.pin_ring.set_offsets([[float("nan"), float("nan")]])
        else:
            self.pin_ring.set_offsets([self.xy[n]])
        self._update_hist()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "up":
            self.steps_per_frame = int(self.steps_per_frame * 1.5) + 1
        elif event.key == "down":
            self.steps_per_frame = max(1, int(self.steps_per_frame / 1.5))
        elif event.key == "r":
            self.reset()
        elif event.key == "s":
            self.write_node_csv(self.node_csv or "node_idleness.csv")

    def reset(self):
        """Restart the run from t=0 with the same map, variant, fleet and seed."""
        if self.make_sim is None:
            return
        self.sim = self.make_sim()
        self.done = False
        self.paused = False
        self.times = []
        self.hist = {n: [] for n in self.all_ids}
        self.road_sum = 0.0
        self.ghost_sum = 0.0
        self.frames_logged = 0
        self._update_hist()
        self._draw()
        self.fig.canvas.draw_idle()

    # ----- per-frame update --------------------------------------------

    def _update_hist(self):
        if self.pinned is None:
            self.hist_line.set_data([], [])
            self.ax_hist.set_title("click a node to pin its history", fontsize=9)
            return
        ys = self.hist[self.pinned]
        self.hist_line.set_data(self.times[:len(ys)], ys)
        self.ax_hist.set_title(f"idleness of {self.pinned}", fontsize=9)
        self.ax_hist.relim()
        self.ax_hist.autoscale_view()

    def step_frame(self, _frame=None):
        sim = self.sim
        if not self.paused and not self.done:
            for _ in range(self.steps_per_frame):
                if sim.t >= self.horizon:
                    self.done = True
                    break
                sim.step()
            self._record()
            if self.done and self.node_csv:
                self.write_node_csv(self.node_csv)
        self._draw()
        return []

    def _record(self):
        sim = self.sim
        idle = sim.node_idleness()
        self.times.append(sim.t)
        for n in self.all_ids:
            self.hist[n].append(idle[n])
        self.road_sum += sum(idle[n] for n in self.ground) / len(self.ground)
        self.ghost_sum += (sum(idle[h] for h in self.ghost_ids) / len(self.ghost_ids)
                           if self.ghost_ids else 0.0)
        self.frames_logged += 1

    def _draw(self):
        sim = self.sim
        idle = sim.node_idleness()
        self.sc_ground.set_array([idle[n] for n in self.ground])
        self.sc_ghost.set_array([idle[h] for h in self.ghost_ids])

        ug = [sim.ugv_xy(g) for g in sim.ugvs]
        self.ugv_pts.set_offsets(ug)
        for lab, p in zip(self.ugv_labels, ug):
            lab.set_position((p[0] + 8, p[1] + 8))
        for line, g in zip(self.walk_lines, sim.ugvs):
            if g.walk and g.walk_idx < len(g.walk):
                pts = [sim.ugv_xy(g)] + [sim.pos[n] for n in g.walk[g.walk_idx:]]
                line.set_data([p[0] for p in pts], [p[1] for p in pts])
            else:
                line.set_data([], [])

        ua = [sim.uav_xy(a) for a in sim.uavs]
        ua_shown = [(p[0] + 10, p[1] - 10) if a.state == UAVState.DOCKED else p
                    for a, p in zip(sim.uavs, ua)]
        self.uav_pts.set_offsets(ua_shown)
        self.uav_pts.set_facecolor([UAV_COLORS[a.state] for a in sim.uavs])
        for lab, p in zip(self.uav_labels, ua_shown):
            lab.set_position((p[0] + 8, p[1] - 8))
        for line, a, p in zip(self.uav_lines, sim.uavs, ua):
            if a.state in (UAVState.TO_GHOST, UAVState.SERVICING) and a.target_ghost:
                gp = sim.ghosts[a.target_ghost]
                line.set_data([p[0], gp[0]], [p[1], gp[1]])
            elif a.state in (UAVState.TO_RENDEZVOUS, UAVState.WAITING_AT_RENDEZVOUS) \
                    and a.recovery_point is not None:
                line.set_data([p[0], a.recovery_point[0]], [p[1], a.recovery_point[1]])
            else:
                line.set_data([], [])

        road_now = sum(idle[n] for n in self.ground) / len(self.ground)
        ghost_now = (sum(idle[h] for h in self.ghost_ids) / len(self.ghost_ids)
                     if self.ghost_ids else 0.0)
        k = max(self.frames_logged, 1)
        stalest = sorted(self.all_ids, key=lambda n: -idle[n])[:6]
        lines = [
            f"time        {hhmmss(sim.t)} / {hhmmss(self.horizon)}",
            f"speed       {self.steps_per_frame} s per frame",
            f"state       {'FINISHED - press r to replay' if self.done else ('PAUSED' if self.paused else 'running')}",
            "",
            f"ground idleness  now {road_now:7.0f}",
            f"                 avg {self.road_sum / k:7.0f}",
            f"ghost idleness   now {ghost_now:7.0f}",
            f"                 avg {self.ghost_sum / k:7.0f}",
            "",
            "stalest nodes:",
        ] + [f"  {n:<7} {idle[n]:7.0f} s" for n in stalest]
        self.txt.set_text("\n".join(lines))
        self.title.set_text(f"{self.layout.name} | {self.variant_key} | "
                            f"fleet {self.fleet} | colour = idleness (clipped at "
                            f"{self.vmax:.0f} s)")
        if self.pinned is not None:
            self._update_hist()

    # ----- export ------------------------------------------------------

    def write_node_csv(self, path):
        """Per-node idleness: current, time-average and peak so far."""
        stats = self.sim.node_idleness_stats()
        now = self.sim.node_idleness()
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["node", "kind", "x", "y", "idleness_now", "idleness_avg", "idleness_max"])
            for n in self.all_ids:
                kind = "ghost" if n in self.sim.ghosts else (
                    "priority" if n in self.sim.priorities else "ground")
                w.writerow([n, kind, f"{self.xy[n][0]:.1f}", f"{self.xy[n][1]:.1f}",
                            f"{now[n]:.1f}", f"{stats[n]['avg']:.1f}", f"{stats[n]['max']:.1f}"])
        print(f"wrote per-node idleness to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", choices=list(MAPS), default="m1")
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    ap.add_argument("--fleet", default="4-2", help="UGVs-UAVs, e.g. 4-2 or 6-3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=7200.0, help="simulated seconds")
    ap.add_argument("--speed", type=int, default=20, help="simulated seconds per frame")
    ap.add_argument("--vmax", type=float, default=1500.0,
                    help="idleness (s) at which the colour scale saturates")
    ap.add_argument("--T", dest="t_hops", type=float, default=None)
    ap.add_argument("--walks-per-pair", type=int, default=500,
                    help="same approximation as run_m1/run_m2 (0 = exact, very slow)")
    ap.add_argument("--save", default=None, help="write the animation (.gif or .mp4)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--node-csv", default=None,
                    help="write per-node idleness CSV when the run finishes (or press s)")
    args = ap.parse_args()

    map_idx, default_hops = MAPS[args.map]
    t_hops = args.t_hops or default_hops
    k = args.walks_per_pair or None
    n_ugv, n_uav = (int(x) for x in args.fleet.split("-"))

    print("building map and walk set (a few seconds)...", flush=True)
    layout = _layout(map_idx)
    walks, gated_walks = _walks(map_idx, t_hops, k)
    def make_sim():
        return Sim(layout, n_ugv=n_ugv, n_uav=n_uav, variant=VARIANTS[args.variant],
                   seed=args.seed, weights=make_weights(t_hops), valid_walks=walks,
                   gated_walks=gated_walks, track_node_idleness=True)

    sim = make_sim()
    v = Viewer(sim, layout, args.variant, args.fleet, args.vmax, args.speed,
               args.horizon, args.node_csv, make_sim=make_sim)
    n_frames = int(math.ceil(args.horizon / max(args.speed, 1))) + 1

    if args.save:
        # rendering to a file never opens a window
        anim = FuncAnimation(v.fig, v.step_frame, frames=n_frames, interval=1000 / args.fps,
                             blit=False, repeat=False)
        print(f"rendering {n_frames} frames to {args.save} ...", flush=True)
        anim.save(args.save, fps=args.fps, dpi=110)
        print(f"saved {args.save}")
    elif args.no_show:
        # headless, no video: just run the sim (and write the node CSV if asked)
        while sim.t < args.horizon:
            sim.step()
        if args.node_csv:
            v.write_node_csv(args.node_csv)
        print(f"finished {hhmmss(sim.t)}; mean ground idleness (avg over run) "
              f"{sum(r['mean_idleness'] for r in sim.metrics_log) / len(sim.metrics_log):.1f}")
    else:
        anim = FuncAnimation(v.fig, v.step_frame, interval=33, blit=False,
                             cache_frame_data=False)
        v._anim = anim   # keep a reference so it is not garbage collected
        plt.show()


if __name__ == "__main__":
    main()
