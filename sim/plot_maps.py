"""Draw the M1 and M2 maps (no OSM download needed) and save a PNG + PDF.

Usage:
    python3 plot_maps.py
Writes m1_m2_maps.png and m1_m2_maps.pdf in the current folder.
"""
import matplotlib
matplotlib.use("Agg")   # save to file; no window needed
import matplotlib.pyplot as plt

from graph_layouts import make_ladder, plot_layout


def main():
    layouts = make_ladder(include_m3=False)
    legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="#546E7A", markersize=9, label="ground node $V_g$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFC107",
                   markeredgecolor="#8D6E00", markersize=10, label="priority node $S$"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#1565C0",
                   markeredgecolor="white", markersize=10, label="ghost node $V_a$ (air-only)"),
    ]
    fig, axes = plt.subplots(1, len(layouts), figsize=(6 * len(layouts), 6))
    for lay, ax in zip(layouts, axes):
        plot_layout(lay, ax)
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("m1_m2_maps.png", dpi=200, bbox_inches="tight")
    fig.savefig("m1_m2_maps.pdf", bbox_inches="tight")
    print("saved m1_m2_maps.png and m1_m2_maps.pdf")


if __name__ == "__main__":
    main()
