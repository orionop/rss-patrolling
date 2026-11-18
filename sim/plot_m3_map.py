"""Draw the M3 map (Boston OSM, from the cache in ./cache) in the same style
as plot_maps.py and save m3_map.png + m3_map.pdf.

Usage:
    python3 plot_m3_map.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph_layouts import make_ladder, plot_layout


def main():
    lay = make_ladder(include_m3=True)[2]
    legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="#546E7A", markersize=9, label="ground node $V_g$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFC107",
                   markeredgecolor="#8D6E00", markersize=10, label="priority node $S$"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#1565C0",
                   markeredgecolor="white", markersize=10, label="ghost node $V_a$ (air-only)"),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_layout(lay, ax)
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("m3_map.png", dpi=200, bbox_inches="tight")
    fig.savefig("m3_map.pdf", bbox_inches="tight")
    print("saved m3_map.png and m3_map.pdf")


if __name__ == "__main__":
    main()
