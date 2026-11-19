"""Draw M1 alone (same style as graph_layouts' locked figure). Writes m1_map.pdf.
Usage: python3 plot_m1_map.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graph_layouts import make_ladder, plot_layout

legend = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#546E7A", markersize=9, label="ground node $V_g$"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFC107",
               markeredgecolor="#8D6E00", markersize=10, label="priority node $S$"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#1565C0",
               markeredgecolor="white", markersize=10, label="ghost node $V_a$"),
]
fig, ax = plt.subplots(figsize=(5.6, 5.4))
plot_layout(make_ladder(include_m3=False)[0], ax)
fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("m1_map.pdf", bbox_inches="tight")
print("saved m1_map.pdf")
