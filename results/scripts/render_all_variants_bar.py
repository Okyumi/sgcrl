"""Wide bar figure: avg best-during-training success across the 10 Sawyer
tasks for every variant. SAC on the left, contrastive 9-cell grid in the
middle, proposed Decomposed Contrastive Critic D on the right."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT_PAPER = Path("/tmp/paper_rlc/figs")
OUT_PAPER.mkdir(parents=True, exist_ok=True)
OUT_ARCH = Path("/home/user/workspace/sgcrl/results/presentation/img/paper")
OUT_ARCH.mkdir(parents=True, exist_ok=True)

# (label, avg_success, family)
ROWS = [
    ("SAC R/R", 0.42, "sac"),
    ("SAC P/P", 0.44, "sac"),
    ("R/R", 0.83, "contrastive"),
    ("R/P", 0.77, "contrastive"),
    ("P/R", 0.83, "contrastive"),
    ("P/P", 0.79, "contrastive"),
    ("R/C", 0.84, "contrastive"),
    ("C/R", 0.84, "contrastive"),
    ("P/C", 0.76, "contrastive"),
    ("C/P", 0.80, "contrastive"),
    ("C/C", 0.78, "contrastive"),
    ("D",   0.87, "proposed"),
]

COLORS = {
    "contrastive": "#7b8190",
    "proposed":    "#2ca58d",
    "sac":         "#d4a64a",
}

labels = [r[0] for r in ROWS]
vals = np.array([r[1] for r in ROWS])
cols = [COLORS[r[2]] for r in ROWS]

fig, ax = plt.subplots(figsize=(12.5, 4.0))
x = np.arange(len(ROWS))
bars = ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.6, width=0.72)

for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}",
            ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Avg best-during-training success", fontsize=11)
ax.set_ylim(0.0, 1.08)
ax.set_yticks(np.linspace(0, 1, 6))
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    Patch(facecolor=COLORS["sac"], edgecolor="black",
          label="Goal-conditioned SAC"),
    Patch(facecolor=COLORS["contrastive"], edgecolor="black",
          label="Contrastive baseline grid (9 cells)"),
    Patch(facecolor=COLORS["proposed"], edgecolor="black",
          label="Decomposed Contrastive Critic (proposed)"),
]
ax.legend(handles=legend_handles, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), ncol=3,
          frameon=False, fontsize=10)

ax.axvline(1.5, color="black", linewidth=0.5, alpha=0.4, linestyle="--")
ax.axvline(10.5, color="black", linewidth=0.5, alpha=0.4, linestyle="--")

fig.tight_layout()
for p in (OUT_PAPER / "figA_all_variants_bar.png",
          OUT_ARCH / "figA_all_variants_bar.png"):
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"wrote {p}")
plt.close(fig)
