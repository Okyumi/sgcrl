"""Per-task line plot across every variant.

Ten Sawyer tasks on the x-axis, one line per variant. Three families:
SAC (warm, dashed), contrastive 9-cell grid (cool, faint), and the
proposed Decomposed Contrastive Critic D (bold teal, on top).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT_PAPER = Path("/tmp/paper_rlc/figs")
OUT_PAPER.mkdir(parents=True, exist_ok=True)
OUT_ARCH = Path("/home/user/workspace/sgcrl/results/presentation/img/paper")
OUT_ARCH.mkdir(parents=True, exist_ok=True)

TASKS = [
    "hammer", "push-wall", "faucet-close", "push-back", "stick-pull",
    "handle-press-side", "push", "shelf-place", "window-close",
    "peg-unplug-side",
]

# Per-task best mean success, copied from tab:full-grid.
SAC = {
    "SAC R/R": [1.00, 0.00, 0.70, 0.00, 1.00, 0.53, 0.07, 0.00, 0.80, 0.07],
    "SAC P/P": [1.00, 0.13, 0.80, 0.00, 0.67, 0.50, 0.30, 0.00, 0.80, 0.17],
}
CONTRASTIVE = {
    "R/R": [1.00, 1.00, 0.92, 1.00, 1.00, 0.24, 0.98, 0.98, 0.37, 0.83],
    "R/P": [1.00, 1.00, 0.84, 1.00, 1.00, 0.12, 1.00, 0.68, 0.43, 0.67],
    "P/R": [1.00, 1.00, 0.86, 1.00, 1.00, 0.24, 1.00, 0.94, 0.40, 0.83],
    "P/P": [1.00, 1.00, 0.88, 1.00, 1.00, 0.20, 1.00, 0.66, 0.37, 0.80],
    "R/C": [1.00, 1.00, 0.84, 1.00, 1.00, 0.20, 1.00, 0.98, 0.48, 0.90],
    "C/R": [1.00, 1.00, 0.92, 1.00, 0.98, 0.26, 1.00, 0.98, 0.40, 0.87],
    "P/C": [1.00, 1.00, 0.82, 1.00, 1.00, 0.12, 0.98, 0.66, 0.37, 0.67],
    "C/P": [1.00, 1.00, 0.90, 1.00, 1.00, 0.20, 1.00, 0.78, 0.38, 0.73],
    "C/C": [1.00, 1.00, 0.90, 1.00, 0.90, 0.18, 0.95, 0.75, 0.37, 0.77],
}
D_NAME = "D (proposed)"
D_VALS = [1.00, 1.00, 0.87, 1.00, 1.00, 0.40, 1.00, 0.97, 0.63, 0.87]

# Aesthetics
COLOR_D = "#1f8f76"            # deep teal
COLOR_SAC = ["#d4823a", "#b86528"]  # two warm tones
# Cool palette for the nine contrastive cells.
cm = plt.get_cmap("Blues")
CONTR_COLORS = [cm(0.30 + 0.55 * i / 8) for i in range(9)]

HARD_TASKS = {5, 8, 9}  # shade for visual continuity with other figures

x = np.arange(len(TASKS))

fig, ax = plt.subplots(figsize=(13.0, 5.0))

# Hard-task shading.
for t in HARD_TASKS:
    ax.axvspan(t - 0.45, t + 0.45, color="#f4d9a3", alpha=0.22, zorder=0)

# Contrastive baselines (back).
for (name, vals), c in zip(CONTRASTIVE.items(), CONTR_COLORS):
    ax.plot(x, vals, color=c, linewidth=1.4, alpha=0.85, marker="o",
            markersize=3.5, label=name, zorder=2)

# SAC (mid).
for (name, vals), c in zip(SAC.items(), COLOR_SAC):
    ax.plot(x, vals, color=c, linewidth=1.8, alpha=0.95, linestyle="--",
            marker="s", markersize=4.5, label=name, zorder=3)

# D on top.
ax.plot(x, D_VALS, color=COLOR_D, linewidth=3.0, marker="o",
        markersize=8.0, markerfacecolor=COLOR_D,
        markeredgecolor="white", markeredgewidth=1.2,
        label=D_NAME, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([f"{i}\n{t}" for i, t in enumerate(TASKS)], fontsize=10)
ax.set_xlabel("Task in continual curriculum", fontsize=11, labelpad=8)
ax.set_ylabel("Best mean success", fontsize=11)
ax.set_ylim(-0.03, 1.05)
ax.set_yticks(np.linspace(0, 1, 6))
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-0.5, len(TASKS) - 0.5)

# Two-column legend grouped by family: SAC + D first column, contrastive second.
handles, labels = ax.get_legend_handles_labels()
# Order: D, SAC R/R, SAC P/P, then nine contrastive cells.
order = ([labels.index(D_NAME)]
         + [labels.index(n) for n in SAC]
         + [labels.index(n) for n in CONTRASTIVE])
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
ax.legend(handles, labels, loc="lower center",
          bbox_to_anchor=(0.5, 1.02), ncol=6,
          frameon=False, fontsize=9.5, columnspacing=1.4,
          handlelength=2.2)

fig.tight_layout()
for p in (OUT_PAPER / "figA_all_variants_per_task.png",
          OUT_ARCH / "figA_all_variants_per_task.png"):
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"wrote {p}")
plt.close(fig)
