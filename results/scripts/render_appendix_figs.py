#!/usr/bin/env python3
"""Paper-quality appendix figures. Style matches the c0_trajectories
aesthetic: multi-panel layouts, viridis colormap for per-task
trajectories, log scale where it helps, mean ± IQR bands.

Outputs to results/presentation/img/paper/ as figA*.png.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PROCESSED = HERE / "data" / "processed"
RAW = HERE / "data" / "raw"
OUT = HERE / "presentation" / "img" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#0f172a"
INK_LIGHT = "#475569"
GRID = "#e2e8f0"
# Darkened slate so the from-scratch line reads against white.
SLATE = "#64748b"
TEAL = "#0d9488"
AMBER = "#d97706"
ROSE = "#be123c"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10.5,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.9,
    "xtick.color": INK_LIGHT,
    "ytick.color": INK_LIGHT,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

CANONICAL_TASKS = [
    (0, "hammer"), (1, "push-wall"), (2, "faucet-close"),
    (3, "push-back"), (4, "stick-pull"), (5, "handle-press-side"),
    (6, "push"), (7, "shelf-place"), (8, "window-close"),
    (9, "peg-unplug-side"),
]

CELLS_FULL = {
    "actor=reset-critic=reset":              ("From-scratch (R/R)",       SLATE,  ("for_real",)),
    "actor=persistent-critic=persistent":    ("Persistent (P/P)",         TEAL,   ("for_real",)),
    "actor=cka-critic=reset":                ("CKA (C/R)",                AMBER,  ("real1", "real2")),
    "actor=cka-critic=cka":                  ("CKA-mix (C/C)",            "#7c3aed",  ("real1", "real2")),
}


def setup_ax(ax, log=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if log:
        ax.set_yscale("log")
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def load_rl(cells=None):
    parts = []
    for g in ("for_real", "real1", "real2"):
        p = RAW / g / "rl_metrics.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df["cell"] = "actor=" + df["actor_mode"] + "-critic=" + df["critic_mode"]

    # Apply routing
    def keep(row):
        if row["cell"] in CELLS_FULL:
            return row["group"] in CELLS_FULL[row["cell"]][2]
        return False
    return df[df.apply(keep, axis=1)].reset_index(drop=True)


# ============== FIGURE A2: per-task rl_metrics trajectories ==============

def fig_rank_trajectories(save_to):
    """For each of the four baseline cells, plot the actor feature_rank
    and the critic-sa feature_rank across SGD steps within a task,
    with one curve per task colour-coded by task index (viridis).
    Mean across seeds bold; IQR band lightly shaded.

    Layout: 4 rows (cells) x 2 columns (actor / critic-sa rank).
    """
    df = load_rl()
    cmap = plt.get_cmap("viridis")
    tasks = list(range(10))
    color_for = {k: cmap(k / 9.0) for k in tasks}

    fig, axes = plt.subplots(4, 2, figsize=(10.5, 11.5), sharex=True)
    metrics = [
        ("rl_metrics/actor/feature_rank", "Actor rank"),
        ("rl_metrics/critic_sa/feature_rank", "Critic state-action rank"),
    ]
    cells = list(CELLS_FULL.keys())
    for row, cell in enumerate(cells):
        label, cell_color, _ = CELLS_FULL[cell]
        sub = df[df["cell"] == cell].copy()
        for col, (mcol, mtitle) in enumerate(metrics):
            ax = axes[row, col]
            for k in tasks:
                tk = sub[sub["task_idx"] == k]
                if tk.empty:
                    continue
                # Group by env_steps, get IQR
                pv = tk.pivot_table(index="rl_metrics/env_steps",
                                    columns="seed", values=mcol).sort_index()
                # Normalise to "SGD step within task" by rank-ordering env_steps
                steps = np.arange(len(pv))
                mean = pv.mean(axis=1).to_numpy()
                p25 = pv.quantile(0.25, axis=1).to_numpy()
                p75 = pv.quantile(0.75, axis=1).to_numpy()
                ax.fill_between(steps, p25, p75, color=color_for[k], alpha=0.10, zorder=2)
                ax.plot(steps, mean, color=color_for[k], lw=1.4, alpha=0.95, zorder=5)
            setup_ax(ax)
            if col == 0:
                ax.set_ylabel(label, fontsize=10.5, fontweight="bold", color=cell_color)
            if row == 0:
                ax.set_title(mtitle, loc="left", pad=6)
            if row == 3:
                ax.set_xlabel("rl-metric sample step within task\n(≈31 samples · ≈ 250 k env steps each)",
                              fontsize=9.5)

    # Single colorbar legend across the right side
    sm = cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(vmin=0, vmax=9))
    cb = fig.colorbar(sm, ax=axes, shrink=0.65, pad=0.025, ticks=range(10))
    cb.set_label("task index $k$", fontsize=10, labelpad=8)
    cb.ax.set_yticklabels([f"{k} · {n}" for k, n in CANONICAL_TASKS], fontsize=8)
    cb.outline.set_visible(False)

    fig.savefig(save_to)
    plt.close(fig)


# ============== FIGURE A3: weight-norm cumulative drift ==============

def fig_weight_norm_drift(save_to):
    """The persistent cell shows an 8x weight_norm blow-up across the
    curriculum; reset stays bounded. Show actor & critic weight norms
    end-of-task across the 10 tasks for each cell.
    """
    df = load_rl()
    end = df.sort_values("rl_metrics/env_steps").drop_duplicates(
        subset=["run_id"], keep="last")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    metrics = [
        ("rl_metrics/actor/weight_norm",  "Actor weight norm"),
        ("rl_metrics/critic/weight_norm", "Critic weight norm"),
    ]
    cells = list(CELLS_FULL.keys())
    n_cells = len(cells)
    dx = 0.10
    for ax, (mcol, ytitle) in zip(axes, metrics):
        for i, cell in enumerate(cells):
            label, color, _ = CELLS_FULL[cell]
            sub = end[end["cell"] == cell]
            g = sub.groupby("task_idx")[mcol].agg(["mean", "std", "count"]).reindex(range(10))
            means = g["mean"].to_numpy()
            counts = g["count"].fillna(0).to_numpy()
            stds = g["std"].fillna(0).to_numpy()
            sem = np.where(counts > 1, stds / np.sqrt(np.maximum(counts, 1)), 0)
            offset = (i - (n_cells - 1) / 2) * dx
            x = np.arange(10) + offset
            ax.plot(x, means, color=color, lw=1.8, marker="o", ms=5,
                    label=label, zorder=5)
            ax.errorbar(x, means, yerr=sem, fmt="none", ecolor=color,
                        alpha=0.65, capsize=2.5, lw=1.0, zorder=4)
        setup_ax(ax)
        ax.set_ylabel(ytitle)
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"{k}\n{n}" for k, n in CANONICAL_TASKS],
                           rotation=20, ha="right", fontsize=8.5)
        for k in (5, 8, 9):
            ax.axvspan(k - 0.45, k + 0.45, color="#fff8f0", zorder=0)
    axes[0].legend(loc="upper left")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.93, wspace=0.20)
    fig.savefig(save_to)
    plt.close(fig)


# ============== FIGURE A4: NRC1 / NRC2 panel ==============

def fig_nrc_panel(save_to):
    """Two-panel: NRC1 (within-class, lower=collapse) and NRC2
    (between-class, higher=collapse) across the 10 tasks. State-action
    representation. Shows the neural-collapse signature of persistent
    critic vs the healthy baselines.
    """
    df = load_rl()
    end = df.sort_values("rl_metrics/env_steps").drop_duplicates(
        subset=["run_id"], keep="last")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    metrics = [
        ("rl_metrics/critic_sa/nrc1", "NRC1\nlower = more collapsed"),
        ("rl_metrics/critic_sa/nrc2", "NRC2\nhigher = more separated"),
    ]
    cells = list(CELLS_FULL.keys())
    n_cells = len(cells)
    dx = 0.10
    for ax, (mcol, ytitle) in zip(axes, metrics):
        for i, cell in enumerate(cells):
            label, color, _ = CELLS_FULL[cell]
            sub = end[end["cell"] == cell]
            g = sub.groupby("task_idx")[mcol].agg(["mean", "std", "count"]).reindex(range(10))
            means = g["mean"].to_numpy()
            counts = g["count"].fillna(0).to_numpy()
            stds = g["std"].fillna(0).to_numpy()
            sem = np.where(counts > 1, stds / np.sqrt(np.maximum(counts, 1)), 0)
            offset = (i - (n_cells - 1) / 2) * dx
            x = np.arange(10) + offset
            ax.plot(x, means, color=color, lw=1.8, marker="o", ms=5,
                    label=label, zorder=5)
            ax.errorbar(x, means, yerr=sem, fmt="none", ecolor=color,
                        alpha=0.65, capsize=2.5, lw=1.0, zorder=4)
        setup_ax(ax)
        ax.set_ylabel(ytitle)
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"{k}\n{n}" for k, n in CANONICAL_TASKS],
                           rotation=20, ha="right", fontsize=8.5)
        for k in (5, 8, 9):
            ax.axvspan(k - 0.45, k + 0.45, color="#fff8f0", zorder=0)
    axes[0].legend(loc="lower left")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.93, wspace=0.20)
    fig.savefig(save_to)
    plt.close(fig)


# ============== FIGURE A5: actor entropy curriculum drift ==============

def fig_entropy_drift(save_to):
    """Actor policy entropy across the 10 tasks. Persistent shows the
    cumulative entropy collapse signature. Single panel."""
    df = load_rl()
    end = df.sort_values("rl_metrics/env_steps").drop_duplicates(
        subset=["run_id"], keep="last")

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    cells = list(CELLS_FULL.keys())
    n_cells = len(cells)
    dx = 0.10
    for i, cell in enumerate(cells):
        label, color, _ = CELLS_FULL[cell]
        sub = end[end["cell"] == cell]
        g = sub.groupby("task_idx")["rl_metrics/actor/entropy"].agg(["mean", "std", "count"]).reindex(range(10))
        means = g["mean"].to_numpy()
        counts = g["count"].fillna(0).to_numpy()
        stds = g["std"].fillna(0).to_numpy()
        sem = np.where(counts > 1, stds / np.sqrt(np.maximum(counts, 1)), 0)
        offset = (i - (n_cells - 1) / 2) * dx
        x = np.arange(10) + offset
        ax.plot(x, means, color=color, lw=1.8, marker="o", ms=5,
                label=label, zorder=5)
        ax.errorbar(x, means, yerr=sem, fmt="none", ecolor=color,
                    alpha=0.65, capsize=2.5, lw=1.0, zorder=4)
    setup_ax(ax)
    ax.set_ylabel("Actor policy entropy")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{k}\n{n}" for k, n in CANONICAL_TASKS],
                       rotation=20, ha="right", fontsize=8.5)
    for k in (5, 8, 9):
        ax.axvspan(k - 0.45, k + 0.45, color="#fff8f0", zorder=0)
    ax.legend(loc="lower left")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.24, top=0.96)
    fig.savefig(save_to)
    plt.close(fig)


def main():
    fig_rank_trajectories(OUT / "figA2_rank_trajectories.png")
    fig_weight_norm_drift(OUT / "figA3_weight_norm_drift.png")
    fig_nrc_panel(OUT / "figA4_nrc_panel.png")
    fig_entropy_drift(OUT / "figA5_entropy_drift.png")
    print("wrote:", sorted(p.name for p in OUT.glob("figA*.png")))


if __name__ == "__main__":
    main()
