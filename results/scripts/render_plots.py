#!/usr/bin/env python3
"""Render the trajectory and bar plots embedded in the presentation HTML.

Outputs:
  results/presentation/img/
    bar_per_task.png      -- per-task best-success: decomposed vs reset/reset
                              (and persistent/persistent), with error bars.
    delta_per_task.png    -- per-task gap (decomposed minus reset/reset).
    trajectories_hard.png -- per-step success_rate trajectories on
                              k = 5 (handle_press_side), k = 8 (window_close),
                              k = 9 (peg_unplug_side).
    avg_best_bar.png      -- avg_best across the 10 tasks, all cells.

All PNGs are 1600x900 at 150 dpi so they look crisp embedded in the HTML.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

HERE = Path(__file__).resolve().parents[1]
PROCESSED = HERE / "data" / "processed"
RAW = HERE / "data" / "raw"
IMG = HERE / "presentation" / "img"
IMG.mkdir(parents=True, exist_ok=True)

# Color palette: a deep ink for baselines + warm accent for decomposed.
INK = "#1f2937"
SLATE = "#94a3b8"
INDIGO = "#4338ca"
TEAL = "#0d9488"
AMBER = "#d97706"
ROSE = "#e11d48"

# Match the renderer's routing.
CELLS = {
    "actor=reset-critic=reset": ("for_real", "Reset / Reset (from-scratch)"),
    "actor=persistent-critic=persistent": ("for_real", "Persistent / Persistent"),
    "actor=cka-critic=reset": ("real1+real2", "CKA / Reset (best 9-cell)"),
    "actor=reset-critic=decomposed": ("c2_decomposed", "Decomposed (ours)"),
}
GROUP_BUCKETS = {
    "for_real": ("for_real",),
    "real1+real2": ("real1", "real2"),
    "c2_decomposed": ("c2_decomposed",),
}

CANONICAL_TASKS = [
    (0, "hammer"), (1, "push-wall"), (2, "faucet-close"),
    (3, "push-back"), (4, "stick-pull"), (5, "handle-press-side"),
    (6, "push"), (7, "shelf-place"), (8, "window-close"),
    (9, "peg-unplug-side"),
]


# ---------------- data loaders ----------------

def load_per_seed() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "per_seed_per_task.csv")
    return df


def routed_per_seed(df: pd.DataFrame, cell: str) -> pd.DataFrame:
    bucket, _label = CELLS[cell]
    groups = GROUP_BUCKETS[bucket]
    sub = df[(df["cell"] == cell) & (df["group"].isin(groups))].copy()
    sub = sub.sort_values("n_evals", ascending=False).drop_duplicates(
        subset=["seed", "task_idx"], keep="first")
    return sub


def per_task_stats(df: pd.DataFrame, cell: str) -> pd.DataFrame:
    sub = routed_per_seed(df, cell)
    g = sub.groupby("task_idx")["best_success"].agg(["mean", "std", "count"])
    g = g.reindex(range(10))  # ensure all tasks present
    return g


def histories_for_cell(cell: str) -> pd.DataFrame:
    bucket, _ = CELLS[cell]
    groups = GROUP_BUCKETS[bucket]
    frames = []
    for g in groups:
        p = RAW / g / "histories.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d = d[d["cell" if "cell" in d.columns else "actor_mode"].notna()]
            # we stored actor/critic columns; reconstruct cell
            if "cell" not in d.columns:
                d["cell"] = "actor=" + d["actor_mode"].astype(str) + "-critic=" + d["critic_mode"].astype(str)
            d = d[d["cell"] == cell]
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------- plot helpers ----------------

def setup_axes(ax, *, ymax=1.05):
    ax.set_ylim(0, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors=INK, labelsize=11)
    ax.yaxis.grid(True, color="#e2e8f0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


# ---------------- 1: per-task grouped bars ----------------

def plot_per_task_bars(df: pd.DataFrame, save_to: Path) -> None:
    cells = [
        ("actor=reset-critic=reset", "Reset / Reset", SLATE),
        ("actor=persistent-critic=persistent", "Persistent / Persistent", AMBER),
        ("actor=cka-critic=reset", "CKA / Reset", INDIGO),
        ("actor=reset-critic=decomposed", "Decomposed (ours)", ROSE),
    ]
    stats = {cell: per_task_stats(df, cell) for cell, _, _ in cells}

    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    n_cells = len(cells)
    width = 0.21
    x = np.arange(10)
    for i, (cell, label, color) in enumerate(cells):
        offset = (i - (n_cells - 1) / 2) * width
        s = stats[cell]
        means = s["mean"].fillna(0).to_numpy()
        stds = s["std"].fillna(0).to_numpy()
        ax.bar(x + offset, means, width, color=color, label=label,
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    ecolor=INK, alpha=0.55, capsize=2.5, lw=1.0, zorder=4)
    setup_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}\n{name}" for k, name in CANONICAL_TASKS],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Best mean success rate")
    ax.set_title("Per-task best mean success across the 10 Sawyer tasks",
                 loc="left", pad=14)
    ax.legend(frameon=False, ncols=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    fig.savefig(save_to, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- 2: delta per task ----------------

def plot_delta_per_task(df: pd.DataFrame, save_to: Path) -> None:
    base = per_task_stats(df, "actor=reset-critic=reset")
    ours = per_task_stats(df, "actor=reset-critic=decomposed")
    delta = ours["mean"] - base["mean"]
    # SE of unpaired difference of means
    se = np.sqrt((ours["std"] ** 2) / ours["count"]
                 + (base["std"] ** 2) / base["count"])

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    colors = [ROSE if d > 0 else SLATE for d in delta]
    x = np.arange(10)
    ax.bar(x, delta.fillna(0), color=colors, edgecolor="white",
           linewidth=0.5, zorder=3)
    ax.errorbar(x, delta.fillna(0), yerr=se.fillna(0), fmt="none",
                ecolor=INK, alpha=0.55, capsize=3, lw=1.0, zorder=4)
    ax.axhline(0, color=INK, lw=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors=INK, labelsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}\n{name}" for k, name in CANONICAL_TASKS],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Δ best mean success rate")
    ax.set_title("Decomposed minus Reset/Reset (unpaired mean difference)",
                 loc="left", pad=14)
    ax.set_ylim(-0.1, 0.32)
    ax.yaxis.grid(True, color="#e2e8f0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    # Annotate the big wins.
    for k in (5, 8):
        ax.annotate(f"+{delta[k]:.2f}",
                    xy=(k, delta[k]),
                    xytext=(k, delta[k] + 0.035),
                    ha="center", color=ROSE, fontsize=12, fontweight="bold")
    fig.savefig(save_to, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- 3: trajectories on hard tasks ----------------

def plot_trajectories(save_to: Path) -> None:
    cells = [
        ("actor=reset-critic=reset", "Reset / Reset", SLATE),
        ("actor=persistent-critic=persistent", "Persistent / Persistent", AMBER),
        ("actor=cka-critic=reset", "CKA / Reset", INDIGO),
        ("actor=reset-critic=decomposed", "Decomposed (ours)", ROSE),
    ]
    hard = [(5, "handle-press-side"), (8, "window-close"), (9, "peg-unplug-side")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (k, env) in zip(axes, hard):
        for cell, label, color in cells:
            h = histories_for_cell(cell)
            h = h[h["task_idx"] == k]
            if h.empty:
                continue
            # Bin by env_steps in 0.2M buckets and average across seeds.
            h2 = h.copy()
            h2["bin"] = (h2["env_steps"] / 2e5).round() * 2e5
            piv = h2.pivot_table(index="bin", columns="seed",
                                  values="success_rate", aggfunc="mean")
            piv = piv.sort_index()
            # Rolling mean for visual clarity (window of 3 bins).
            piv_smooth = piv.rolling(window=3, min_periods=1).mean()
            # Dim per-seed curves.
            for seed in piv_smooth.columns:
                ax.plot(piv_smooth.index / 1e6, piv_smooth[seed],
                        color=color, alpha=0.16, lw=1.0)
            mean = piv_smooth.mean(axis=1)
            ax.plot(mean.index / 1e6, mean.values, color=color, lw=2.6,
                    label=label, zorder=5)
        ax.set_title(f"k={k}: {env}", loc="left")
        ax.set_xlabel("Env steps (M)")
        setup_axes(ax)
        if ax is axes[0]:
            ax.set_ylabel("Success rate (eval)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    fig.suptitle("Training trajectories on the three hardest tasks",
                 x=0.02, ha="left", fontsize=14, color=INK, y=1.02)
    fig.savefig(save_to, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------- 4: avg_best across cells ----------------

def plot_avg_best(df: pd.DataFrame, save_to: Path) -> None:
    cell_order = [
        "actor=reset-critic=reset",
        "actor=persistent-critic=reset",
        "actor=persistent-critic=persistent",
        "actor=reset-critic=persistent",
        "actor=reset-critic=cka",
        "actor=cka-critic=reset",
        "actor=cka-critic=persistent",
        "actor=cka-critic=cka",
        "actor=persistent-critic=cka",
        "actor=reset-critic=decomposed",
    ]
    labels_short = {
        "actor=reset-critic=reset": "R / R",
        "actor=persistent-critic=reset": "P / R",
        "actor=persistent-critic=persistent": "P / P",
        "actor=reset-critic=persistent": "R / P",
        "actor=reset-critic=cka": "R / CKA",
        "actor=cka-critic=reset": "CKA / R",
        "actor=cka-critic=persistent": "CKA / P",
        "actor=cka-critic=cka": "CKA / CKA",
        "actor=persistent-critic=cka": "P / CKA",
        "actor=reset-critic=decomposed": "Decomposed (ours)",
    }
    means = []
    for cell in cell_order:
        # cell may not be in CELLS map; build routing on the fly using CELLS keys
        # by reconstructing routing as in render_tables. We just reuse a fresh
        # pool here.
        if cell == "actor=reset-critic=decomposed":
            bucket = ("c2_decomposed",)
        elif "cka" in cell.split("-critic=")[0].split("=")[1]:
            bucket = ("real1", "real2")
        else:
            bucket = ("for_real",)
        sub = df[(df["cell"] == cell) & (df["group"].isin(bucket))].copy()
        sub = sub.sort_values("n_evals", ascending=False).drop_duplicates(
            subset=["seed", "task_idx"], keep="first")
        per_task = sub.groupby("task_idx")["best_success"].mean()
        means.append((cell, per_task.mean()))

    labels = [labels_short[c] for c, _ in means]
    vals = [v for _, v in means]
    colors_by = [ROSE if c == "actor=reset-critic=decomposed" else SLATE
                 for c, _ in means]

    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(means))
    ax.bar(x, vals, color=colors_by, edgecolor="white", linewidth=0.5, zorder=3)
    for xi, vi in zip(x, vals):
        ax.text(xi, vi + 0.012, f"{vi:.3f}", ha="center",
                color=INK, fontsize=10)
    setup_axes(ax, ymax=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10.5)
    ax.set_ylabel("Best mean success (avg over 10 tasks)")
    ax.set_title("Average best success across the 10 Sawyer tasks", loc="left", pad=14)
    fig.savefig(save_to, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df = load_per_seed()
    plot_avg_best(df, IMG / "avg_best_bar.png")
    plot_per_task_bars(df, IMG / "bar_per_task.png")
    plot_delta_per_task(df, IMG / "delta_per_task.png")
    plot_trajectories(IMG / "trajectories_hard.png")
    print("wrote:", sorted(p.name for p in IMG.glob("*.png")))


if __name__ == "__main__":
    main()
