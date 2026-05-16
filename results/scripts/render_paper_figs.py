#!/usr/bin/env python3
"""Paper-quality figures for the RLC workshop submission.

Outputs to `paper_figs/` (siblings of the LaTeX source); also written
to `results/presentation/img/paper/` for archival.

Consistent design across all figures:
  - Single shared palette (slate baseline, accent for ours, two
    qualitative colours for the other two baselines).
  - Single shared font (DejaVu Sans Bold for titles, DejaVu Sans
    Regular for labels).
  - Same axis styling, same legend treatment, same grid.

We deliberately keep this to a small number of figures -- the paper's
job is to make a focused argument, not to display everything.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PROCESSED = HERE / "data" / "processed"
RAW = HERE / "data" / "raw"
OUT = HERE / "presentation" / "img" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- design ----------------
INK = "#0f172a"
INK_LIGHT = "#475569"
GRID = "#e2e8f0"
SLATE = "#94a3b8"       # baseline 1 (R/R)
TEAL = "#0d9488"        # baseline 2 (P/P)
AMBER = "#d97706"       # baseline 3 (CKA/R)
ROSE = "#be123c"        # ours (decomposed)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.9,
    "xtick.color": INK_LIGHT,
    "ytick.color": INK_LIGHT,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
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

CELLS = {
    "actor=reset-critic=reset":              ("for_real",       "From-scratch (R/R)",        SLATE),
    "actor=persistent-critic=persistent":    ("for_real",       "Persistent (P/P)",         TEAL),
    "actor=cka-critic=reset":                ("real1+real2",    "CKA (C/R, prior CL)",      AMBER),
    "actor=reset-critic=decomposed":         ("c2_decomposed",  "Decomposed (ours)",         ROSE),
}
BUCKETS = {"for_real": ("for_real",),
           "real1+real2": ("real1", "real2"),
           "c2_decomposed": ("c2_decomposed",)}


# ---------------- helpers ----------------

def setup_ax(ax, ymax=1.05, ymin=0):
    ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def task_xticks(ax):
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{k}\n{n}" for k, n in CANONICAL_TASKS],
                       rotation=20, ha="right", fontsize=8.5)


def load_per_seed():
    return pd.read_csv(PROCESSED / "per_seed_per_task.csv")


def routed(df, cell):
    bucket, _, _ = CELLS[cell]
    groups = BUCKETS[bucket]
    sub = df[(df["cell"] == cell) & (df["group"].isin(groups))].copy()
    sub = sub.sort_values("n_evals", ascending=False).drop_duplicates(
        subset=["seed", "task_idx"], keep="first")
    return sub


def per_task_stats(df, cell, metric="best_success"):
    sub = routed(df, cell)
    g = sub.groupby("task_idx")[metric].agg(["mean", "std", "count"]).reindex(range(10))
    return g


def load_histories(cell, task_idx):
    bucket, _, _ = CELLS[cell]
    groups = BUCKETS[bucket]
    frames = []
    for g in groups:
        p = RAW / g / "histories.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if "cell" not in d.columns:
            d["cell"] = "actor=" + d["actor_mode"].astype(str) + "-critic=" + d["critic_mode"].astype(str)
        d = d[(d["cell"] == cell) & (d["task_idx"] == task_idx)]
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_rl_metrics():
    parts = []
    for g in ("for_real", "real1", "real2"):
        p = RAW / g / "rl_metrics.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    df["cell"] = "actor=" + df["actor_mode"] + "-critic=" + df["critic_mode"]
    def route(row):
        if row["cell"] in {"actor=reset-critic=reset", "actor=persistent-critic=persistent"}:
            return row["group"] == "for_real"
        if row["actor_mode"] == "cka":
            return row["group"] in ("real1", "real2")
        return row["group"] == "for_real"
    return df[df.apply(route, axis=1)].reset_index(drop=True)


# ============== FIGURE 1: per-task headline ==============

def fig_per_task(df_per_seed, save_to):
    """Grouped bar chart: 4 cells x 10 tasks, with the decomposed cell
    visually leading. The hard tasks (k=5, 8, 9) are shaded for emphasis.
    """
    cells = list(CELLS.keys())
    fig, ax = plt.subplots(figsize=(11, 4.2))

    # Shade the hard tasks.
    for k in (5, 8, 9):
        ax.axvspan(k - 0.45, k + 0.45, color="#fff8f0", zorder=0)

    width = 0.20
    x = np.arange(10)
    for i, cell in enumerate(cells):
        _, label, color = CELLS[cell]
        s = per_task_stats(df_per_seed, cell)
        means = s["mean"].fillna(0).to_numpy()
        stds = s["std"].fillna(0).to_numpy()
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, color=color, label=label,
                      edgecolor="white", linewidth=0.6, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    ecolor=INK_LIGHT, alpha=0.5, capsize=2.0, lw=0.8, zorder=4)

    setup_ax(ax, ymax=1.08)
    task_xticks(ax)
    ax.set_ylabel("Success rate (best during training)")
    ax.text(5, 1.04, "hardest tasks", ha="center", color="#9f1239",
            fontsize=9, style="italic")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=4,
              columnspacing=1.5, handlelength=1.2)

    fig.savefig(save_to)
    plt.close(fig)


# ============== FIGURE 2: hard-task trajectories ==============

def fig_trajectories(save_to):
    """Three-panel trajectory plot on the hard tasks. Mean across seeds
    bold; per-seed dim. Clean rolling-mean smoothing."""
    cells = list(CELLS.keys())
    hard = [(5, "handle-press-side"), (8, "window-close"), (9, "peg-unplug-side")]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for ax, (k, env) in zip(axes, hard):
        for cell in cells:
            _, label, color = CELLS[cell]
            h = load_histories(cell, k)
            if h.empty:
                continue
            h2 = h.copy()
            h2["bin"] = (h2["env_steps"] / 2e5).round() * 2e5
            piv = h2.pivot_table(index="bin", columns="seed",
                                 values="success_rate", aggfunc="mean").sort_index()
            piv = piv.rolling(window=3, min_periods=1).mean()
            for seed in piv.columns:
                ax.plot(piv.index / 1e6, piv[seed], color=color,
                        alpha=0.12, lw=0.9)
            mean = piv.mean(axis=1)
            ax.plot(mean.index / 1e6, mean.values, color=color, lw=2.4,
                    label=label, zorder=5)
        ax.set_title(f"k={k}: {env}", loc="left", pad=8)
        ax.set_xlabel("Env steps (M)")
        setup_ax(ax)
    axes[0].set_ylabel("Success rate")
    axes[2].legend(loc="upper left", fontsize=9)
    fig.savefig(save_to)
    plt.close(fig)


# ============== FIGURE 3: representation analysis ==============

def fig_representation_analysis(save_to):
    """Two-panel: actor feature_rank (left) and actor dormant_ratio
    (right) at end-of-task for each of the four baseline cells across
    the 10 tasks. Decomposed has no rl_metrics so it is absent; the
    panel is a 'why baselines fail' figure, not a head-to-head.
    """
    rl = load_rl_metrics()
    rl = rl.sort_values("rl_metrics/env_steps").drop_duplicates(
        subset=["run_id"], keep="last")

    cells = [c for c in CELLS if c != "actor=reset-critic=decomposed"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    metric_cols = [
        ("rl_metrics/actor/feature_rank", "Effective rank of actor representation"),
        ("rl_metrics/actor/dormant_ratio", "Dormant-neuron ratio (actor)"),
    ]
    for ax, (mcol, ylab) in zip(axes, metric_cols):
        for cell in cells:
            _, label, color = CELLS[cell]
            sub = rl[rl["cell"] == cell]
            g = sub.groupby("task_idx")[mcol].agg(["mean", "std", "count"]).reindex(range(10))
            x = np.arange(10)
            means = g["mean"].fillna(0).to_numpy()
            stds = g["std"].fillna(0).to_numpy()
            ax.plot(x, means, color=color, lw=2.0, marker="o", ms=4.5,
                    label=label, zorder=5)
            ax.fill_between(x, means - stds, means + stds, color=color,
                            alpha=0.12, zorder=2)
        ymax = max(1.05 * np.nanmax(g["mean"] + g["std"]) if mcol.endswith("rank") else 0.5, 1.05)
        setup_ax(ax, ymax=ymax)
        task_xticks(ax)
        ax.set_ylabel(ylab)
        for k in (5, 8, 9):
            ax.axvspan(k - 0.45, k + 0.45, color="#fff8f0", zorder=0)

    axes[1].legend(loc="upper right", fontsize=9)
    fig.savefig(save_to)
    plt.close(fig)


# ============== FIGURE 4: appendix — CKA diagnostic compact ==============

def fig_cka_appendix(save_to):
    """Compact appendix figure: two-panel scatter showing that the
    critic-side CKA mixture coefficient anti-predicts success while the
    actor-side does not. (Reuses the H1/H2 data; new clean version.)
    """
    csv = HERE.parent / "docs" / "wandb_analysis" / "csv" / "h1_h2_alpha_vs_success.csv"
    if not csv.exists():
        print(f"  [skip cka_appendix] missing {csv}")
        return
    df = pd.read_csv(csv)
    # The diagnostic CSV uses 'task_id' and 'best_success'.
    df = df.rename(columns={"task_id": "task_idx"})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    titles = [
        ("actor_alpha_scale_end",  "Actor-side mixture",  axes[0]),
        ("critic_alpha_scale_end", "Critic-side mixture", axes[1]),
    ]
    cmap = plt.get_cmap("viridis")
    tasks_unique = sorted(df["task_idx"].unique())
    color_for = {t: cmap(i / max(1, len(tasks_unique) - 1))
                 for i, t in enumerate(tasks_unique)}
    from scipy.stats import pearsonr
    for col, title, ax in titles:
        x = df[col].to_numpy()
        y = df["best_success"].to_numpy()
        r, p = pearsonr(x, y)
        for t in tasks_unique:
            m = df["task_idx"] == t
            ax.scatter(df.loc[m, col], df.loc[m, "best_success"],
                       color=color_for[t], s=46, edgecolor="white",
                       lw=0.8, alpha=0.95, zorder=5,
                       label=f"k={t}")
        ax.set_xlabel(r"end-of-task $s_\alpha$")
        ax.set_title(f"{title}: r = {r:+.2f} (p = {p:.3f})", loc="left", pad=6)
        setup_ax(ax)
    axes[0].set_ylabel("End-of-task success")
    axes[1].legend(loc="lower left", ncols=2, fontsize=8, frameon=False)
    fig.savefig(save_to)
    plt.close(fig)


# ---------------- driver ----------------

def main():
    df_per_seed = load_per_seed()
    fig_per_task(df_per_seed, OUT / "fig1_per_task.png")
    fig_trajectories(OUT / "fig2_trajectories.png")
    fig_representation_analysis(OUT / "fig3_representation.png")
    fig_cka_appendix(OUT / "figA1_cka_diagnostic.png")
    print("wrote:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
