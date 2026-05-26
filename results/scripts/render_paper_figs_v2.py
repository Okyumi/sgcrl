#!/usr/bin/env python3
"""Refined paper figures (v2). Reuses the data loaders of the original
render_paper_figs.py and only changes the aesthetic layer."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PROCESSED = HERE / "data" / "processed"
RAW = HERE / "data" / "raw"
OUT_PAPER = Path("/tmp/paper_rlc/figs")
OUT_PAPER.mkdir(parents=True, exist_ok=True)
OUT_ARCH = HERE / "presentation" / "img" / "paper_v2"
OUT_ARCH.mkdir(parents=True, exist_ok=True)

# ---------- refined palette ----------
INK = "#1f2937"
INK_LIGHT = "#475569"
GRID = "#e5e7eb"

COL_RR   = "#94a3b8"   # muted slate
COL_PP   = "#2a9d8f"   # calmer teal
COL_CKA  = "#e0a458"   # warm sand
COL_OURS = "#9d174d"   # deep plum-crimson

HARD_BG = "#fff7ed"
HARD_TASKS = (5, 8, 9)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8,
    "xtick.color": INK_LIGHT,
    "ytick.color": INK_LIGHT,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

CANONICAL_TASKS = [
    (0, "hammer"), (1, "push-wall"), (2, "faucet-close"),
    (3, "push-back"), (4, "stick-pull"), (5, "handle-press-side"),
    (6, "push"), (7, "shelf-place"), (8, "window-close"),
    (9, "peg-unplug-side"),
]

CELLS = {
    "actor=reset-critic=reset":              ("for_real",      "From-scratch (R/R)",  COL_RR),
    "actor=persistent-critic=persistent":    ("for_real",      "Persistent (P/P)",    COL_PP),
    "actor=cka-critic=reset":                ("real1+real2",   "CKA (C/R)",           COL_CKA),
    "actor=reset-critic=decomposed":         ("c2_decomposed", "Decomposed (ours)",   COL_OURS),
}
BUCKETS = {"for_real": ("for_real",),
           "real1+real2": ("real1", "real2"),
           "c2_decomposed": ("c2_decomposed",)}


def setup_ax(ax, ymax=1.05, ymin=0):
    ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def task_xticks(ax):
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{k}\n{n}" for k, n in CANONICAL_TASKS],
                       rotation=20, ha="right", fontsize=8)


def hard_bg(ax):
    for k in HARD_TASKS:
        ax.axvspan(k - 0.5, k + 0.5, color=HARD_BG, zorder=0)


# ---------- loaders, copied from render_paper_figs.py ----------

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
    return sub.groupby("task_idx")[metric].agg(["mean", "std", "count"]).reindex(range(10))


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
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["cell"] = "actor=" + df["actor_mode"] + "-critic=" + df["critic_mode"]
    def keep(row):
        if row["cell"] in {"actor=reset-critic=reset", "actor=persistent-critic=persistent"}:
            return row["group"] == "for_real"
        if row["actor_mode"] == "cka":
            return row["group"] in ("real1", "real2")
        return row["group"] == "for_real"
    return df[df.apply(keep, axis=1)].reset_index(drop=True)


# ============== Figure 1: per-task bar chart ==============

def fig1_bars():
    df = load_per_seed()
    cells = list(CELLS.keys())

    # Wider aspect makes the bars feel less tall and less dense.
    fig, ax = plt.subplots(figsize=(11, 3.4))
    hard_bg(ax)

    width = 0.18
    x = np.arange(10)
    for i, cell in enumerate(cells):
        _, label, color = CELLS[cell]
        s = per_task_stats(df, cell)
        means = s["mean"].fillna(0).to_numpy()
        stds = s["std"].fillna(0).to_numpy()
        offset = (i - (len(cells) - 1) / 2) * width
        ax.bar(x + offset, means, width,
               color=color, edgecolor="white", linewidth=0.6,
               label=label, zorder=3)
        ax.errorbar(x + offset, means, yerr=stds, fmt="none",
                    ecolor=INK_LIGHT, alpha=0.55, capsize=2.0, lw=0.7, zorder=4)

    setup_ax(ax, ymax=1.08)
    task_xticks(ax)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_ylabel("Success rate (best during training)", fontsize=9.5)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42),
              ncol=len(cells), columnspacing=2.2, handlelength=1.4,
              handleheight=1.0, fontsize=9)

    fig.tight_layout()
    out = OUT_PAPER / "fig1_per_task.png"
    fig.savefig(out)
    fig.savefig(OUT_ARCH / "fig1_per_task.png")
    plt.close(fig)
    print(f"wrote {out}")


# ============== Figure 2: hard-task trajectories ==============

def fig2_trajectories():
    cells = list(CELLS.keys())
    hard = [(5, "handle-press-side"), (8, "window-close"), (9, "peg-unplug-side")]

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), sharey=True)
    for ax, (k, env) in zip(axes, hard):
        ax.set_facecolor(HARD_BG)
        for cell in cells:
            _, label, color = CELLS[cell]
            h = load_histories(cell, k)
            if h.empty:
                continue
            h2 = h.copy()
            h2["bin"] = (h2["env_steps"] / 2e5).round() * 2e5
            piv = (h2.pivot_table(index="bin", columns="seed",
                                  values="success_rate", aggfunc="mean")
                   .sort_index()
                   .rolling(window=3, min_periods=1).mean())
            for seed in piv.columns:
                ax.plot(piv.index / 1e6, piv[seed],
                        color=color, alpha=0.13, lw=0.8)
            mean = piv.mean(axis=1)
            ax.plot(mean.index / 1e6, mean.values,
                    color=color, lw=2.0, label=label, zorder=5)
        ax.set_title(f"$k={k}$: {env}", loc="left", pad=6)
        ax.set_xlabel("Env steps (M)", fontsize=9.5)
        setup_ax(ax)
    axes[0].set_ylabel("Success rate", fontsize=9.5)
    axes[2].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    out = OUT_PAPER / "fig2_trajectories.png"
    fig.savefig(out)
    fig.savefig(OUT_ARCH / "fig2_trajectories.png")
    plt.close(fig)
    print(f"wrote {out}")


# ============== Figure 3: actor representation ==============

def fig3_representation():
    rl = load_rl_metrics()
    if rl.empty:
        print("rl_metrics not available; skipping fig3")
        return
    rl = rl.sort_values("rl_metrics/env_steps").drop_duplicates(
        subset=["run_id"], keep="last")

    cells = [c for c in CELLS if c != "actor=reset-critic=decomposed"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0))
    for ax in axes:
        hard_bg(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    metrics = [
        ("rl_metrics/actor/feature_rank", "Effective rank (actor)"),
        ("rl_metrics/actor/weight_norm", "Weight norm (actor)"),
    ]
    n = len(cells)
    dx = 0.08
    for ax, (mcol, ylab) in zip(axes, metrics):
        for i, cell in enumerate(cells):
            _, label, color = CELLS[cell]
            sub = rl[rl["cell"] == cell]
            g = sub.groupby("task_idx")[mcol].agg(["mean", "std", "count"]).reindex(range(10))
            means = g["mean"].to_numpy()
            stds = g["std"].to_numpy()
            counts = g["count"].fillna(0).to_numpy()
            sem = np.where(counts > 1, stds / np.sqrt(np.maximum(counts, 1)), 0)
            offset = (i - (n - 1) / 2) * dx
            x = np.arange(10) + offset
            ax.plot(x, means, color=color, lw=1.6, marker="o", ms=4.5,
                    label=label, zorder=5)
            ax.errorbar(x, means, yerr=sem, fmt="none", ecolor=color,
                        alpha=0.45, capsize=1.8, lw=0.7, zorder=4)
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"{k}\n{n}" for k, n in CANONICAL_TASKS],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylab, fontsize=9.5)
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    out = OUT_PAPER / "fig3_representation.png"
    fig.savefig(out)
    fig.savefig(OUT_ARCH / "fig3_representation.png")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1_bars()
    fig2_trajectories()
    fig3_representation()
