#!/usr/bin/env python3
"""Render the user-facing markdown tables from the processed CSVs.

Reads ``results/data/processed/*.csv`` and writes:

  - results/tables/per_task_table.md
      One row per task; columns are cells. For each (cell, task) the
      cell value is ``mean(best_success) +/- std`` across seeds.

  - results/tables/cell_summary_table.md
      One row per cell; columns are avg_best_success, avg_end_success,
      stability_mean, forward_transfer, n_seeds, n_task_seed_pairs.

  - results/tables/crl_metrics_table.md
      Continual-RL headline numbers per cell.

This script does not compute anything new; it only formats CSVs into
human-readable markdown. Re-run after compute_metrics.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Cell ordering for tables (most-baseline first).
CELL_ORDER = [
    "actor=reset-critic=reset",
    "actor=reset-critic=persistent",
    "actor=persistent-critic=reset",
    "actor=persistent-critic=persistent",
    "actor=reset-critic=cka",
    "actor=cka-critic=reset",
    "actor=persistent-critic=cka",
    "actor=cka-critic=persistent",
    "actor=cka-critic=cka",
    "actor=reset-critic=decomposed",
]

# Routing rule from the project spec:
#   - Cells with actor and critic both in {reset, persistent} -> 'for_real'
#   - Cells where actor == 'cka' (the main CKA-actor cells) -> pool 'real1' + 'real2'
#   - Cells where critic == 'cka' but actor != 'cka' -> fall back to 'for_real'
#     (they were never re-run under real1/real2 and only exist in for_real)
#
# After routing we dedup by seed across the pooled groups, keeping the
# row with the highest n_evals (i.e. the most-complete run for that seed).
GCRL_CELLS = {
    "actor=reset-critic=reset",
    "actor=reset-critic=persistent",
    "actor=persistent-critic=reset",
    "actor=persistent-critic=persistent",
}
CKA_ACTOR_CELLS = {
    "actor=cka-critic=reset",
    "actor=cka-critic=persistent",
    "actor=cka-critic=cka",
}
DECOMPOSED_CELLS = {
    "actor=reset-critic=decomposed",
}
GCRL_GROUPS = ("for_real",)
CKA_GROUPS = ("real1", "real2")
DECOMPOSED_GROUPS = ("c2_decomposed",)


def _routed_groups(cell: str) -> tuple[str, ...]:
    if cell in GCRL_CELLS:
        return GCRL_GROUPS
    if cell in CKA_ACTOR_CELLS:
        return CKA_GROUPS
    if cell in DECOMPOSED_CELLS:
        return DECOMPOSED_GROUPS
    # actor != cka but critic == cka: fall back to for_real.
    return GCRL_GROUPS

CANONICAL_TASKS = [
    (0, "sawyer_hammer"),
    (1, "sawyer_push_wall"),
    (2, "sawyer_faucet_close"),
    (3, "sawyer_push_back"),
    (4, "sawyer_stick_pull"),
    (5, "sawyer_handle_press_side"),
    (6, "sawyer_push"),
    (7, "sawyer_shelf_place"),
    (8, "sawyer_window_close"),
    (9, "sawyer_peg_unplug_side"),
]


def _fmt(mean: float, std: float, n: int) -> str:
    if not np.isfinite(mean):
        return "--"
    if n <= 0:
        return "--"
    if not np.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _pool_per_seed_rows(per_seed: pd.DataFrame, cell: str) -> pd.DataFrame:
    """Return per-seed rows for a cell, restricted to its routed groups and
    deduplicated across groups by (seed, task_idx) -- keeping the row with
    the highest n_evals."""
    groups = _routed_groups(cell)
    sub = per_seed[(per_seed["cell"] == cell) & (per_seed["group"].isin(groups))].copy()
    if sub.empty:
        return sub
    # If n_evals isn't present (shouldn't happen) default to 0 so sort is stable.
    if "n_evals" not in sub.columns:
        sub["n_evals"] = 0
    sub = sub.sort_values("n_evals", ascending=False)
    sub = sub.drop_duplicates(subset=["seed", "task_idx"], keep="first").reset_index(drop=True)
    return sub


def _agg_pool(per_seed: pd.DataFrame, cell: str, task_idx: int, metric: str) -> tuple[float, float, int]:
    sub = _pool_per_seed_rows(per_seed, cell)
    sub = sub[sub["task_idx"] == task_idx]
    vals = sub[metric].dropna()
    n = int(vals.shape[0])
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    return mean, std, n


def render_per_task_table(per_seed: pd.DataFrame) -> str:
    """One row per task, one column per cell.

    Aggregation is done HERE (re-pooled from per_seed rows) under the
    project's group-routing rule:

      * {reset, persistent}^2 cells pool seeds from `for_real` only.
      * any-CKA cells pool seeds from `real1 + real2`.

    Value: ``best_success`` mean ± std across the pooled seeds.
    """
    all_cells = sorted(per_seed["cell"].unique())
    cells_present = [c for c in CELL_ORDER if c in all_cells]
    cells_extra = sorted(c for c in all_cells if c not in CELL_ORDER)
    cells = cells_present + cells_extra

    header_cells = " | ".join(cells)
    lines = [
        "# Per-task best mean success (± std across seeds)",
        "",
        "Source: ``results/data/processed/per_seed_per_task.csv``, re-pooled here under the project's group-routing rule.",
        "",
        "Group routing:",
        "* GCRL baseline cells (actor and critic both in {reset, persistent}) -> ``for_real`` only.",
        "* Any-CKA cells -> pooled across ``real1`` and ``real2``.",
        "",
        "Metric: ``best_success`` = max of ``evaluator/success_rate`` over the per-step evaluation trajectory during training of that task. The cell value is the mean ± sample standard deviation across seeds (n shown in the bottom row).",
        "",
        f"| task_idx | env | {header_cells} |",
        "|" + "|".join(["---"] * (2 + len(cells))) + "|",
    ]
    for k, env in CANONICAL_TASKS:
        row = [str(k), env]
        for cell in cells:
            m, s, n = _agg_pool(per_seed, cell, k, "best_success")
            row.append(_fmt(m, s, n))
        lines.append("| " + " | ".join(row) + " |")

    # Per-cell average across the 10 tasks at the bottom + n_seeds row.
    avg_row = ["**avg**", "--"]
    n_row = ["_seeds (median)_", "--"]
    for cell in cells:
        means = []
        ns = []
        for k, _ in CANONICAL_TASKS:
            m, _s, n = _agg_pool(per_seed, cell, k, "best_success")
            if n > 0:
                means.append(m)
                ns.append(n)
        if not means:
            avg_row.append("--")
            n_row.append("--")
        else:
            avg_row.append(f"**{sum(means)/len(means):.3f}**")
            n_row.append(f"{int(np.median(ns))}")
    lines.append("| " + " | ".join(avg_row) + " |")
    lines.append("| " + " | ".join(n_row) + " |")

    return "\n".join(lines) + "\n"


def render_cell_summary_table(per_seed: pd.DataFrame) -> str:
    """Per-cell averages, with the same group-routing rule."""
    all_cells = sorted(per_seed["cell"].unique())
    cells = [c for c in CELL_ORDER if c in all_cells] + sorted(c for c in all_cells if c not in CELL_ORDER)

    out = ["# Per-cell summary (averaged across 10 tasks)",
           "",
           "Source: ``results/data/processed/per_seed_per_task.csv``, re-pooled under the project's group-routing rule.",
           "",
           "Group routing: {reset, persistent}^2 cells use ``for_real``; any-CKA cells pool ``real1`` and ``real2``.",
           "",
           "Each column averages the per-task mean of the named metric across the 10 tasks. n_seeds is the median seed count over the 10 tasks; n_pairs is the total (task, seed) pair count for the cell.",
           "",
           "| cell | avg_best | avg_end | avg_mean_during | avg_auc | stability | groups | n_seeds | n_pairs |",
           "|---|---|---|---|---|---|---|---|---|"]

    for cell in cells:
        sub = _pool_per_seed_rows(per_seed, cell)
        if sub.empty:
            continue
        # Per-task means, then average over tasks (one number per cell per metric).
        per_task_means = sub.groupby("task_idx")[[
            "best_success", "end_success",
            "mean_success_during_training", "auc_success", "stability",
        ]].mean()
        per_task_seed_count = sub.groupby("task_idx")["seed"].nunique()
        avg_best = per_task_means["best_success"].mean()
        avg_end = per_task_means["end_success"].mean()
        avg_mean = per_task_means["mean_success_during_training"].mean()
        avg_auc = per_task_means["auc_success"].mean()
        stab = per_task_means["stability"].mean()
        n_seeds_median = int(per_task_seed_count.median())
        n_pairs = int(sub.shape[0])
        groups_str = "+".join(_routed_groups(cell))
        out.append(
            f"| {cell} | {avg_best:.3f} | {avg_end:.3f} | {avg_mean:.3f} | "
            f"{avg_auc:.3f} | {stab:.3f} | {groups_str} | {n_seeds_median} | {n_pairs} |"
        )
    return "\n".join(out) + "\n"


def render_crl_metrics_table(per_seed: pd.DataFrame, reference_cell: str) -> str:
    """Continual-RL headline numbers per cell, with group routing.

    Forward transfer is computed per (task >= 1, seed) as
    ``best_success(cell) - best_success(reference_cell)`` on the SAME seed.
    Because the reference cell lives in ``for_real`` and the CKA cells may
    live in ``real1/real2``, we match seeds across the routed groups.
    """
    all_cells = sorted(per_seed["cell"].unique())
    cells = [c for c in CELL_ORDER if c in all_cells] + sorted(c for c in all_cells if c not in CELL_ORDER)

    # Reference: pooled and deduped by the same routing rule, then
    # narrowed to the columns we need for the FT computation.
    ref_full = _pool_per_seed_rows(per_seed, reference_cell)
    ref_groups = _routed_groups(reference_cell)
    ref = ref_full[["task_idx", "seed", "best_success"]].rename(
        columns={"best_success": "ref_best_success"}
    )

    out = ["# Continual-RL metrics per cell",
           "",
           "Source: ``results/data/processed/per_seed_per_task.csv``, re-pooled under the project's group-routing rule.",
           "",
           "Group routing: {reset, persistent}^2 cells use ``for_real``; any-CKA cells pool ``real1`` and ``real2``.",
           "",
           f"Forward-transfer reference cell: ``{reference_cell}`` (per-task ``best_success`` of each cell minus reference's ``best_success`` on the same task & seed, averaged over k=1..9). The reference is sourced from its own routed group(s): {ref_groups}.",
           "",
           "BWT / forgetting is NOT computable from these runs because ``intra_eval_previous_tasks=False`` on every run; ``stability = end / best`` within a task is the closest in-data proxy.",
           "",
           "| cell | avg_best | forward_transfer (vs ref) | stability | groups | n_seeds | n_pairs |",
           "|---|---|---|---|---|---|---|"]
    for cell in cells:
        sub = _pool_per_seed_rows(per_seed, cell)
        if sub.empty:
            continue
        per_task_means = sub.groupby("task_idx")["best_success"].mean()
        avg_best = per_task_means.mean()
        stab = sub.groupby("task_idx")["stability"].mean().mean()
        merged = sub.merge(ref, on=["task_idx", "seed"], how="inner")
        merged = merged[merged["task_idx"] >= 1]
        if merged.empty:
            ft_str = "--"
        else:
            delta = merged["best_success"] - merged["ref_best_success"]
            n = int(delta.shape[0])
            ft_mean = float(delta.mean())
            ft_sem = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            ft_str = f"{ft_mean:+.3f} ± {ft_sem:.3f} (n={n})"
        groups_str = "+".join(_routed_groups(cell))
        n_seeds = int(sub["seed"].nunique())
        n_pairs = int(sub.shape[0])
        out.append(
            f"| {cell} | {avg_best:.3f} | {ft_str} | {stab:.3f} | "
            f"{groups_str} | {n_seeds} | {n_pairs} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", default="results/data/processed")
    p.add_argument("--out_dir", default="results/tables")
    p.add_argument("--reference_cell", default="actor=reset-critic=reset")
    args = p.parse_args()

    proc = Path(args.processed_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_seed = pd.read_csv(proc / "per_seed_per_task.csv")

    (out / "per_task_table.md").write_text(render_per_task_table(per_seed))
    (out / "cell_summary_table.md").write_text(render_cell_summary_table(per_seed))
    (out / "crl_metrics_table.md").write_text(render_crl_metrics_table(per_seed, args.reference_cell))
    print(f"Wrote {out}/per_task_table.md, cell_summary_table.md, crl_metrics_table.md")


if __name__ == "__main__":
    main()
