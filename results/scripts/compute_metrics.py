#!/usr/bin/env python3
"""Compute per-task success summaries and continual-RL metrics from the
raw W&B history dumps produced by ``fetch_wandb_runs.py``.

Inputs (read from ``results/data/raw/<group>/``):
  - histories.parquet (or histories.csv.gz)
  - runs.csv
  - fetch_manifest.json

Outputs (written to ``results/data/processed/``):
  - per_seed_per_task.csv   : one row per (cell, seed, task_idx).
  - per_task_aggregate.csv  : one row per (cell, task_idx) -- mean / std /
                              sem / n across seeds. ``cell`` is the string
                              ``actor={a}-critic={c}``.
  - cell_summary.csv        : one row per cell, with the mean across tasks
                              of each per-task metric.
  - crl_metrics.csv         : continual-RL metrics per cell (one row).
  - documentation.md        : human-readable doc of every metric, how it
                              is computed, and what its caveats are.

Metric definitions (this file is the single source of truth; the
documentation is regenerated from these docstrings at the bottom of
the file).

Per-seed-per-task metrics (computed from the
``evaluator/success_rate`` trajectory of a single (run, task) pair):

  best_success
    The maximum value of the per-step ``evaluator/success_rate``
    trajectory observed during training of this task. Robust to
    in-task instability. This is the headline ``best mean'' metric
    the user requested.

  end_success
    The value of ``evaluator/success_rate`` at the final evaluator
    log row for this task. Sensitive to end-of-training instability;
    useful as a stability indicator next to ``best_success``.

  mean_success_during_training
    The mean of ``evaluator/success_rate`` across all evaluator log
    rows for this task. Equivalent (up to discretisation) to the
    success-rate curve's AUC normalised by the trajectory length.

  auc_success
    Trapezoidal integral of ``evaluator/success_rate`` over
    ``evaluator/env_steps`` for this task, divided by the total
    env-step range. Same units as the success rate. Differs from
    ``mean_success_during_training`` only in that the trapezoidal
    rule weights the contribution of each evaluator point by the
    actual env-step spacing rather than treating all points equally.

  stability
    ``end_success / max(best_success, 1e-9)``. Fraction of the
    best-during-training success that survives at the end of the
    task; 1.0 means no within-task forgetting, lower values mean the
    policy lost capability before training ended.

  n_evals
    Number of evaluator log rows for this run. Useful to flag
    truncated or crashed runs.

Per-task aggregate metrics (across seeds within a cell):

  best_success_{mean,std,sem,n}, end_success_{mean,std,sem,n},
  mean_success_during_training_{mean,std,sem,n}, auc_success_{mean,std,sem,n}
    Mean, sample standard deviation, standard error of the mean,
    and seed count for each per-task metric. Reported in
    ``per_task_aggregate.csv``.

Per-cell summary metrics:

  avg_best_success
    Mean across the 10 tasks of (``best_success`` averaged over
    seeds). This is the headline single-number score per cell:
    "best mean performance over the 10 tasks''.

  avg_end_success
    Same as above but using ``end_success``.

  avg_mean_success
    Same as above but using ``mean_success_during_training``.

  forward_transfer_vs_resetreset
    For each task k, ``best_success`` of this cell minus
    ``best_success`` of the reference cell (actor=reset,
    critic=reset, same seed), then averaged across seeds and tasks
    k=1..9 (task 0 excluded because both cells start from scratch
    there). Positive values indicate the cell transfers usefully
    forward relative to from-scratch training. The reference cell
    is the canonical sparse-GCRL no-transfer baseline.

  stability_mean
    Mean of the per-(task, seed) ``stability`` ratio across all
    (task, seed) pairs in this cell. 1.0 = no within-task
    forgetting; <1.0 = within-task forgetting.

  crashed_or_missing_cells
    Count of (task, seed) cells that did not produce any evaluator
    history. Reported so the headline metric isn't silently averaged
    over partial data.

Continual-RL metrics that this dataset CANNOT directly support, with
the reason and the workaround chosen:

  backward_transfer / forgetting (BWT)
    Standard formulation requires evaluation of the agent on task j
    AFTER it has finished training on every later task k > j. None
    of the runs in the for_real / real1 / real2 groups logged
    cross-task evaluation (``intra_eval_previous_tasks=False`` on
    all 764 runs as of 2026-05-14). We therefore cannot compute
    classical BWT from these runs. ``stability`` (above) is a
    within-task proxy. Cross-task BWT would need a fresh run with
    ``--intra_eval_previous_tasks=True`` or a separate eval pass on
    saved checkpoints.

Usage
-----

  python results/scripts/compute_metrics.py \\
      --raw_dir results/data/raw \\
      --groups for_real real1 real2 \\
      --out_dir results/data/processed \\
      [--reference_cell actor=reset-critic=reset]

The ``--reference_cell`` flag controls which cell is used as the
zero-baseline for forward-transfer computation. Default is the
sparse-GCRL no-transfer cell ``actor=reset-critic=reset``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


# --- I/O helpers ---------------------------------------------------------

def _load_history_for_group(raw_dir: Path, group: str) -> pd.DataFrame:
    g_dir = raw_dir / group
    parquet = g_dir / "histories.parquet"
    csv = g_dir / "histories.csv.gz"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv, compression="gzip")
    raise FileNotFoundError(f"No histories file in {g_dir} (looked for parquet and csv.gz)")


def load_all_histories(raw_dir: Path, groups: list[str]) -> pd.DataFrame:
    parts = [_load_history_for_group(raw_dir, g) for g in groups]
    df = pd.concat(parts, ignore_index=True)
    # Drop rows without a finite success rate; downstream metrics expect
    # numeric values.
    df = df.dropna(subset=["success_rate"]).copy()
    df["success_rate"] = df["success_rate"].astype(float)
    df["env_steps"] = pd.to_numeric(df["env_steps"], errors="coerce")
    return df


# --- Per-seed-per-task metrics ------------------------------------------

def _per_run_metrics(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("env_steps")
    success = g["success_rate"].to_numpy()
    steps = g["env_steps"].to_numpy()
    n = len(success)
    best = float(np.nanmax(success)) if n else np.nan
    end = float(success[-1]) if n else np.nan
    mean = float(np.nanmean(success)) if n else np.nan
    if n >= 2 and np.isfinite(steps).all() and steps[-1] > steps[0]:
        auc = float(np.trapz(success, steps) / (steps[-1] - steps[0]))
    else:
        auc = mean
    stability = end / max(best, 1e-9) if n else np.nan
    return pd.Series({
        "n_evals": n,
        "best_success": best,
        "end_success": end,
        "mean_success_during_training": mean,
        "auc_success": auc,
        "stability": stability,
    })


def compute_per_seed_per_task(histories: pd.DataFrame) -> pd.DataFrame:
    """One row per (group, actor_mode, critic_mode, seed, task_idx).

    Where multiple W&B runs exist for the same (cell, seed, task) (this
    happens when a run was re-launched after a crash), we KEEP THE RUN
    WITH THE MOST EVALUATOR LOG ROWS. That is the most-complete attempt
    for that (cell, seed, task).
    """
    grouped = histories.groupby(
        ["group", "actor_mode", "critic_mode", "seed", "task_idx", "env", "run_id"],
        dropna=False,
    )
    per_run = grouped.apply(_per_run_metrics).reset_index()
    # If multiple runs exist for the same (cell, seed, task), keep the
    # one with the most eval rows.
    per_run = per_run.sort_values("n_evals", ascending=False)
    per_seed = per_run.drop_duplicates(
        subset=["group", "actor_mode", "critic_mode", "seed", "task_idx"],
        keep="first",
    ).reset_index(drop=True)
    per_seed["cell"] = (
        "actor=" + per_seed["actor_mode"].astype(str)
        + "-critic=" + per_seed["critic_mode"].astype(str)
    )
    return per_seed


# --- Per-task aggregates --------------------------------------------------

def _agg(s: pd.Series) -> dict:
    s = s.dropna()
    n = int(s.shape[0])
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "sem": np.nan, "n": 0}
    m = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else 0.0
    return {"mean": m, "std": sd, "sem": sem, "n": n}


def compute_per_task_aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = ["best_success", "end_success",
               "mean_success_during_training", "auc_success", "stability"]
    rows = []
    keys = ["cell", "group", "actor_mode", "critic_mode", "task_idx", "env"]
    for key_vals, sub in per_seed.groupby(keys, dropna=False):
        row = dict(zip(keys, key_vals))
        for met in metrics:
            stats = _agg(sub[met])
            for k, v in stats.items():
                row[f"{met}_{k}"] = v
        row["seeds"] = ",".join(map(str, sorted(sub["seed"].dropna().unique().tolist())))
        rows.append(row)
    df = pd.DataFrame(rows)
    # Stable sort: cell, then task_idx.
    return df.sort_values(["cell", "task_idx"]).reset_index(drop=True)


# --- Continual-RL metrics per cell ---------------------------------------

def _ft_one_cell(per_seed: pd.DataFrame, cell: str, ref_per_seed: pd.DataFrame) -> dict:
    """Forward transfer vs reference cell.

    For each (task_idx>=1, seed) present in BOTH the cell and the reference
    cell, compute (cell.best_success - ref.best_success). Average across
    (task, seed) pairs. Also report the per-task means and pair count.
    """
    cell_df = per_seed[per_seed["cell"] == cell].copy()
    if cell_df.empty:
        return {"forward_transfer_mean": np.nan,
                "forward_transfer_sem": np.nan,
                "forward_transfer_n_pairs": 0}
    merged = cell_df.merge(
        ref_per_seed[["task_idx", "seed", "best_success"]].rename(
            columns={"best_success": "ref_best_success"}),
        on=["task_idx", "seed"], how="inner",
    )
    merged = merged[merged["task_idx"] >= 1]
    if merged.empty:
        return {"forward_transfer_mean": np.nan,
                "forward_transfer_sem": np.nan,
                "forward_transfer_n_pairs": 0}
    delta = merged["best_success"] - merged["ref_best_success"]
    n = int(delta.shape[0])
    sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    return {
        "forward_transfer_mean": float(delta.mean()),
        "forward_transfer_sem": sd / np.sqrt(n) if n > 1 else 0.0,
        "forward_transfer_n_pairs": n,
    }


def compute_cell_summary_and_crl(
    per_seed: pd.DataFrame,
    per_task_agg: pd.DataFrame,
    reference_cell: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cell_summary_df, crl_metrics_df).

    cell_summary aggregates per-task means into single scalars per cell.
    crl_metrics adds forward-transfer, stability, and crash counts.
    """
    ref_per_seed = per_seed[per_seed["cell"] == reference_cell].copy()
    if ref_per_seed.empty:
        print(f"  [warn] reference cell {reference_cell!r} not found; "
              f"forward-transfer values will be NaN.", file=sys.stderr)

    summary_rows = []
    crl_rows = []
    keys = ["cell", "group", "actor_mode", "critic_mode"]
    for key_vals, sub in per_seed.groupby(keys, dropna=False):
        row = dict(zip(keys, key_vals))
        # Average each per-task metric across the 10 tasks.
        for met in ("best_success", "end_success",
                    "mean_success_during_training", "auc_success", "stability"):
            row[f"avg_{met}"] = float(sub[met].mean())
        row["n_task_seed_pairs"] = int(sub.shape[0])
        row["n_seeds"] = int(sub["seed"].nunique())
        row["seeds"] = ",".join(map(str, sorted(sub["seed"].dropna().unique().tolist())))
        summary_rows.append(row)

        cell = row["cell"]
        ft = _ft_one_cell(per_seed, cell, ref_per_seed)
        crl_row = {**row, **ft}
        # crashed_or_missing := expected 10 tasks * n_seeds pairs minus
        # observed; >=0 indicates how many (task, seed) pairs are missing
        # entirely (not just low-quality).
        expected = 10 * row["n_seeds"]
        crl_row["crashed_or_missing_cells"] = max(0, expected - row["n_task_seed_pairs"])
        crl_rows.append(crl_row)

    cell_summary = pd.DataFrame(summary_rows).sort_values(["cell"]).reset_index(drop=True)
    crl_metrics = pd.DataFrame(crl_rows).sort_values(["cell"]).reset_index(drop=True)
    return cell_summary, crl_metrics


# --- Documentation regeneration -----------------------------------------

DOC_HEADER = """# Metric definitions (auto-generated)

This file is regenerated by ``results/scripts/compute_metrics.py``;
do not edit by hand. It documents every metric reported in
``per_seed_per_task.csv``, ``per_task_aggregate.csv``,
``cell_summary.csv``, and ``crl_metrics.csv``.

Headline metric (from W&B):
``evaluator/success_rate``, the deterministic per-step evaluation
success rate logged during training of a single task. Each W&B run
in the ``for_real / real1 / real2`` groups corresponds to one
(task k, seed) pair on the ten-task Continual World V2 Sawyer
manipulation sequence; the run name follows the pattern
``task{k}_{env}_s{seed}``.

"""


def render_documentation(reference_cell: str) -> str:
    return (
        DOC_HEADER
        + (compute_metrics_doc := __doc__.split("Metric definitions")[1])
        .replace("Usage\n-----\n", "Reference cell: ``"
                 + reference_cell + "`` (used to compute forward transfer).\n\nUsage\n-----\n")
    )


# --- Main ----------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", default="results/data/raw")
    p.add_argument("--groups", nargs="+", default=["for_real", "real1", "real2"])
    p.add_argument("--out_dir", default="results/data/processed")
    p.add_argument("--reference_cell", default="actor=reset-critic=reset",
                   help="Cell used as the zero-baseline for forward transfer.")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading histories from {raw_dir} for groups {args.groups}", flush=True)
    histories = load_all_histories(raw_dir, args.groups)
    print(f"  loaded {len(histories):,} history rows from "
          f"{histories[['run_id']].drop_duplicates().shape[0]} runs", flush=True)

    print("Computing per-seed-per-task metrics ...", flush=True)
    per_seed = compute_per_seed_per_task(histories)
    per_seed.to_csv(out_dir / "per_seed_per_task.csv", index=False)
    print(f"  -> {out_dir / 'per_seed_per_task.csv'} "
          f"({len(per_seed)} rows)", flush=True)

    print("Computing per-task aggregates ...", flush=True)
    per_task = compute_per_task_aggregate(per_seed)
    per_task.to_csv(out_dir / "per_task_aggregate.csv", index=False)
    print(f"  -> {out_dir / 'per_task_aggregate.csv'} "
          f"({len(per_task)} rows)", flush=True)

    print("Computing cell summary and CRL metrics ...", flush=True)
    cell_summary, crl_metrics = compute_cell_summary_and_crl(
        per_seed, per_task, args.reference_cell)
    cell_summary.to_csv(out_dir / "cell_summary.csv", index=False)
    crl_metrics.to_csv(out_dir / "crl_metrics.csv", index=False)
    print(f"  -> {out_dir / 'cell_summary.csv'} ({len(cell_summary)} rows)")
    print(f"  -> {out_dir / 'crl_metrics.csv'} ({len(crl_metrics)} rows)")

    with open(out_dir / "documentation.md", "w") as f:
        f.write(render_documentation(args.reference_cell))
    with open(out_dir / "processing_manifest.json", "w") as f:
        json.dump({
            "raw_dir": str(raw_dir),
            "groups": args.groups,
            "reference_cell": args.reference_cell,
            "n_history_rows": int(len(histories)),
            "n_cells": int(per_seed["cell"].nunique()),
        }, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
