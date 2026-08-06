#!/usr/bin/env python3
"""Fetch W&B run histories for the three baseline groups.

Pulls `evaluator/success_rate` and `evaluator/env_steps` (the per-step
success-rate trajectory during training of a given task) for every
finished run in the project that matches one of three groups:

  - for_real : 9-cell ablation grid (actor x critic in {reset, persistent, cka})
               -- we keep ALL nine cells here, but the metrics pipeline
                  later filters to the 4-cell {reset, persistent}^2 subset
                  for the GCRL baseline table.
  - real1    : CKA cells (actor=cka, critic in {reset, persistent}).
  - real2    : CKA cells (actor=cka, critic in {reset, persistent, cka}).

Each W&B run corresponds to a single (task k, seed) pair. The run name
follows the pattern  task{k}_{env}_s{seed}  (e.g. task7_sawyer_shelf_place_s99).

We write three artefacts per group under `results/data/raw/<group>/`:

  - histories.parquet : long-format dataframe with one row per (run, step).
                        Columns: run_id, run_name, group, actor_mode,
                        critic_mode, seed, task_idx, env, state,
                        env_steps, success_rate.
  - runs.csv          : one row per W&B run with config + summary fields.
  - fetch_manifest.json : metadata about the fetch (time, project, counts,
                          api key fingerprint).

Usage
-----

  WANDB_API_KEY=... python results/scripts/fetch_wandb_runs.py \\
      --project nyuad_mmvc/continual_gcrl_paper \\
      --groups for_real real1 real2 \\
      --out_dir results/data/raw \\
      [--include_crashed]      # default: only state=finished
      [--limit_per_group N]    # smoke-test

Re-running is idempotent: artefacts are overwritten in place.

Dependencies: wandb, pandas, pyarrow (or fastparquet).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Lazy-import wandb so --help works without it.

RUN_NAME_RE = re.compile(r"^task(\d+)_(.+)_s(\d+)$")

# Evaluation and learner-diagnostic history keys.  W&B writes evaluator
# and learner rows at different cadences, so the fetcher keeps their union.
DIAGNOSTIC_KEYS = [
    "learner/shortcut/categorical_accuracy",
    "learner/shortcut/action_shuffled_categorical_accuracy",
    "learner/shortcut/zero_action_categorical_accuracy",
    "learner/shortcut/action_shuffle_retention",
    "learner/shortcut/zero_action_retention",
    "learner/shortcut/logit_saturation_fraction",
    "learner/shortcut/positive_negative_margin",
    "learner/action/dcc_shuffle_delta_rms",
    "learner/action/dcc_shuffle_delta_abs",
    "learner/action/dcc_candidate_std_policy",
    "learner/action/dcc_candidate_std_uniform",
    "learner/action/dcc_action_grad_norm",
    "learner/action/dcc_progress_spearman",
    "learner/action/q_candidate_std_policy",
    "learner/action/q_candidate_std_uniform",
    "learner/action/q_progress_spearman",
    "learner/action/dcc_q_candidate_spearman",
    "learner/q/twin_disagreement_periodic",
    "learner/acdcc/action_contrast_loss",
    "learner/acdcc/action_contrast_accuracy",
    "learner/acdcc/action_contrast_margin",
    "learner/acdcc/action_score_std",
    "learner/dcc_sac/q_loss",
    "learner/dcc_sac/q_mean",
    "learner/dcc_sac/q_std",
    "learner/dcc_sac/q_min",
    "learner/dcc_sac/q_max",
    "learner/dcc_sac/q_p01",
    "learner/dcc_sac/q_p99",
    "learner/dcc_sac/q_target_mean",
    "learner/dcc_sac/q_target_std",
    "learner/dcc_sac/q_target_min",
    "learner/dcc_sac/q_target_max",
    "learner/dcc_sac/q_target_p01",
    "learner/dcc_sac/q_target_p99",
    "learner/dcc_sac/td_error_abs",
    "learner/dcc_sac/td_error_max",
    "learner/dcc_sac/td_error_p95",
    "learner/dcc_sac/twin_disagreement_abs",
    "learner/dcc_sac/twin_disagreement_normalized",
    "learner/dcc_sac/td_error_ema",
    "learner/dcc_sac/twin_disagreement_ema",
    "learner/dcc_sac/q_stable",
    "learner/dcc_sac/q_gate_ramp",
    "learner/dcc_sac/q_gate",
    "learner/dcc_sac/beta_effective",
    "learner/dcc_sac/q_correction_mean",
    "learner/dcc_sac/action_saturation_fraction",
    "learner/dcc_sac/q_grad_norm",
    "learner/dcc_sac/dcc_grad_norm",
    "learner/dcc_sac/actor_grad_norm",
    "learner/dcc_sac/alpha_grad_abs",
    "learner/alpha",
    "learner/log_alpha",
]
HISTORY_KEYS = [
    "evaluator/env_steps",
    "evaluator/success_rate",
    "learner/env_steps",
    *DIAGNOSTIC_KEYS,
]

# Config and summary fields we cache alongside each run (small set; the
# full config is reachable via the W&B API if needed later).
RUN_FIELDS = [
    "actor_mode",
    "critic_mode",
    "seed",
    "env_name",
    "use_task_id",
    "adapt_heads_only",
    "encoder_from_base",
    "k_max",
    "k_sample_k",
    "intra_eval_previous_tasks",
    "dyn_aux_weight",
    "phi_task_width",
    "phi_task_depth",
    "dcc_sac_beta_max",
    "dcc_sac_q_warmup_updates",
    "dcc_sac_q_ramp_updates",
    "action_contrast_weight",
    "shortcut_diagnostic_interval",
    "post_task_eval_scope",
]


def _parse_run_name(name: str) -> tuple[int | None, str | None, int | None]:
    """task{k}_{env}_s{seed} -> (k, env, seed). Returns (None, None, None)
    if the name does not match the expected pattern."""
    m = RUN_NAME_RE.match(name)
    if not m:
        return None, None, None
    return int(m.group(1)), m.group(2), int(m.group(3))


def _select_finished_states(state: str, include_crashed: bool) -> bool:
    if state == "finished":
        return True
    if include_crashed and state in ("crashed", "failed"):
        return True
    return False


def _fetch_history_df(run, keys: list[str]) -> pd.DataFrame:
    """Pull a history dataframe for the requested keys.

    Uses ``run.history`` which is server-paginated and faster than
    ``scan_history`` for moderate trajectories.
    """
    try:
        df = run.history(keys=keys, pandas=True, samples=100000)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] history fetch failed for {run.name}: {e}", file=sys.stderr)
        return pd.DataFrame(columns=keys)
    if df is None or df.empty:
        return pd.DataFrame(columns=keys)
    # Keep the union of evaluator and learner rows.  Downstream success
    # metrics filter success_rate explicitly; diagnostic metrics use their
    # own non-null rows.
    return df


def fetch_group(
    api,
    project: str,
    group: str,
    include_crashed: bool,
    limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (histories_df, runs_df, manifest) for one group."""
    runs = list(api.runs(project, filters={"group": group}, per_page=500))
    if limit is not None:
        runs = runs[:limit]
    print(f"[{group}] discovered {len(runs)} runs", flush=True)

    history_rows: list[pd.DataFrame] = []
    run_rows: list[dict] = []
    n_kept = 0
    n_skipped_state = 0
    n_skipped_name = 0
    n_no_history = 0

    for i, r in enumerate(runs, 1):
        if not _select_finished_states(r.state, include_crashed):
            n_skipped_state += 1
            continue
        k, env, seed = _parse_run_name(r.name)
        if k is None:
            n_skipped_name += 1
            continue
        cfg = {f: r.config.get(f) for f in RUN_FIELDS}
        # Prefer the run-name seed if config seed is missing.
        if cfg["seed"] is None:
            cfg["seed"] = seed
        run_row = {
            "run_id": r.id,
            "run_name": r.name,
            "group": group,
            "state": r.state,
            "task_idx": k,
            "env": env,
            **cfg,
            "created_at": str(r.created_at),
            "evaluator_success_rate_summary": r.summary.get("evaluator/success_rate"),
            "evaluator_mean_return_summary": r.summary.get("evaluator/mean_return"),
        }
        run_rows.append(run_row)

        hist = _fetch_history_df(r, HISTORY_KEYS)
        if hist.empty:
            n_no_history += 1
            continue
        for key in HISTORY_KEYS:
            if key not in hist.columns:
                hist[key] = float("nan")
        hist = hist.rename(columns={
            "evaluator/env_steps": "evaluator_env_steps",
            "evaluator/success_rate": "success_rate",
        })
        hist["env_steps"] = (
            pd.to_numeric(hist["learner/env_steps"], errors="coerce")
            .combine_first(pd.to_numeric(
                hist["evaluator_env_steps"], errors="coerce")))
        hist["run_id"] = r.id
        hist["run_name"] = r.name
        hist["group"] = group
        hist["actor_mode"] = cfg["actor_mode"]
        hist["critic_mode"] = cfg["critic_mode"]
        hist["seed"] = cfg["seed"]
        hist["task_idx"] = k
        hist["env"] = env
        hist["state"] = r.state
        output_columns = [
            "run_id", "run_name", "group", "actor_mode", "critic_mode",
            "seed", "task_idx", "env", "state", "env_steps", "success_rate",
            *DIAGNOSTIC_KEYS,
        ]
        history_rows.append(hist[output_columns])
        n_kept += 1
        if i % 25 == 0:
            print(f"  [{group}] processed {i}/{len(runs)} (kept={n_kept})", flush=True)

    runs_df = pd.DataFrame(run_rows)
    if history_rows:
        histories_df = pd.concat(history_rows, ignore_index=True)
    else:
        histories_df = pd.DataFrame(columns=[
            "run_id", "run_name", "group", "actor_mode", "critic_mode",
            "seed", "task_idx", "env", "state", "env_steps", "success_rate",
            *DIAGNOSTIC_KEYS,
        ])

    manifest = {
        "group": group,
        "project": project,
        "fetched_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_runs_discovered": len(runs),
        "n_runs_kept": n_kept,
        "n_skipped_state": n_skipped_state,
        "n_skipped_name_pattern": n_skipped_name,
        "n_no_history": n_no_history,
        "include_crashed": include_crashed,
        "history_keys": HISTORY_KEYS,
        "run_fields": RUN_FIELDS,
    }
    return histories_df, runs_df, manifest


def write_group_outputs(
    out_dir: Path,
    group: str,
    histories_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    manifest: dict,
) -> None:
    g_dir = out_dir / group
    g_dir.mkdir(parents=True, exist_ok=True)
    # Prefer parquet; fall back to csv if pyarrow / fastparquet are missing.
    parquet_path = g_dir / "histories.parquet"
    try:
        histories_df.to_parquet(parquet_path, index=False)
        print(f"  [{group}] wrote {parquet_path} ({len(histories_df):,} rows)")
    except Exception as e:  # noqa: BLE001
        csv_path = g_dir / "histories.csv.gz"
        histories_df.to_csv(csv_path, index=False, compression="gzip")
        print(f"  [{group}] parquet unavailable ({e}); wrote {csv_path}")
    runs_csv = g_dir / "runs.csv"
    runs_df.to_csv(runs_csv, index=False)
    print(f"  [{group}] wrote {runs_csv} ({len(runs_df)} runs)")
    with open(g_dir / "fetch_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="nyuad_mmvc/continual_gcrl_paper")
    p.add_argument("--groups", nargs="+", default=["for_real", "real1", "real2"])
    p.add_argument("--out_dir", default="results/data/raw")
    p.add_argument("--include_crashed", action="store_true")
    p.add_argument("--limit_per_group", type=int, default=None,
                   help="Optional: cap runs per group (for smoke tests).")
    args = p.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        print("ERROR: WANDB_API_KEY must be set in the environment.", file=sys.stderr)
        sys.exit(2)
    import wandb  # local import so --help works without wandb installed
    api = wandb.Api()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_manifest = {
        "project": args.project,
        "groups": args.groups,
        "fetched_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "include_crashed": args.include_crashed,
        "per_group": {},
    }
    for grp in args.groups:
        hist, runs, mani = fetch_group(api, args.project, grp,
                                       args.include_crashed,
                                       args.limit_per_group)
        write_group_outputs(out_dir, grp, hist, runs, mani)
        overall_manifest["per_group"][grp] = {
            "n_runs_discovered": mani["n_runs_discovered"],
            "n_runs_kept": mani["n_runs_kept"],
        }

    with open(out_dir / "fetch_manifest.json", "w") as f:
        json.dump(overall_manifest, f, indent=2)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
