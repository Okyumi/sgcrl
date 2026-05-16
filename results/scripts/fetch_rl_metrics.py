#!/usr/bin/env python3
"""Fetch the `rl_metrics/*` history time series from W&B for the baseline
cells the paper discusses. The C2 (decomposed) group does not log
`rl_metrics`, so this script only pulls baseline groups.

Outputs to `results/data/raw/<group>/rl_metrics.parquet` with one row
per (run, history step) and the columns we care about for the paper's
plasticity / feature-rank story.

Usage:
  WANDB_API_KEY=... python results/scripts/fetch_rl_metrics.py \
      --project nyuad_mmvc/continual_gcrl_paper \
      --groups for_real real1 real2

Idempotent: re-running overwrites the parquet.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

RUN_NAME_RE = re.compile(r"^task(\d+)_(.+)_s(\d+)$")

# Keys are fetched in groups because W&B's history(keys=...) requires
# every key to be present on the same logged step. rl_metrics are logged
# every ~250k env steps; evaluator/* are logged every ~50k.
KEY_GROUPS = {
    "rl_metrics": [
        "rl_metrics/env_steps",
        "rl_metrics/actor/feature_rank",
        "rl_metrics/actor/dormant_ratio",
        "rl_metrics/actor/weight_norm",
        "rl_metrics/actor/final_layer_norm",
        "rl_metrics/actor/entropy",
        "rl_metrics/actor/nrc1",
        "rl_metrics/actor/nrc2",
        "rl_metrics/critic_sa/feature_rank",
        "rl_metrics/critic_sa/dormant_ratio",
        "rl_metrics/critic_sa/entropy",
        "rl_metrics/critic_sa/nrc1",
        "rl_metrics/critic_sa/nrc2",
        "rl_metrics/critic_g/feature_rank",
        "rl_metrics/critic_g/dormant_ratio",
        "rl_metrics/critic_g/entropy",
        "rl_metrics/critic/weight_norm",
    ],
}


def fetch_one(run, keys):
    try:
        df = run.history(keys=keys, pandas=True, samples=200000)
    except Exception as e:
        print(f"  [warn] history fetch failed for {run.name}: {e}", file=sys.stderr)
        return pd.DataFrame(columns=keys)
    if df is None or df.empty:
        return pd.DataFrame(columns=keys)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="nyuad_mmvc/continual_gcrl_paper")
    p.add_argument("--groups", nargs="+", default=["for_real", "real1", "real2"])
    p.add_argument("--out_dir", default="results/data/raw")
    p.add_argument("--limit_per_group", type=int, default=None)
    p.add_argument("--cells", nargs="+", default=None,
                   help="Optional filter: only runs matching one of these "
                        "(actor_mode, critic_mode) tuples, formatted as "
                        "actor:critic e.g. reset:reset persistent:persistent")
    args = p.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        print("ERROR: set WANDB_API_KEY", file=sys.stderr); sys.exit(2)
    import wandb
    api = wandb.Api()

    cell_filter = None
    if args.cells:
        cell_filter = {tuple(c.split(":")) for c in args.cells}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for grp in args.groups:
        runs = list(api.runs(args.project, filters={"group": grp}, per_page=500))
        if args.limit_per_group is not None:
            runs = runs[: args.limit_per_group]
        print(f"[{grp}] {len(runs)} runs total")

        frames = []
        kept, skipped = 0, 0
        for i, r in enumerate(runs, 1):
            if r.state != "finished":
                skipped += 1; continue
            m = RUN_NAME_RE.match(r.name)
            if not m:
                skipped += 1; continue
            k = int(m.group(1)); env = m.group(2); seed = int(m.group(3))
            am = r.config.get("actor_mode"); cm = r.config.get("critic_mode")
            if cell_filter and (am, cm) not in cell_filter:
                skipped += 1; continue
            h = fetch_one(r, KEY_GROUPS["rl_metrics"])
            if h.empty:
                continue
            h["run_id"] = r.id
            h["run_name"] = r.name
            h["group"] = grp
            h["actor_mode"] = am
            h["critic_mode"] = cm
            h["seed"] = r.config.get("seed", seed)
            h["task_idx"] = k
            h["env"] = env
            frames.append(h)
            kept += 1
            if i % 25 == 0:
                print(f"  [{grp}] processed {i}/{len(runs)} (kept={kept})", flush=True)
        if not frames:
            print(f"[{grp}] nothing kept; skip"); continue
        df = pd.concat(frames, ignore_index=True)
        path = out / grp / "rl_metrics.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"[{grp}] wrote {path}: {df.shape[0]:,} rows, {df['run_id'].nunique()} runs")


if __name__ == "__main__":
    main()
