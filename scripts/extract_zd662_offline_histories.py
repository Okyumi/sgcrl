#!/usr/bin/env python3
"""Decode collaborator offline ``*.wandb`` files into success-rate histories.

Reads ``logs/zd662_wandb/run-*/run-*.wandb`` with wandb 0.15.12 and writes
evaluator trajectories plus a per-run table under
``results/data/raw/zd662_continual_sac/``. That is the local fallback if
the cloud project ``nyuad_mmvc/zd662_sparse_sac_her`` is still catching
up.

Must be run with the ``contrastive_rl`` env (wandb==0.15.12).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

SOURCE = Path("/scratch/yd2247/sgcrl/logs/zd662_wandb")
OUT = Path("/scratch/yd2247/sgcrl/results/data/raw/zd662_continual_sac")
RUN_NAME_RE = re.compile(r"^task(\d+)_(.+)_s(\d+)$")
CFG_KEYS = (
    "alg_name", "actor_mode", "critic_mode", "seed", "task_id",
    "env_name", "step_penalty_reward", "network_width", "num_tasks",
)


def _cfg_value(text: str, key: str):
    m = re.search(rf"(?m)^{re.escape(key)}:\n  desc:.*\n  value: (.*)$", text)
    if not m:
        return None
    v = m.group(1).strip()
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v in ("null", "None"):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d+\.\d+", v):
        return float(v)
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return v


def _scan_history(wandb_path: Path) -> list[dict]:
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_path))
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for it in rec.history.item:
            try:
                row[it.key] = json.loads(it.value_json)
            except Exception:
                row[it.key] = it.value_json
        rows.append(row)
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        p for p in SOURCE.iterdir()
        if p.is_dir() and p.name.startswith("run-") and not p.is_symlink()
    )
    hist_path = OUT / "histories.csv"
    runs_path = OUT / "runs.csv"
    hist_fields = [
        "run_id", "run_name", "group", "actor_mode", "critic_mode",
        "seed", "task_idx", "env", "network_width", "step_penalty_reward",
        "env_steps", "success_rate",
    ]
    run_fields = [
        "run_id", "dir", "run_name", "group", "alg_name", "actor_mode",
        "critic_mode", "seed", "task_idx", "env", "network_width",
        "step_penalty_reward", "n_eval_rows", "best_success", "end_success",
        "eval_mean_success",
    ]
    n_hist = n_runs = 0
    with hist_path.open("w", newline="") as hf, runs_path.open("w", newline="") as rf:
        hw = csv.DictWriter(hf, fieldnames=hist_fields)
        rw = csv.DictWriter(rf, fieldnames=run_fields)
        hw.writeheader()
        rw.writeheader()
        for i, d in enumerate(run_dirs, 1):
            wandb_files = list(d.glob("*.wandb"))
            cfg_text = (d / "files" / "config.yaml").read_text(errors="ignore") if (d / "files" / "config.yaml").exists() else ""
            summ = {}
            summ_path = d / "files" / "wandb-summary.json"
            if summ_path.exists():
                try:
                    summ = json.loads(summ_path.read_text())
                except Exception:
                    summ = {}
            cfg = {k: _cfg_value(cfg_text, k) for k in CFG_KEYS}
            run_id = d.name.rsplit("-", 1)[-1]
            env = cfg.get("env_name")
            seed = cfg.get("seed")
            task_idx = cfg.get("task_id")
            run_name = None
            if task_idx is not None and env and seed is not None:
                run_name = f"task{task_idx}_{env}_s{seed}"
            eval_rows = []
            if wandb_files:
                try:
                    history = _scan_history(wandb_files[0])
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] decode failed {d.name}: {exc}", file=sys.stderr)
                    history = []
                for row in history:
                    if "evaluator/success_rate" not in row:
                        continue
                    eval_rows.append({
                        "run_id": run_id,
                        "run_name": run_name,
                        "group": "sac_test",
                        "actor_mode": cfg.get("actor_mode"),
                        "critic_mode": cfg.get("critic_mode"),
                        "seed": seed,
                        "task_idx": task_idx,
                        "env": env,
                        "network_width": cfg.get("network_width"),
                        "step_penalty_reward": cfg.get("step_penalty_reward"),
                        "env_steps": row.get("evaluator/env_steps"),
                        "success_rate": row.get("evaluator/success_rate"),
                    })
            for row in eval_rows:
                hw.writerow(row)
            successes = [r["success_rate"] for r in eval_rows if r["success_rate"] is not None]
            rw.writerow({
                "run_id": run_id,
                "dir": d.name,
                "run_name": run_name,
                "group": "sac_test",
                "alg_name": cfg.get("alg_name"),
                "actor_mode": cfg.get("actor_mode"),
                "critic_mode": cfg.get("critic_mode"),
                "seed": seed,
                "task_idx": task_idx,
                "env": env,
                "network_width": cfg.get("network_width"),
                "step_penalty_reward": cfg.get("step_penalty_reward"),
                "n_eval_rows": len(eval_rows),
                "best_success": max(successes) if successes else None,
                "end_success": successes[-1] if successes else None,
                "eval_mean_success": summ.get("eval/mean_success"),
            })
            n_hist += len(eval_rows)
            n_runs += 1
            if i % 25 == 0 or i == len(run_dirs):
                print(f"[{i}/{len(run_dirs)}] runs={n_runs} eval_rows={n_hist}", flush=True)
    print(f"wrote {hist_path} ({n_hist} rows) and {runs_path} ({n_runs} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
