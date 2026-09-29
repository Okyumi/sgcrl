#!/usr/bin/env python3
"""Upload collaborator offline W&B run dirs into nyuad_mmvc.

The co-author's sparse SAC+HER logs live in ``logs/zd662_wandb/`` (copied
from ``/scratch/zd662/sgcrl/wandb``). They were originally attached to
``d_konoki/continual_sac``, which this account cannot read. ``wandb sync``
with ``--entity/--project`` retargets them so the paper agent can query
them through the normal W&B API.

Usage
-----

  WANDB_API_KEY=... python scripts/sync_zd662_offline_wandb.py \\
      [--source logs/zd662_wandb] \\
      [--entity nyuad_mmvc] \\
      [--project zd662_sparse_sac_her] \\
      [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_SOURCE = Path("/scratch/yd2247/sgcrl/logs/zd662_wandb")
DEFAULT_ENTITY = "nyuad_mmvc"
DEFAULT_PROJECT = "zd662_sparse_sac_her"
WANDB_BIN = "/scratch/yd2247/miniconda3/envs/contrastive_rl/bin/wandb"


def _run_dirs(source: Path) -> list[Path]:
    return sorted(
        p for p in source.iterdir()
        if p.is_dir() and p.name.startswith("run-") and not p.is_symlink()
    )


def _run_id(run_dir: Path) -> str:
    # run-YYYYMMDD_HHMMSS-<id>
    name = run_dir.name
    return name.rsplit("-", 1)[-1]


def _already_uploaded(entity: str, project: str) -> set[str]:
    import wandb
    api = wandb.Api()
    path = f"{entity}/{project}"
    try:
        runs = api.runs(path, per_page=500)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not list {path}: {exc}", file=sys.stderr)
        return set()
    ids = {r.id for r in runs}
    print(f"[info] {path} already has {len(ids)} runs", flush=True)
    return ids


def _sync_one(run_dir: Path, entity: str, project: str) -> dict:
    cmd = [
        WANDB_BIN, "sync",
        "-e", entity,
        "-p", project,
        "--include-online",
        "--include-offline",
        "--include-synced",
        "--no-mark-synced",
        str(run_dir),
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    url = None
    for line in out.splitlines():
        if "https://wandb.ai/" in line:
            url = line.strip()
            break
    return {
        "dir": run_dir.name,
        "run_id": _run_id(run_dir),
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 1),
        "url": url,
        "error_in_output": "ERROR" in out,
        "tail": "\n".join(out.strip().splitlines()[-8:]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--log",
        type=Path,
        default=Path("results/data/raw/zd662_continual_sac/sync_log.jsonl"),
    )
    args = p.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        key_file = Path.home() / ".wandb_api_key"
        if key_file.exists():
            os.environ["WANDB_API_KEY"] = key_file.read_text().strip()
        else:
            print("ERROR: WANDB_API_KEY is not set", file=sys.stderr)
            return 2

    dirs = _run_dirs(args.source)
    if args.limit is not None:
        dirs = dirs[: args.limit]
    already = _already_uploaded(args.entity, args.project)
    pending = [d for d in dirs if _run_id(d) not in already]
    print(
        f"[info] {len(dirs)} local run dirs, {len(already)} already uploaded, "
        f"{len(pending)} to sync, workers={args.workers}",
        flush=True,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {
            pool.submit(_sync_one, d, args.entity, args.project): d
            for d in pending
        }
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            ok = result["returncode"] == 0
            n_ok += int(ok)
            n_err += int(not ok)
            status = "ok" if ok else "FAIL"
            print(
                f"[{i}/{len(pending)}] {status} {result['dir']} "
                f"{result['seconds']}s url={result['url']}",
                flush=True,
            )
            if result["error_in_output"] or not ok:
                print(result["tail"], flush=True)
            with args.log.open("a") as f:
                f.write(json.dumps(result) + "\n")
    print(f"[done] ok={n_ok} fail={n_err} log={args.log}", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
