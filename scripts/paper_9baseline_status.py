#!/usr/bin/env python3
"""Completion helper for the 9-baseline × 10-seed paper rerun."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_paper_9baseline_10seed as configs

DEFAULT_CHECKPOINT_DIR = (
    '/scratch/yd2247/sgcrl/logs/paper_9baseline_checkpoints/10seed')
NUM_TASKS = 10


def _checkpoint_dir(explicit=None):
  return explicit or os.environ.get(
      'CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR)


def latest_completed_task(config, checkpoint_dir):
  latest = -1
  for task_id in range(NUM_TASKS):
    path = configs.checkpoint_path(
        checkpoint_dir, config['actor_mode'], config['critic_mode'],
        config['seed'], task_id)
    if os.path.exists(path):
      latest = task_id
    else:
      break
  return latest


def is_complete(config, checkpoint_dir):
  return latest_completed_task(config, checkpoint_dir) >= (NUM_TASKS - 1)


def configs_for_array_task(array_task_id, tasks_per_gpu, offset=0, limit=0):
  all_configs = configs.build_configs()
  end = len(all_configs) if limit <= 0 else min(
      len(all_configs), offset + limit)
  start = offset + tasks_per_gpu * array_task_id
  return [
      (index, all_configs[index])
      for index in range(start, min(start + tasks_per_gpu, end))
  ]


def incomplete_array_ids(tasks_per_gpu, checkpoint_dir, offset=0, limit=0):
  all_configs = configs.build_configs()
  n = len(all_configs) if limit <= 0 else min(
      len(all_configs), offset + limit)
  n_array = max(1, math.ceil(n / tasks_per_gpu))
  incomplete = []
  for array_id in range(n_array):
    cells = configs_for_array_task(
        array_id, tasks_per_gpu, offset=offset, limit=limit)
    if cells and not all(
        is_complete(config, checkpoint_dir) for _, config in cells):
      incomplete.append(array_id)
  return incomplete


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint-dir', default=None)
  parser.add_argument('--tasks-per-gpu', type=int, default=4)
  parser.add_argument('--offset', type=int, default=0)
  parser.add_argument('--limit', type=int, default=0)
  parser.add_argument('--array-task-id', type=int, default=0)
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument('--list', action='store_true')
  mode.add_argument('--incomplete-array-ids', action='store_true')
  mode.add_argument('--array-task-complete', action='store_true')
  mode.add_argument('--summary', action='store_true')
  args = parser.parse_args()
  checkpoint_dir = _checkpoint_dir(args.checkpoint_dir)

  if args.list or args.summary:
    complete = 0
    for index, config in enumerate(configs.build_configs()):
      latest = latest_completed_task(config, checkpoint_dir)
      done = latest >= (NUM_TASKS - 1)
      complete += int(done)
      status = 'done' if done else f'task_{latest}'
      print(
          f'{index:3d} {config["wandb_group"]} seed={config["seed"]} '
          f'{status}')
    print(f'complete {complete}/{len(configs.build_configs())} '
          f'under {checkpoint_dir}')
    return

  if args.incomplete_array_ids:
    ids = incomplete_array_ids(
        args.tasks_per_gpu, checkpoint_dir,
        offset=args.offset, limit=args.limit)
    if ids:
      print(','.join(str(i) for i in ids))
    return

  cells = configs_for_array_task(
      args.array_task_id, args.tasks_per_gpu,
      offset=args.offset, limit=args.limit)
  if not cells:
    raise SystemExit(0)
  if all(is_complete(config, checkpoint_dir) for _, config in cells):
    raise SystemExit(0)
  raise SystemExit(1)


if __name__ == '__main__':
  main()
