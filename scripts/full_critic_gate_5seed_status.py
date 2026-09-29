#!/usr/bin/env python3
"""Completion helper for the 5-seed from-scratch gated DCC curriculum."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_full_critic_gate_5seed as configs

DEFAULT_CHECKPOINT_DIR = (
    '/scratch/yd2247/sgcrl/logs/full_critic_gate_5seed/checkpoints')
NUM_TASKS = configs.NUM_TASKS


def _checkpoint_dir(explicit=None):
  return explicit or os.environ.get(
      'CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR)


def latest_completed_task(config, checkpoint_dir):
  seed_dir = configs.checkpoint_seed_dir(checkpoint_dir, config)
  if seed_dir is None:
    return -1
  latest = -1
  for task_id in range(NUM_TASKS):
    if (seed_dir / f'task_{task_id}.pkl').exists():
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
  parser.add_argument('--tasks-per-gpu', type=int, default=1)
  parser.add_argument('--offset', type=int, default=0)
  parser.add_argument('--limit', type=int, default=0)
  parser.add_argument('--array-task-id', type=int, default=0)
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument('--list', action='store_true')
  mode.add_argument('--incomplete-array-ids', action='store_true')
  mode.add_argument('--array-task-complete', action='store_true')
  args = parser.parse_args()
  checkpoint_dir = _checkpoint_dir(args.checkpoint_dir)
  if args.list:
    for index, config in enumerate(configs.build_configs()):
      latest = latest_completed_task(config, checkpoint_dir)
      print(index, f"seed={config['seed']}", f"done={latest}",
            'COMPLETE' if latest >= NUM_TASKS - 1 else 'INCOMPLETE')
    return
  if args.incomplete_array_ids:
    ids = incomplete_array_ids(
        args.tasks_per_gpu, checkpoint_dir, args.offset, args.limit)
    print(','.join(str(i) for i in ids))
    return
  cells = configs_for_array_task(
      args.array_task_id, args.tasks_per_gpu, args.offset, args.limit)
  if cells and all(is_complete(config, checkpoint_dir) for _, config in cells):
    return
  sys.exit(1)


if __name__ == '__main__':
  main()
