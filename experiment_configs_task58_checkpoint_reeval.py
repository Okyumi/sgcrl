#!/usr/bin/env python3
"""Six evaluation-only cells for historical continual DCC checkpoints."""
from __future__ import annotations

import argparse
import shlex


TASKS = (
    (5, 'sawyer_handle_press_side'),
    (8, 'sawyer_window_close'),
)
SEEDS = (5, 6, 7)
# Historical C2 DCC checkpoints for seeds 5/6/7 were written before
# ``_ckpt_path`` started appending ``_dyn{w}_pt{W}x{D}``. Those six
# Task-5/Task-8 actors live in the unsuffixed directory. The later
# ``..._dyn1.000_pt256x4`` tree has seeds 97-101 (and incomplete seed 5
# with only task_0), not the C2 seed-5/6/7 matrix.
CHECKPOINT_CONFIG = (
    'actor_reset_critic_decomposed_tid_False_heads_True'
)
DISAMBIGUATED_CHECKPOINT_CONFIG = (
    'actor_reset_critic_decomposed_tid_False_heads_True_'
    'dyn1.000_pt256x4'
)


def build_configs():
  configs = []
  for task_id, env_name in TASKS:
    for seed in SEEDS:
      configs.append({
          'task_id': task_id,
          'env_name': env_name,
          'seed': seed,
          'episodes': 100,
          'checkpoint_relative': (
              f'{CHECKPOINT_CONFIG}/seed_{seed}/task_{task_id}.pkl'),
          'output_relative': (
              f'task{task_id}_{env_name}_seed{seed}.json'),
      })
  return configs


def _emit(config):
  for key, value in config.items():
    print(f'{key.upper()}={shlex.quote(str(value))}')


def main():
  parser = argparse.ArgumentParser()
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--setting', type=int)
  group.add_argument('--total', action='store_true')
  group.add_argument('--list', action='store_true')
  args = parser.parse_args()
  configs = build_configs()
  if args.total:
    print(len(configs))
    return
  if args.list:
    for index, config in enumerate(configs):
      print(index, config['task_id'], config['env_name'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
