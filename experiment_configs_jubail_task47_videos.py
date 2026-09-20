#!/usr/bin/env python3
"""Offline GIFs from paper Task-4 / Task-7 final checkpoints.

Cells:
  0  stick_plain  — seed-6 plain DCC, sawyer_stick_pull
  1  stick_bc     — seed-6 DCC + Success-BC
  2  shelf_plain  — seed-6 plain DCC, sawyer_shelf_place
  3  shelf_bc     — seed-6 DCC + Success-BC
"""
from __future__ import annotations

import argparse
import shlex


SEED = 6
CKPT_ROOT = (
    '/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail_checkpoints/10seed')
PLAIN_DIR = (
    CKPT_ROOT
    + '/actor_reset_critic_decomposed_tid_False_heads_True'
    + '_success_native_info_dyn0.000_pt256x4')
BC_DIR = (
    PLAIN_DIR + '_bridge_d2186abe4046_ret_847a4481eb68')

VARIANTS = (
    {
        'variant': 'stick_plain',
        'single_task': 'sawyer_stick_pull',
        'task_id': 4,
        'label': 'dcc_plain',
        'checkpoint_file': f'{PLAIN_DIR}/seed_{SEED}/task_4.pkl',
        'success_bc_weight': 0.0,
    },
    {
        'variant': 'stick_bc',
        'single_task': 'sawyer_stick_pull',
        'task_id': 4,
        'label': 'dcc_bc',
        'checkpoint_file': f'{BC_DIR}/seed_{SEED}/task_4.pkl',
        'success_bc_weight': 0.1,
    },
    {
        'variant': 'shelf_plain',
        'single_task': 'sawyer_shelf_place',
        'task_id': 7,
        'label': 'dcc_plain',
        'checkpoint_file': f'{PLAIN_DIR}/seed_{SEED}/task_7.pkl',
        'success_bc_weight': 0.0,
    },
    {
        'variant': 'shelf_bc',
        'single_task': 'sawyer_shelf_place',
        'task_id': 7,
        'label': 'dcc_bc',
        'checkpoint_file': f'{BC_DIR}/seed_{SEED}/task_7.pkl',
        'success_bc_weight': 0.1,
    },
)


def build_configs():
  configs = []
  for variant in VARIANTS:
    config = dict(variant)
    config['seed'] = SEED
    config['sawyer_success_mode'] = 'native_info'
    config['network_width'] = 1024
    config['critic_depth'] = 4
    config['actor_depth'] = 4
    config['rollouts'] = 10
    config['hunt_episodes'] = 20
    configs.append(config)
  return configs


def _emit(config):
  for key, value in config.items():
    if isinstance(value, bool):
      value = 'true' if value else 'false'
    elif isinstance(value, str):
      value = shlex.quote(value)
    print(f'{key.upper()}={value}')


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
    for i, c in enumerate(configs):
      print(i, c['variant'], c['single_task'], c['label'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
