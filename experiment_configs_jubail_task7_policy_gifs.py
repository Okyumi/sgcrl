#!/usr/bin/env python3
"""Task-7 shelf_place GIFs: plain DCC, episode-wide BC, terminal BC.

Paper plain / episode-wide cells only saved ``task_7.pkl``, so those two
cells are final-policy rollouts plus a first-success hunt. Terminal BC
saved 100k mid-ckpts, so that cell is 10 paced snapshots over 8M plus a
hunt at the first 10% eval (~1.5M, seed 6).

Cells:
  0  shelf_plain     paper DCC seed-6 final
  1  shelf_bc        paper episode-wide Success-BC seed-6 final
  2  shelf_terminal  terminal-episode Success-BC seed-6 mid-ckpts
"""
from __future__ import annotations

import argparse
import shlex


SEED = 6
WANDB_GROUP = 'PAPER-DCC-TASK7-POLICY-GIFS'
CKPT_ROOT = (
    '/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail_checkpoints/10seed')
PLAIN_DIR = (
    CKPT_ROOT
    + '/actor_reset_critic_decomposed_tid_False_heads_True'
    + '_success_native_info_dyn0.000_pt256x4')
BC_DIR = PLAIN_DIR + '_bridge_d2186abe4046_ret_847a4481eb68'
TERMINAL_DIR = (
    '/scratch/yd2247/sgcrl/logs/terminal_bc_task47/checkpoints/'
    'actor_reset_critic_decomposed_tid_False_heads_True'
    '_success_native_info_dyn0.000_pt256x4'
    '_bridge_4c3aef3792be_ret_7d91520b6c16')

VARIANTS = (
    {
        'variant': 'shelf_plain',
        'job_mode': 'checkpoint_file',
        'label': 'dcc_plain',
        'checkpoint_file': f'{PLAIN_DIR}/seed_{SEED}/task_7.pkl',
        'checkpoint_dir': '',
        'success_bc_weight': 0.0,
        'success_bc_label_mode': 'raw_horizon',
        'first_success_step': 0,
        'horizon': 8_000_000,
        'task_id': 7,
    },
    {
        'variant': 'shelf_bc',
        'job_mode': 'checkpoint_file',
        'label': 'dcc_bc',
        'checkpoint_file': f'{BC_DIR}/seed_{SEED}/task_7.pkl',
        'checkpoint_dir': '',
        'success_bc_weight': 0.1,
        'success_bc_label_mode': 'episode_sparse_reward',
        'first_success_step': 0,
        'horizon': 8_000_000,
        'task_id': 7,
    },
    {
        'variant': 'shelf_terminal',
        'job_mode': 'mid_ckpts',
        'label': 'dcc_terminal_bc',
        'checkpoint_file': '',
        'checkpoint_dir': TERMINAL_DIR,
        'success_bc_weight': 0.1,
        'success_bc_label_mode': 'terminal_episode',
        'first_success_step': 1_500_750,
        'horizon': 8_000_000,
        'task_id': 7,
    },
)


def build_configs():
  configs = []
  for variant in VARIANTS:
    config = dict(variant)
    config['seed'] = SEED
    config['single_task'] = 'sawyer_shelf_place'
    config['sawyer_success_mode'] = 'native_info'
    config['network_width'] = 1024
    config['critic_depth'] = 4
    config['actor_depth'] = 4
    config['rollouts'] = 10
    config['hunt_episodes'] = 30
    config['even'] = 10
    configs.append(config)
  return configs


_EXTRA = ('variant', 'job_mode')


def _emit(config):
  for key, value in config.items():
    if key in _EXTRA:
      continue
    if isinstance(value, bool):
      value = 'true' if value else 'false'
    elif isinstance(value, str):
      value = shlex.quote(value)
    print(f'{key.upper()}={value}')
  print(f"VARIANT={shlex.quote(config['variant'])}")
  print(f"JOB_MODE={shlex.quote(config['job_mode'])}")


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
      print(i, c['variant'], c['job_mode'], c['label'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
