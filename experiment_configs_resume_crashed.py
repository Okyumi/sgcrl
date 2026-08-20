#!/usr/bin/env python3
"""Exact configurations for unresolved continual runs found on 2026-08-20.

This file is deliberately separate from experiment_configs.py so resuming old
runs cannot reorder, replace, or accidentally relaunch the active experiment
matrix.  DRAFT_resume_crashed.sh selects this file through CONFIG_SCRIPT.
"""

import argparse
import shlex
import sys


CELLS = [
    # C2 DCC seeds 100/101 completed tasks 0--2 and crashed during task 3.
    *[
        {
            'actor_mode': 'reset',
            'critic_mode': 'decomposed',
            'seed': seed,
            'dyn_aux_weight': 1.0,
            'combine_mode': 'add',
            'goal_encoder_mode': 'shared',
            'use_task_id': False,
            'actor_auto_reset': False,
            'log_probe_data': True,
            'wandb_group': 'C2:',
        }
        for seed in (100, 101)
    ],
    # AC-DCC seeds 5/6 crashed during task 5; seed 7 crashed during task 0.
    *[
        {
            'actor_mode': 'reset',
            'critic_mode': 'action_dcc',
            'seed': seed,
            'dyn_aux_weight': 1.0,
            'combine_mode': 'add',
            'goal_encoder_mode': 'shared',
            'action_contrast_weight': 1.0,
            'action_contrast_temperature': 1.0,
            'shortcut_diagnostic_interval': 1000,
            'post_task_eval_scope': 'current',
            'wandb_group': 'AC-DCC-continual-10-task',
        }
        for seed in (5, 6, 7)
    ],
    # DCC-SAC seed 7 completed tasks 0--2 and crashed during task 3.
    {
        'actor_mode': 'reset',
        'critic_mode': 'dcc_sac',
        'seed': 7,
        'dyn_aux_weight': 1.0,
        'combine_mode': 'add',
        'goal_encoder_mode': 'shared',
        'dcc_sac_beta_max': 0.1,
        'dcc_sac_q_warmup_updates': 10000,
        'dcc_sac_q_ramp_updates': 25000,
        'shortcut_diagnostic_interval': 1000,
        'her_reward_threshold': 0.05,
        'step_penalty_reward': True,
        'post_task_eval_scope': 'current',
        'wandb_group': 'DCC-SAC-continual-10-task',
    },
]


def _emit(config):
  for key, value in config.items():
    if isinstance(value, bool):
      value = 'true' if value else 'false'
    elif isinstance(value, str):
      value = shlex.quote(value)
    print(f'{key.upper()}={value}')


def main():
  parser = argparse.ArgumentParser()
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument('--setting', type=int)
  mode.add_argument('--total', action='store_true')
  mode.add_argument('--list', action='store_true')
  args = parser.parse_args()

  if args.total:
    print(len(CELLS))
    return
  if args.list:
    print('idx  actor  critic       seed  group')
    for i, cell in enumerate(CELLS):
      print(f'{i:>3}  {cell["actor_mode"]:<5}  '
            f'{cell["critic_mode"]:<11}  {cell["seed"]:>4}  '
            f'{cell["wandb_group"]}')
    return
  if args.setting < 0 or args.setting >= len(CELLS):
    print(f'ERROR: setting {args.setting} out of range [0, {len(CELLS)-1}]',
          file=sys.stderr)
    raise SystemExit(1)
  _emit(CELLS[args.setting])


if __name__ == '__main__':
  main()
