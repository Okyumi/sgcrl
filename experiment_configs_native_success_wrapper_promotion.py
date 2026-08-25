#!/usr/bin/env python3
"""Guarded full-horizon ten-task native-success revalidation matrix."""
from __future__ import annotations

import argparse
import os
import shlex

from experiment_configs_native_success_wrapper import METHODS, SEEDS


def build_configs():
  if os.environ.get('NATIVE_SUCCESS_WRAPPER_PROMOTED', '').lower() != 'true':
    raise ValueError(
        'Set NATIVE_SUCCESS_WRAPPER_PROMOTED=true only after the V5 positive '
        'controls and ten-task smoke matrix pass.')
  configs = []
  for method, actor_mode, critic_mode in METHODS:
    for seed in SEEDS:
      configs.append({
          'actor_mode': actor_mode,
          'critic_mode': critic_mode,
          'seed': seed,
          'num_tasks': 10,
          'steps_per_task': 8_000_000,
          'base_steps': 8_000_000,
          # Evaluation was a material overhead in earlier runs. This cadence
          # is 4x cheaper than 50k and still yields 40 points per task.
          'eval_every': 200_000,
          'eval_episodes': 10,
          'sawyer_success_mode': 'native_info',
          'goal_conditioning_mode': 'full_state',
          'use_task_id': False,
          'actor_auto_reset': False,
          'dyn_aux_weight': 1.0,
          'in_trajectory_negative_repeats': 1,
          'counterfactual_rank_interval_steps': 0,
          'counterfactual_oracle_interval_steps': 0,
          'action_landscape_diagnostic_interval_steps': 0,
          'shortcut_diagnostic_interval': 0,
          'log_rl_metrics': False,
          'log_pool_cosine': False,
          'log_mixture_norm': False,
          'log_probe_data': False,
          'profile_runtime': True,
          'intra_eval_previous': False,
          'post_task_eval_scope': 'current',
          'wandb_group': f'NATIVE-SUCCESS-V2-10TASK-PROMOTE-{method}',
      })
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
  try:
    configs = build_configs()
  except ValueError as error:
    raise SystemExit(f'ERROR: {error}') from error
  if args.total:
    print(len(configs))
    return
  if args.list:
    for index, config in enumerate(configs):
      print(index, config['wandb_group'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
