#!/usr/bin/env python3
"""Eight-million-step promotion cells for the validated mechanism goal."""
from __future__ import annotations

import argparse
import os
import shlex


def build_configs():
  if os.environ.get('GOAL_VALIDITY_PROMOTED', '').lower() != 'true':
    raise ValueError(
        'Set GOAL_VALIDITY_PROMOTED=true only after '
        'scripts/evaluate_goal_semantics.py exits successfully.')
  configs = []
  for task, label in (
      ('sawyer_handle_press_side', 'task5'),
      ('sawyer_window_close', 'task8')):
    for seed in (5, 6, 7):
      configs.append({
          'actor_mode': 'reset',
          'critic_mode': 'decomposed',
          'seed': seed,
          'single_task': task,
          'steps_per_task': 8_000_000,
          'base_steps': 8_000_000,
          'eval_every': 50_000,
          'eval_episodes': 10,
          'dyn_aux_weight': 1.0,
          'combine_mode': 'add',
          'goal_encoder_mode': 'shared',
          'goal_conditioning_mode': 'success_mechanism',
          'use_task_id': False,
          'actor_auto_reset': False,
          'in_trajectory_negative_repeats': 12,
          'counterfactual_rank_interval_steps': 0,
          'counterfactual_oracle_interval_steps': 0,
          'action_landscape_diagnostic_interval_steps': 0,
          'shortcut_diagnostic_interval': 0,
          'log_rl_metrics': False,
          'profile_runtime': True,
          'post_task_eval_scope': 'current',
          'wandb_group': f'GOAL-VALIDITY-PROMOTE-mechanism-{label}',
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

