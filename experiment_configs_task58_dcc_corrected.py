#!/usr/bin/env python3
"""Clean single-task DCC baselines under the corrected Task-5/8 wrapper."""
from __future__ import annotations

import argparse
import shlex


TASKS = ('sawyer_handle_press_side', 'sawyer_window_close')
SEEDS = (5, 6, 7)


def build_configs():
  configs = []
  for env_name in TASKS:
    for seed in SEEDS:
      configs.append({
          'actor_mode': 'reset',
          'critic_mode': 'decomposed',
          'seed': seed,
          'single_task': env_name,
          'num_tasks': 1,
          'steps_per_task': 1_000_000,
          'base_steps': 1_000_000,
          'network_width': 1024,
          'critic_depth': 4,
          'actor_depth': 4,
          'dyn_aux_weight': 1.0,
          'phi_task_width': 256,
          'phi_task_depth': 4,
          'in_trajectory_negative_repeats': 1,
          'eval_every': 50_000,
          'eval_episodes': 10,
          'sawyer_success_mode': 'corrected',
          'goal_conditioning_mode': 'full_state',
          'use_task_id': False,
          'actor_auto_reset': False,
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
          'wandb_group': 'TASK58-DCC-CORRECTED-REACHABLE-GOAL-1M',
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
  configs = build_configs()
  if args.total:
    print(len(configs))
    return
  if args.list:
    for index, config in enumerate(configs):
      print(index, config['single_task'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
