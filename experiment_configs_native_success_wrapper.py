#!/usr/bin/env python3
"""Cheap ten-task smoke matrix for corrected MetaWorld success semantics."""
from __future__ import annotations

import argparse
import shlex


METHODS = (
    ('dcc', 'reset', 'decomposed'),
    ('reset-reset', 'reset', 'reset'),
    ('persistent-persistent', 'persistent', 'persistent'),
)
SEEDS = (5, 6, 7)


def build_configs():
  configs = []
  for method, actor_mode, critic_mode in METHODS:
    for seed in SEEDS:
      configs.append({
          'actor_mode': actor_mode,
          'critic_mode': critic_mode,
          'seed': seed,
          'num_tasks': 10,
          'steps_per_task': 100_000,
          'base_steps': 100_000,
          'eval_every': 100_000,
          'eval_episodes': 5,
          'sawyer_success_mode': 'native_info',
          # Preserve the paper architecture; only reward/success semantics
          # change in this paired revalidation.
          'goal_conditioning_mode': 'full_state',
          'use_task_id': False,
          'actor_auto_reset': False,
          'dyn_aux_weight': 1.0,
          'in_trajectory_negative_repeats': 1,
          # Disable every serial simulator probe and expensive representation
          # sweep. These cells test correctness and throughput, not causality.
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
          'wandb_group': f'NATIVE-SUCCESS-V1-10TASK-SMOKE-{method}',
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
      print(index, config['wandb_group'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
