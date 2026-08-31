#!/usr/bin/env python3
"""Small continual pilot for alpha*phi_shared + phi_task in DCC."""
from __future__ import annotations

import argparse
import shlex


SHARED_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
SEEDS = (5, 6, 7)
PREFIX_CHECKPOINT_DIR = (
    '/scratch/yd2247/sgcrl/logs/dcc_shared_scale/'
    'task5_prefix5/checkpoints')


def build_prefix_configs():
  return [_base_config(seed, shared_repr_scale=1.0) | {
      'num_tasks': 5,
      'wandb_group': 'DCC-SHARED-SCALE-TASK5-PREFIX5-1M',
  } for seed in SEEDS]


def _base_config(seed, shared_repr_scale):
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': seed,
      'num_tasks': 6,
      'steps_per_task': 1_000_000,
      'base_steps': 1_000_000,
      'network_width': 1024,
      'critic_depth': 4,
      'actor_depth': 4,
      'dyn_aux_weight': 1.0,
      'shared_repr_scale': shared_repr_scale,
      'phi_task_width': 256,
      'phi_task_depth': 4,
      'combine_mode': 'add',
      'energy_fn': 'inner_product',
      'eval_every': 50_000,
      'eval_episodes': 10,
      'sawyer_success_mode': 'native_info',
      'goal_conditioning_mode': 'full_state',
      'use_task_id': False,
      'actor_auto_reset': False,
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
  }


def build_configs():
  configs = []
  for shared_repr_scale in SHARED_SCALES:
    for seed in SEEDS:
      configs.append(_base_config(seed, shared_repr_scale) | {
          'start_task': 5,
          'resume_checkpoint_dir': PREFIX_CHECKPOINT_DIR,
          'wandb_group': 'DCC-SHARED-SCALE-TASK5-BRANCH-1M',
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
  parser.add_argument('--phase', choices=('prefix', 'sweep'), default='sweep')
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--setting', type=int)
  group.add_argument('--total', action='store_true')
  group.add_argument('--list', action='store_true')
  args = parser.parse_args()
  configs = (build_prefix_configs()
             if args.phase == 'prefix' else build_configs())
  if args.total:
    print(len(configs))
    return
  if args.list:
    for index, config in enumerate(configs):
      print(index, config['shared_repr_scale'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
