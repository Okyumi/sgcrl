#!/usr/bin/env python3
"""Distral-faithful Task-5 sweep with unit-normalized DCC branches."""
from __future__ import annotations

import argparse
import shlex


SHARED_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
SEEDS = (5, 6, 7)
PREFIX_CHECKPOINT_DIR = (
    '/scratch/yd2247/sgcrl/logs/dcc_shared_scale/'
    'task5_prefix5/checkpoints')
WANDB_GROUP = 'DCC-DISTRAL-UNIT-SHARED-SCALE-TASK5-BRANCH-1M'


def _base_config(seed, shared_repr_scale):
  """Match the completed plain-DCC prefix experiment except for the mode."""
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': seed,
      'num_tasks': 6,
      'start_task': 5,
      'resume_checkpoint_dir': PREFIX_CHECKPOINT_DIR,
      'steps_per_task': 1_000_000,
      'base_steps': 1_000_000,
      'network_width': 1024,
      'critic_depth': 4,
      'actor_depth': 4,
      'dyn_aux_weight': 1.0,
      'shared_repr_scale': shared_repr_scale,
      'shared_repr_normalization': 'unit_distral',
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
      'interaction_weighted_relabeling': False,
      'action_effect_enabled': False,
      'success_bc_weight': 0.0,
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
      'wandb_group': WANDB_GROUP,
  }


def build_configs():
  """Reuse matched Tasks-0-to-4 checkpoints and train only Task 5."""
  return [_base_config(seed, shared_repr_scale)
          for shared_repr_scale in SHARED_SCALES for seed in SEEDS]


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
      print(index, config['shared_repr_scale'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
