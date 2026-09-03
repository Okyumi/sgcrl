#!/usr/bin/env python3
"""Single-task corrected Task-5/Task-8 DCC baselines and dyn-aux ablations."""
from __future__ import annotations

import argparse
import shlex


TASKS = ('sawyer_handle_press_side', 'sawyer_window_close')
SEEDS = (5, 6, 7)
DYN_AUX_WEIGHTS = (1.0, 0.0)
STEPS_PER_TASK = 8_000_000
WANDB_GROUP = 'TASK58-DCC-CORRECTED-FULLNET-8M-DYN-ABLATION'


def build_configs():
  configs = []
  for env_name in TASKS:
    for dyn_aux_weight in DYN_AUX_WEIGHTS:
      for seed in SEEDS:
        configs.append({
            'actor_mode': 'reset',
            'critic_mode': 'decomposed',
            'seed': seed,
            'single_task': env_name,
            'num_tasks': 1,
            'steps_per_task': STEPS_PER_TASK,
            'base_steps': STEPS_PER_TASK,
            'network_width': 1024,
            'critic_depth': 4,
            'actor_depth': 4,
            'dyn_aux_weight': dyn_aux_weight,
            'phi_task_width': 256,
            'phi_task_depth': 4,
            'in_trajectory_negative_repeats': 1,
          'eval_every': 50_000,
          'eval_episodes': 10,
          'eval_record_video': True,
          'eval_video_every': 50_000,
          'eval_video_fps': 20,
          'sawyer_success_mode': 'corrected',
            'goal_conditioning_mode': 'full_state',
            'use_task_id': False,
            'adapt_heads_only': False,
            'encoder_from_base': False,
            'actor_auto_reset': False,
            'counterfactual_rank_interval_steps': 0,
            'counterfactual_oracle_interval_steps': 0,
            'action_landscape_diagnostic_interval_steps': 0,
            'shortcut_diagnostic_interval': 0,
            'log_rl_metrics': True,
            'log_pool_cosine': True,
            'log_mixture_norm': False,
            'log_probe_data': False,
            'profile_runtime': True,
            'intra_eval_previous': False,
            'post_task_eval_scope': 'current',
            'wandb_group': WANDB_GROUP,
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
      print(index, config['single_task'], config['dyn_aux_weight'],
            config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
