#!/usr/bin/env python3
"""Small corrected-wrapper ablation for Task 5 and Task 8."""
from __future__ import annotations

import argparse
import shlex


TASKS = ('sawyer_handle_press_side', 'sawyer_window_close')
SEEDS = (5, 6)
VARIANTS = (
    {
        'name': 'advantage_1step',
        'wandb_group': 'TASK58-CORRECTED-ADVANTAGE-1STEP-1M',
        'critic_mode': 'advantage_decomposed',
        'action_effect_enabled': True,
        'action_effect_target_mode': 'psi_one_step',
        'success_bc_weight': 0.0,
        'success_bc_label_mode': 'raw_horizon',
    },
    {
        'name': 'terminal_success_bc',
        'wandb_group': 'TASK58-CORRECTED-TERMINAL-SUCCESS-BC-1M',
        'critic_mode': 'decomposed',
        'action_effect_enabled': False,
        'action_effect_target_mode': 'psi_one_step',
        'success_bc_weight': 0.1,
        'success_bc_label_mode': 'terminal_episode',
    },
)


def build_configs():
  configs = []
  for variant in VARIANTS:
    for env_name in TASKS:
      for seed in SEEDS:
        configs.append({
            **variant,
            'actor_mode': 'reset',
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
            'action_effect_actor_mode': 'combined',
            'action_effect_loss_weight': 1.0,
            'action_effect_actor_weight': 1.0,
            'outcome_horizon': 25,
            'outcome_success_threshold': 0.05,
            'outcome_progress_loss_weight': 1.0,
            'outcome_success_loss_weight': 1.0,
            'outcome_success_actor_weight': 1.0,
            'success_buffer_capacity': 4096,
            'success_bc_batch_size': 64,
            'eval_every': 50_000,
            'eval_episodes': 10,
            'sawyer_success_mode': 'corrected',
            'goal_conditioning_mode': 'full_state',
            'use_task_id': False,
            'actor_auto_reset': False,
            'profile_runtime': True,
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
      print(index, config['name'], config['single_task'], config['seed'],
            config['wandb_group'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
