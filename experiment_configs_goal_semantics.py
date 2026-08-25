#!/usr/bin/env python3
"""One-million-step goal-contract validity cells for Sawyer Tasks 5 and 8."""
from __future__ import annotations

import argparse
import shlex


TASKS = (
    ('sawyer_handle_press_side', 'task5'),
    ('sawyer_window_close', 'task8'),
)
GOAL_MODES = ('full_state', 'success_mechanism')
SEEDS = (5, 6)


def build_configs():
  configs = []
  for goal_mode in GOAL_MODES:
    for task, label in TASKS:
      for seed in SEEDS:
        configs.append({
            'actor_mode': 'reset',
            'critic_mode': 'decomposed',
            'seed': seed,
            'single_task': task,
            'steps_per_task': 1_000_000,
            'base_steps': 1_000_000,
            'eval_every': 50_000,
            'eval_episodes': 10,
            'dyn_aux_weight': 1.0,
            'combine_mode': 'add',
            'goal_encoder_mode': 'shared',
            'goal_conditioning_mode': goal_mode,
            'use_task_id': False,
            'actor_auto_reset': False,
            'in_trajectory_negative_repeats': 12,
            # This stage isolates goal semantics. Serial MuJoCo probes and
            # representation sweeps are deliberately absent from the hot path.
            'counterfactual_rank_interval_steps': 0,
            'counterfactual_oracle_interval_steps': 0,
            'action_landscape_diagnostic_interval_steps': 0,
            'shortcut_diagnostic_interval': 0,
            'log_rl_metrics': False,
            'profile_runtime': True,
            'post_task_eval_scope': 'current',
            'wandb_group': (
                f'GOAL-VALIDITY-V1-{goal_mode.replace("_", "-")}-{label}'),
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

