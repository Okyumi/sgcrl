#!/usr/bin/env python3
"""Staged 1M-step Outcome-Calibrated Sequence DCC falsification cells.

Indices 0--3 are Stage 1, 4--7 Stage 2, and 8--11 Stage 3.  Within each
stage the order is task 5 seeds 5/6 followed by task 8 seeds 5/6.
"""
import argparse
import shlex
import sys


TASKS = (
    ('sawyer_handle_press_side', 'task5'),
    ('sawyer_window_close', 'task8'),
)
SEEDS = (5, 6)


def _base(task, label, seed):
  return {
      'actor_mode': 'reset',
      'critic_mode': 'advantage_decomposed',
      'seed': seed,
      'single_task': task,
      'steps_per_task': 1_000_000,
      'base_steps': 1_000_000,
      'eval_every': 50_000,
      'dyn_aux_weight': 1.0,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      'use_task_id': False,
      'actor_auto_reset': False,
      'in_trajectory_negative_repeats': 12,
      'interaction_weighted_relabeling': False,
      'action_effect_enabled': True,
      'action_effect_loss_weight': 1.0,
      'action_effect_temperature': 1.0,
      'action_effect_actor_weight': 1.0,
      'action_effect_hidden_dim': 256,
      'shortcut_diagnostic_interval': 50,
      'action_landscape_diagnostic_interval_steps': 250_000,
      'action_landscape_num_anchors': 2,
      'action_landscape_candidates_per_family': 4,
      'action_landscape_rollout_horizon': 100,
      'action_landscape_anchor_prefix_steps': 20,
      'action_landscape_local_noise_std': 0.10,
      'action_landscape_interaction_aware_anchor': True,
      'action_landscape_anchor_search_steps': 200,
      'action_landscape_interaction_threshold': 0.09,
      'log_probe_data': False,
      'post_task_eval_scope': 'current',
      '_task_label': label,
  }


def build_configs():
  configs = []
  stages = (
      (1, 'effect-only-psi', {
          'action_effect_actor_mode': 'effect_only',
          'action_effect_target_mode': 'psi_one_step',
          'success_bc_weight': 0.0,
      }),
      (2, 'raw-H25-outcome', {
          'action_effect_actor_mode': 'effect_only',
          'action_effect_target_mode': 'raw_horizon',
          'outcome_horizon': 25,
          'outcome_success_threshold': 0.05,
          'outcome_progress_loss_weight': 1.0,
          'outcome_success_loss_weight': 1.0,
          'outcome_success_actor_weight': 1.0,
          'outcome_progress_ema_decay': 0.99,
          'outcome_progress_std_floor': 0.01,
          'success_bc_weight': 0.0,
      }),
      (3, 'raw-H25-retention', {
          'action_effect_actor_mode': 'effect_only',
          'action_effect_target_mode': 'raw_horizon',
          'outcome_horizon': 25,
          'outcome_success_threshold': 0.05,
          'outcome_progress_loss_weight': 1.0,
          'outcome_success_loss_weight': 1.0,
          'outcome_success_actor_weight': 1.0,
          'outcome_progress_ema_decay': 0.99,
          'outcome_progress_std_floor': 0.01,
          'success_bc_weight': 0.1,
          'success_buffer_capacity': 4096,
          'success_bc_batch_size': 64,
      }),
  )
  for stage, name, overrides in stages:
    for task, label in TASKS:
      for seed in SEEDS:
        cfg = _base(task, label, seed)
        cfg.update(overrides)
        cfg['wandb_group'] = f'OCSDCC-S{stage}-{name}-{label}'
        cfg['falsification_stage'] = stage
        del cfg['_task_label']
        configs.append(cfg)
  return configs


def _emit(config):
  for key, value in config.items():
    if key == 'falsification_stage':
      continue
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
    for index, cfg in enumerate(configs):
      print(index, cfg['falsification_stage'], cfg['wandb_group'],
            cfg['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    print(f'ERROR: setting {args.setting} out of range', file=sys.stderr)
    raise SystemExit(1)
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
