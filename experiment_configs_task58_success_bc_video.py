#!/usr/bin/env python3
"""Task-5 success-buffer BC retry with W&B eval videos.

Six cells: plain corrected-wrapper DCC baseline vs Advantage-DCC + raw-horizon
success retention, seeds 5/6/7 each.
"""
from __future__ import annotations

import argparse
import shlex


TASK = 'sawyer_handle_press_side'
SEEDS = (5, 6, 7)
STEPS_PER_TASK = 4_000_000
WANDB_GROUP = 'TASK58-SUCCESS-BC-VIDEO-4M'


def _shared_base(seed: int) -> dict:
  return {
      'actor_mode': 'reset',
      'seed': seed,
      'single_task': TASK,
      'num_tasks': 1,
      'steps_per_task': STEPS_PER_TASK,
      'base_steps': STEPS_PER_TASK,
      'network_width': 1024,
      'critic_depth': 4,
      'actor_depth': 4,
      'dyn_aux_weight': 1.0,
      'phi_task_width': 256,
      'phi_task_depth': 4,
      'in_trajectory_negative_repeats': 1,
      'eval_every': 50_000,
      'eval_episodes': 10,
      'eval_record_video': True,
      'eval_video_every': 100_000,
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
      'profile_runtime': True,
      'intra_eval_previous': False,
      'post_task_eval_scope': 'current',
      'interaction_weighted_relabeling': False,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      'wandb_group': WANDB_GROUP,
  }


def build_configs():
  configs = []
  for seed in SEEDS:
    baseline = _shared_base(seed)
    baseline.update({
        'variant': 'dcc_baseline',
        'critic_mode': 'decomposed',
        'action_effect_enabled': False,
        'success_bc_weight': 0.0,
    })
    configs.append(baseline)

  for seed in SEEDS:
    retention = _shared_base(seed)
    retention.update({
        'variant': 'success_bc_combined',
        'critic_mode': 'advantage_decomposed',
        'action_effect_enabled': True,
        'action_effect_loss_weight': 1.0,
        'action_effect_temperature': 1.0,
        'action_effect_actor_weight': 1.0,
        'action_effect_actor_mode': 'combined',
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
    })
    configs.append(retention)
  return configs


def _emit(config):
  for key, value in config.items():
    if key == 'variant':
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
    for index, config in enumerate(configs):
      print(index, config['variant'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
