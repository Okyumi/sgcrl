#!/usr/bin/env python3
"""Task5 vs push success-propagation comparison (D1–D4).

Measurement cells (D1–D3 on, D4 off):
  handle_measure × seeds 5/6/7
  push_measure   × seeds 5/6/7

Counterfactual cells (D1–D3 on, D4 inject on):
  handle_inject  × seeds 5/6/7
  push_inject    × seeds 5/6/7

12 cells × 1M steps. W&B: TASK58-SUCCESS-PROP-1M
"""
from __future__ import annotations

import argparse
import shlex


SEEDS = (5, 6, 7)
STEPS_PER_TASK = 1_000_000
WANDB_GROUP = 'TASK58-SUCCESS-PROP-1M'

VARIANTS = (
    {
        'variant': 'handle_measure',
        'single_task': 'sawyer_handle_press_side',
        'success_inject_enabled': False,
        'geometry_family': 'short_contact_mechanism',
    },
    {
        'variant': 'push_measure',
        'single_task': 'sawyer_push',
        'success_inject_enabled': False,
        'geometry_family': 'continuous_object_progress',
    },
    {
        'variant': 'handle_inject',
        'single_task': 'sawyer_handle_press_side',
        'success_inject_enabled': True,
        'geometry_family': 'short_contact_mechanism',
    },
    {
        'variant': 'push_inject',
        'single_task': 'sawyer_push',
        'success_inject_enabled': True,
        'geometry_family': 'continuous_object_progress',
    },
)


def _shared_base(seed: int, task: str, *, inject: bool) -> dict:
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': seed,
      'single_task': task,
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
      'action_effect_enabled': False,
      'success_bc_weight': 0.0,
      'actor_goal_mode': 'her',
      'actor_success_score_weight': 0.0,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      # D1: Task5/8 4-cell critic phase matrix (no-op on push).
      'critic_phase_probe_enabled': True,
      'critic_phase_probe_episodes': 10,
      # HER mass readout.
      'her_phase_log_enabled': True,
      'her_phase_log_ema_decay': 0.99,
      'her_phase_log_every_episodes': 10,
      # D2 / D3.
      'success_trace_log_enabled': True,
      'actor_follow_probe_enabled': True,
      'actor_follow_num_candidates': 32,
      'actor_follow_max_anchors': 16,
      # D4 counterfactual.
      'success_inject_enabled': inject,
      'success_inject_n': 256,
      'success_inject_success_rate': 0.2,
      'success_inject_min_env_steps': 50_000,
      'success_inject_max_attempts': 40,
      'mid_task_checkpoint_every': 50_000,
      'her_future_sampling_mode': 'discounted',
      'her_future_discount': -1.0,
      'freeze_critic_after_success_rate': -1.0,
      'use_action_entropy': True,
      'wandb_group': WANDB_GROUP,
  }


def build_configs():
  configs = []
  for variant in VARIANTS:
    for seed in SEEDS:
      config = _shared_base(
          seed, variant['single_task'],
          inject=bool(variant['success_inject_enabled']))
      config['variant'] = variant['variant']
      config['geometry_family'] = variant['geometry_family']
      configs.append(config)
  return configs


def _emit(config):
  for key, value in config.items():
    if key in ('variant', 'geometry_family'):
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
      print(index, config['variant'], config['single_task'], config['seed'],
            'inject=' + str(config['success_inject_enabled']))
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
