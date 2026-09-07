#!/usr/bin/env python3
"""Cross-task HER-phase geometry probe (lucky-spike buffer composition).

Compares four Sawyer tasks from the continual sequence under identical
corrected-wrapper DCC training, with online HER future-phase logging:

  continuous progress : sawyer_push
  short-contact mech. : sawyer_handle_press_side, sawyer_faucet_close,
                        sawyer_peg_unplug_side

Prediction: handle/faucet/peg show high hover_unsolved HER fractions after
any success spike; push shows sustained object_progress / success futures.

12 cells × 1M steps × seeds 5/6/7. W&B: TASK58-HER-PHASE-GEOMETRY-1M
"""
from __future__ import annotations

import argparse
import shlex


SEEDS = (5, 6, 7)
STEPS_PER_TASK = 1_000_000
WANDB_GROUP = 'TASK58-HER-PHASE-GEOMETRY-1M'

TASKS = (
    {
        'variant': 'push_progress',
        'single_task': 'sawyer_push',
        'geometry_family': 'continuous_object_progress',
    },
    {
        'variant': 'handle_contact',
        'single_task': 'sawyer_handle_press_side',
        'geometry_family': 'short_contact_mechanism',
    },
    {
        'variant': 'faucet_contact',
        'single_task': 'sawyer_faucet_close',
        'geometry_family': 'short_contact_mechanism',
    },
    {
        'variant': 'peg_contact',
        'single_task': 'sawyer_peg_unplug_side',
        'geometry_family': 'short_contact_mechanism',
    },
)


def _shared_base(seed: int, task: str) -> dict:
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
      'critic_phase_probe_enabled': True,
      'critic_phase_probe_episodes': 10,
      'her_phase_log_enabled': True,
      'her_phase_log_ema_decay': 0.99,
      'her_phase_log_every_episodes': 10,
      'mid_task_checkpoint_every': 50_000,
      'her_future_sampling_mode': 'discounted',
      'her_future_discount': -1.0,
      'freeze_critic_after_success_rate': -1.0,
      'use_action_entropy': True,
      'wandb_group': WANDB_GROUP,
  }


def build_configs():
  configs = []
  for task in TASKS:
    for seed in SEEDS:
      config = _shared_base(seed, task['single_task'])
      config['variant'] = task['variant']
      config['geometry_family'] = task['geometry_family']
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
            config['geometry_family'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
