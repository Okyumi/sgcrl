#!/usr/bin/env python3
"""Mass-matched inject + stage dwell + press-vs-pi (Task5 retention cause).

Cells (1M, seeds 5/6/7):
  0-2  handle_measure      — dwell + pressπ, no inject
  3-5  push_measure        — dwell + pressπ, no inject
  6-8  handle_inject_10pct — clone success eps to ~10% of buffer
  9-11 handle_inject_20pct — clone success eps to ~20% of buffer

W&B: TASK58-MASS-INJECT-DWELL-1M
"""
from __future__ import annotations

import argparse
import shlex


SEEDS = (5, 6, 7)
STEPS_PER_TASK = 1_000_000
WANDB_GROUP = 'TASK58-MASS-INJECT-DWELL-1M'

VARIANTS = (
    {
        'variant': 'handle_measure',
        'single_task': 'sawyer_handle_press_side',
        'success_inject_enabled': False,
        'success_inject_target_frac': 0.0,
    },
    {
        'variant': 'push_measure',
        'single_task': 'sawyer_push',
        'success_inject_enabled': False,
        'success_inject_target_frac': 0.0,
    },
    {
        'variant': 'handle_inject_10pct',
        'single_task': 'sawyer_handle_press_side',
        'success_inject_enabled': True,
        'success_inject_target_frac': 0.10,
    },
    {
        'variant': 'handle_inject_20pct',
        'single_task': 'sawyer_handle_press_side',
        'success_inject_enabled': True,
        'success_inject_target_frac': 0.20,
    },
)


def _shared(seed: int, task: str, *, inject: bool, target_frac: float) -> dict:
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
      'success_trace_log_enabled': True,
      'actor_follow_probe_enabled': True,
      'actor_follow_num_candidates': 32,
      'actor_follow_max_anchors': 16,
      'stage_dwell_log_enabled': True,
      'press_vs_pi_probe_enabled': True,
      'success_inject_enabled': inject,
      'success_inject_n': 256,
      'success_inject_target_frac': target_frac,
      'success_inject_clone': inject,
      'success_inject_success_rate': 0.2,
      'success_inject_min_env_steps': 50_000,
      'success_inject_max_attempts': 80,
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
      config = _shared(
          seed, variant['single_task'],
          inject=bool(variant['success_inject_enabled']),
          target_frac=float(variant['success_inject_target_frac']))
      config['variant'] = variant['variant']
      configs.append(config)
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
    for i, c in enumerate(configs):
      print(i, c['variant'], c['single_task'], c['seed'],
            f"frac={c['success_inject_target_frac']}")
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
