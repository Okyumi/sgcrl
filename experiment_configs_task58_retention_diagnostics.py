#!/usr/bin/env python3
"""Task-5 retention diagnostics v2: remaining HER/freeze cells + entropy-off.

Fixes the v1 checkpoint-path collision (HER/freeze variants now fingerprint
into ``_ckpt_path``). Re-runs the cells that auto-resumed in v1, and adds an
entropy-off actor ablation.

15 cells on corrected-wrapper handle-press (1M steps, seeds 5/6/7):

  0-2   her_uniform
  3-5   her_final_state
  6-8   her_success_oversample
  9-11  freeze_critic_0p3
 12-14  discounted_entropy_off   (legacy HER γ^Δt, actor entropy bonus off)

All cells enable the online critic phase probe and mid-task checkpoints
every 50k steps. W&B group: TASK58-RETENTION-DIAGNOSTICS-1M-V2
"""
from __future__ import annotations

import argparse
import shlex


TASK = 'sawyer_handle_press_side'
SEEDS = (5, 6, 7)
STEPS_PER_TASK = 1_000_000
WANDB_GROUP = 'TASK58-RETENTION-DIAGNOSTICS-1M-V2'

VARIANTS = (
    {
        'variant': 'her_uniform',
        'her_future_sampling_mode': 'uniform',
        'her_future_discount': 1.0,
        'freeze_critic_after_success_rate': -1.0,
        'use_action_entropy': True,
    },
    {
        'variant': 'her_final_state',
        'her_future_sampling_mode': 'final_state',
        'her_future_discount': -1.0,
        'freeze_critic_after_success_rate': -1.0,
        'use_action_entropy': True,
    },
    {
        'variant': 'her_success_oversample',
        'her_future_sampling_mode': 'success_oversample',
        'her_future_discount': -1.0,
        'her_success_oversample_boost': 9.0,
        'freeze_critic_after_success_rate': -1.0,
        'use_action_entropy': True,
    },
    {
        'variant': 'freeze_critic_0p3',
        'her_future_sampling_mode': 'discounted',
        'her_future_discount': -1.0,
        'freeze_critic_after_success_rate': 0.3,
        'freeze_critic_min_env_steps': 50_000,
        'use_action_entropy': True,
    },
    {
        'variant': 'discounted_entropy_off',
        'her_future_sampling_mode': 'discounted',
        'her_future_discount': -1.0,
        'freeze_critic_after_success_rate': -1.0,
        'use_action_entropy': False,
    },
)


def _shared_base(seed: int) -> dict:
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
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
      'action_effect_enabled': False,
      'success_bc_weight': 0.0,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      'critic_phase_probe_enabled': True,
      'critic_phase_probe_episodes': 10,
      'critic_phase_probe_interaction_threshold': 0.09,
      'critic_phase_probe_mid_reach_threshold': 0.15,
      'mid_task_checkpoint_every': 50_000,
      'her_success_distance_threshold': 0.05,
      'wandb_group': WANDB_GROUP,
  }


def build_configs():
  configs = []
  for variant in VARIANTS:
    for seed in SEEDS:
      config = _shared_base(seed)
      config.update(variant)
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
    for index, config in enumerate(configs):
      print(index, config['variant'], config['seed'],
            f"entropy={config['use_action_entropy']}")
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
