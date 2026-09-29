#!/usr/bin/env python3
"""From-scratch 5-seed full DCC: decomposed critic + gated success matching.

Paper stack, tasks 0–9, 8M steps each. Local-σ_s critic gate, whole-episode
success buffer, λ=0.1. Seeds 5–9.
"""
from __future__ import annotations

import argparse
import shlex


SEEDS = tuple(range(5, 10))
NUM_TASKS = 10
WANDB_PROJECT = 'continual_gcrl_paper'
WANDB_GROUP = 'PAPER-DCC-FULL-CRITICGATE-5SEED'


def _shared(seed: int) -> dict:
  return {
      'variant': 'dcc_gated_success_matching',
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': seed,
      'start_task': 0,
      'num_tasks': NUM_TASKS,
      'steps_per_task': 8_000_000,
      'base_steps': 8_000_000,
      'k_max': 5,
      'eval_every': 100_000,
      'eval_episodes': 10,
      'eval_record_video': False,
      'network_width': 1024,
      'critic_depth': 4,
      'actor_depth': 4,
      'use_residual': True,
      'sawyer_success_mode': 'native_info',
      'goal_conditioning_mode': 'full_state',
      'use_task_id': False,
      'adapt_heads_only': True,
      'encoder_from_base': False,
      'actor_auto_reset': False,
      'dyn_aux_weight': 0.0,
      'dyn_aux_after_task0': -1.0,
      'phi_task_width': 256,
      'phi_task_depth': 4,
      'combine_mode': 'add',
      'goal_encoder_mode': 'shared',
      'in_trajectory_negative_repeats': 1,
      'action_effect_enabled': False,
      'success_bc_weight': 0.1,
      'success_bc_label_mode': 'episode_sparse_reward',
      'success_bc_critic_gate': True,
      'success_bc_critic_gate_mode': 'local',
      'success_bc_critic_state_noise': 0.1,
      'success_buffer_capacity': 4096,
      'success_bc_batch_size': 64,
      'actor_goal_mode': 'her',
      'actor_success_score_weight': 0.0,
      'truncate_on_success': False,
      'counterfactual_rank_interval_steps': 0,
      'counterfactual_oracle_interval_steps': 0,
      'action_landscape_diagnostic_interval_steps': 0,
      'shortcut_diagnostic_interval': 0,
      'log_rl_metrics': True,
      'rl_metrics_occasional_multiplier': 2,
      'log_pool_cosine': False,
      'log_mixture_norm': False,
      'log_probe_data': False,
      'profile_runtime': True,
      'num_actors': 2,
      'intra_eval_previous': False,
      'post_task_eval_scope': 'current',
      'her_phase_log_enabled': False,
      'success_trace_log_enabled': False,
      'actor_follow_probe_enabled': False,
      'stage_dwell_log_enabled': False,
      'press_vs_pi_probe_enabled': False,
      'critic_phase_probe_enabled': False,
      'success_inject_enabled': False,
      'use_action_entropy': True,
      'wandb_project': WANDB_PROJECT,
      'wandb_group': WANDB_GROUP,
  }


def build_configs():
  return [_shared(seed) for seed in SEEDS]


def checkpoint_config_prefix():
  return (
      'actor_reset_critic_decomposed_tid_False_heads_True'
      '_success_native_info_dyn0.000_pt256x4'
  )


def checkpoint_seed_dir(checkpoint_dir, config):
  from pathlib import Path
  root = Path(checkpoint_dir)
  if not root.exists():
    return None
  matches = [
      seed_dir for seed_dir in root.glob(f'*/seed_{config["seed"]}')
      if seed_dir.parent.name.startswith(checkpoint_config_prefix())
      and '_bridge_' in seed_dir.parent.name
  ]
  if not matches:
    return None
  if len(matches) > 1:
    raise RuntimeError(
        f'Multiple checkpoint dirs for seed={config["seed"]}: {matches}')
  return matches[0]


def _emit(config):
  skip = {'variant'}
  for key, value in config.items():
    if key in skip:
      continue
    if isinstance(value, bool):
      value = 'true' if value else 'false'
    elif isinstance(value, str):
      value = shlex.quote(value)
    print(f'{key.upper()}={value}')
  print(f"VARIANT={shlex.quote(config['variant'])}")


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
            f"gate={config['success_bc_critic_gate_mode']}",
            config['wandb_group'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
