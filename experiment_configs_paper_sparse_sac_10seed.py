#!/usr/bin/env python3
"""Ten-seed paper rerun of the two sparse goal-conditioned SAC cells.

Appendix E.1 reports SAC (R/R) and SAC (P/P) under the step-penalty sparse
reward r in {-1, 0} with reach radius tau=0.05. Seeds 5..14 match the
contrastive 9-cell native-info rerun.
"""
from __future__ import annotations

import argparse
import shlex


SEEDS = tuple(range(5, 15))
VARIANTS = (
    ('reset-reset', 'reset', 'reset'),
    ('persistent-persistent', 'persistent', 'persistent'),
)


def build_configs():
  configs = []
  for variant, actor_mode, critic_mode in VARIANTS:
    for seed in SEEDS:
      configs.append({
          'actor_mode': actor_mode,
          'critic_mode': critic_mode,
          'seed': seed,
          'num_tasks': 10,
          'steps_per_task': 8_000_000,
          'base_steps': 8_000_000,
          'eval_every': 100_000,
          'eval_episodes': 10,
          'network_width': 1024,
          'critic_depth': 4,
          'actor_depth': 4,
          'use_residual': True,
          'sawyer_success_mode': 'native_info',
          'goal_conditioning_mode': 'full_state',
          'use_task_id': False,
          'step_penalty_reward': True,
          'her_reward_threshold': 0.05,
          'log_rl_metrics': True,
          'rl_metrics_occasional_multiplier': 2,
          'actor_auto_reset': False,
          'intra_eval_previous': False,
          'post_task_eval_scope': 'current',
          'auto_resume': True,
          'use_wandb': True,
          'wandb_project': 'continual_gcrl_paper',
          'wandb_entity': 'nyuad_mmvc',
          'wandb_group': f'PAPER-SPARSE-SAC-10SEED-{variant}',
      })
  return configs


def checkpoint_config_key(actor_mode, critic_mode):
  """Directory key matching ``sac.checkpointing.config_key`` for this sweep."""
  return (
      f'actor_{actor_mode}_critic_{critic_mode}'
      f'_tid_False_heads_True_rew_steppen_tau_0p05_success_native_info'
  )


def checkpoint_path(checkpoint_dir, actor_mode, critic_mode, seed, task_id):
  return (
      f'{checkpoint_dir}/{checkpoint_config_key(actor_mode, critic_mode)}'
      f'/seed_{seed}/task_{task_id}.pkl'
  )


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
