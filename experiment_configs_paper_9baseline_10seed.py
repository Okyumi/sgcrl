#!/usr/bin/env python3
"""Ten-seed paper rerun of the 9-cell contrastive actor/critic transfer grid.

Actor and critic each use reset (R), persistent (P), or CKA (C), giving the
nine baseline cells in the DCC paper. Seeds 5..14 replace the earlier
three-seed (5, 6, 7) runs because the Sawyer wrapper was revised to
``native_info`` success.
"""
from __future__ import annotations

import argparse
import itertools
import shlex


SEEDS = tuple(range(5, 15))
BASE_MODES = ('reset', 'persistent', 'cka')
VARIANTS = tuple(
    (f'{actor}-{critic}', actor, critic)
    for actor, critic in itertools.product(BASE_MODES, BASE_MODES)
)


def _uses_cka(actor_mode, critic_mode):
  return actor_mode == 'cka' or critic_mode == 'cka'


def build_configs():
  configs = []
  for variant, actor_mode, critic_mode in VARIANTS:
    uses_cka = _uses_cka(actor_mode, critic_mode)
    for seed in SEEDS:
      configs.append({
          'actor_mode': actor_mode,
          'critic_mode': critic_mode,
          'seed': seed,
          'num_tasks': 10,
          'steps_per_task': 8_000_000,
          'base_steps': 8_000_000,
          'k_max': 5,
          'eval_every': 100_000,
          'eval_episodes': 10,
          'network_width': 1024,
          'critic_depth': 4,
          'actor_depth': 4,
          'sawyer_success_mode': 'native_info',
          'goal_conditioning_mode': 'full_state',
          'use_task_id': False,
          'actor_auto_reset': False,
          'in_trajectory_negative_repeats': 1,
          'counterfactual_rank_interval_steps': 0,
          'counterfactual_oracle_interval_steps': 0,
          'action_landscape_diagnostic_interval_steps': 0,
          'shortcut_diagnostic_interval': 0,
          'log_rl_metrics': True,
          'rl_metrics_occasional_multiplier': 2,
          'log_pool_cosine': uses_cka,
          'log_mixture_norm': uses_cka,
          'log_probe_data': False,
          'profile_runtime': True,
          'intra_eval_previous': False,
          'post_task_eval_scope': 'current',
          'wandb_group': f'PAPER-9BASELINE-10SEED-{variant}',
      })
  return configs


def checkpoint_config_key(actor_mode, critic_mode):
  """Directory key matching ``run_continual_contrastive._ckpt_path``."""
  return (
      f'actor_{actor_mode}_critic_{critic_mode}'
      f'_tid_False_heads_True_success_native_info'
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
