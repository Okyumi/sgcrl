#!/usr/bin/env python3
"""Jubail 10-seed DCC ablation vs Success-BC, with the dynamics head off.

Both cells use the decomposed critic and a reset actor. Neither trains the
masked-dynamics auxiliary (mu=0). The proposed cell adds the paper's
terminal-success behavioural-cloning regularizer.
"""
from __future__ import annotations

import argparse
import shlex


SEEDS = tuple(range(5, 15))
NUM_TASKS = 10
WANDB_PROJECT = 'continual_gcrl_paper'

VARIANTS = (
    {
        'variant': 'dcc_no_dyn_no_bc',
        'success_bc_weight': 0.0,
        'wandb_group': 'PAPER-DCC-NODYN-10SEED-ablation',
    },
    {
        'variant': 'dcc_success_bc',
        'success_bc_weight': 0.1,
        'wandb_group': 'PAPER-DCC-NODYN-10SEED-success-bc',
    },
)


def _shared(seed: int, success_bc_weight: float, wandb_group: str) -> dict:
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': seed,
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
      'success_bc_weight': success_bc_weight,
      'success_bc_label_mode': 'episode_sparse_reward',
      'success_buffer_capacity': 4096,
      'success_bc_batch_size': 64,
      'actor_goal_mode': 'her',
      'actor_success_score_weight': 0.0,
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
      'wandb_group': wandb_group,
  }


def build_configs():
  configs = []
  for variant in VARIANTS:
    for seed in SEEDS:
      config = _shared(
          seed, variant['success_bc_weight'], variant['wandb_group'])
      config['variant'] = variant['variant']
      configs.append(config)
  return configs


def checkpoint_config_prefix():
  """Directory prefix matching ``run_continual_contrastive._ckpt_path``."""
  return (
      'actor_reset_critic_decomposed_tid_False_heads_True'
      '_success_native_info_dyn0.000_pt256x4'
  )


def seed_dir_matches(dir_name, success_bc_weight):
  """Return whether a checkpoint folder belongs to this Success-BC setting."""
  prefix = checkpoint_config_prefix()
  if not dir_name.startswith(prefix):
    return False
  has_bc_suffix = ('_bridge_' in dir_name) or ('_ret_' in dir_name)
  return has_bc_suffix == (float(success_bc_weight) > 0.0)


def checkpoint_seed_dir(checkpoint_dir, config):
  """Return the unique seed directory for this cell, if it already exists."""
  from pathlib import Path
  root = Path(checkpoint_dir)
  if not root.exists():
    return None
  matches = []
  for seed_dir in root.glob(f'*/seed_{config["seed"]}'):
    if seed_dir_matches(seed_dir.parent.name, config['success_bc_weight']):
      matches.append(seed_dir)
  if not matches:
    return None
  if len(matches) > 1:
    raise RuntimeError(
        f'Multiple checkpoint dirs for seed={config["seed"]} '
        f'bc={config["success_bc_weight"]}: {matches}')
  return matches[0]


def checkpoint_path(checkpoint_dir, config, task_id):
  seed_dir = checkpoint_seed_dir(checkpoint_dir, config)
  if seed_dir is None:
    return (
        f'{checkpoint_dir}/{checkpoint_config_prefix()}'
        f'/seed_{config["seed"]}/task_{task_id}.pkl')
  return str(seed_dir / f'task_{task_id}.pkl')


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
            f"dyn={config['dyn_aux_weight']}",
            f"bc={config['success_bc_weight']}",
            config['wandb_group'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
