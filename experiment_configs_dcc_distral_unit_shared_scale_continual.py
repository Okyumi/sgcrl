#!/usr/bin/env python3
"""From-scratch continual Distral alpha sweep with unit-normalized DCC."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex


SHARED_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
ALLOWED_SEEDS = (5, 6, 7)
ALLOWED_NUM_TASKS = (7, 9, 10)
DEFAULT_SEEDS = (5, 6)
DEFAULT_NUM_TASKS = 7
DEFAULT_STEPS_PER_TASK = 1_000_000
LOG_ROOT = '/scratch/yd2247/sgcrl/logs/dcc_distral_unit_continual'
WANDB_PROJECT = 'continual-contrastive-rl'


def _match_scale(text):
  value = float(text)
  for allowed in SHARED_SCALES:
    if abs(value - allowed) < 1e-9:
      return allowed
  raise SystemExit(
      f'ERROR: invalid alpha {text!r}; allowed values are {SHARED_SCALES}')


def selected_scales():
  """Return the six Distral alphas. A partial or unknown list fails."""
  raw = os.environ.get('ALPHA_SCALES', '').strip()
  if not raw:
    return SHARED_SCALES
  scales = tuple(_match_scale(part) for part in raw.split(',') if part.strip())
  if scales != SHARED_SCALES:
    raise SystemExit(
        'ERROR: ALPHA_SCALES must be exactly '
        f'{SHARED_SCALES}, got {scales}')
  return scales


def selected_seeds():
  """Seeds for this launch. Default is the two-seed pilot, not all three."""
  raw = os.environ.get('ALPHA_SEEDS', '5,6').strip()
  if not raw:
    raise SystemExit('ERROR: ALPHA_SEEDS is empty')
  requested = []
  for part in raw.split(','):
    part = part.strip()
    if not part:
      continue
    try:
      seed = int(part)
    except ValueError as exc:
      raise SystemExit(f'ERROR: invalid seed {part!r}') from exc
    if seed not in ALLOWED_SEEDS:
      raise SystemExit(
          f'ERROR: seed {seed} is not allowed; choose from {ALLOWED_SEEDS}')
    requested.append(seed)
  if not requested:
    raise SystemExit('ERROR: ALPHA_SEEDS is empty')
  if len(requested) != len(set(requested)):
    raise SystemExit(f'ERROR: duplicate seeds in {requested}')
  return tuple(seed for seed in ALLOWED_SEEDS if seed in requested)


def selected_num_tasks():
  raw = os.environ.get('ALPHA_NUM_TASKS', str(DEFAULT_NUM_TASKS)).strip()
  try:
    num_tasks = int(raw)
  except ValueError as exc:
    raise SystemExit(f'ERROR: invalid ALPHA_NUM_TASKS {raw!r}') from exc
  if num_tasks not in ALLOWED_NUM_TASKS:
    raise SystemExit(
        f'ERROR: ALPHA_NUM_TASKS must be one of {ALLOWED_NUM_TASKS}, '
        f'got {num_tasks}')
  return num_tasks


def selected_steps_per_task():
  raw = os.environ.get(
      'ALPHA_STEPS_PER_TASK', str(DEFAULT_STEPS_PER_TASK)).strip()
  try:
    steps = int(raw)
  except ValueError as exc:
    raise SystemExit(f'ERROR: invalid ALPHA_STEPS_PER_TASK {raw!r}') from exc
  if steps <= 0:
    raise SystemExit(
        f'ERROR: ALPHA_STEPS_PER_TASK must be positive, got {steps}')
  return steps


def wandb_group(num_tasks, steps_per_task):
  if steps_per_task == DEFAULT_STEPS_PER_TASK:
    budget = '1M'
  else:
    budget = str(steps_per_task)
  return f'DCC-DISTRAL-UNIT-ALPHA-CONTINUAL{num_tasks}-{budget}'


def scale_tag(shared_repr_scale):
  return f'{shared_repr_scale:g}'.replace('.', 'p')


def run_name(num_tasks, shared_repr_scale, seed):
  return (
      f'continual{num_tasks}_alpha{scale_tag(shared_repr_scale)}_seed{seed}')


def run_root(num_tasks, shared_repr_scale, seed):
  return f'{LOG_ROOT}/{run_name(num_tasks, shared_repr_scale, seed)}'


def is_curriculum_complete(checkpoint_dir, num_tasks):
  """True once the runner has written the final task checkpoint."""
  root = Path(checkpoint_dir)
  if not root.is_dir():
    return False
  last_task = int(num_tasks) - 1
  return any(root.glob(f'**/task_{last_task}.pkl'))


def _base_config(seed, shared_repr_scale, num_tasks, steps_per_task):
  """Plain DCC from task 0. Alpha is the only intentional change."""
  root = run_root(num_tasks, shared_repr_scale, seed)
  return {
      'actor_mode': 'reset',
      'critic_mode': 'decomposed',
      'seed': seed,
      'num_tasks': num_tasks,
      'steps_per_task': steps_per_task,
      'base_steps': steps_per_task,
      'network_width': 1024,
      'critic_depth': 4,
      'actor_depth': 4,
      'dyn_aux_weight': 1.0,
      'shared_repr_scale': shared_repr_scale,
      'shared_repr_normalization': 'unit_distral',
      'phi_task_width': 256,
      'phi_task_depth': 4,
      'combine_mode': 'add',
      'energy_fn': 'inner_product',
      'eval_every': 50_000,
      'eval_episodes': 10,
      'sawyer_success_mode': 'native_info',
      'goal_conditioning_mode': 'full_state',
      'use_task_id': False,
      'actor_auto_reset': False,
      'in_trajectory_negative_repeats': 1,
      'interaction_weighted_relabeling': False,
      'action_effect_enabled': False,
      'success_bc_weight': 0.0,
      'counterfactual_rank_interval_steps': 0,
      'counterfactual_oracle_interval_steps': 0,
      'action_landscape_diagnostic_interval_steps': 0,
      'shortcut_diagnostic_interval': 0,
      'log_rl_metrics': False,
      'log_pool_cosine': False,
      'log_mixture_norm': False,
      'log_probe_data': False,
      'profile_runtime': True,
      'intra_eval_previous': False,
      'post_task_eval_scope': 'current',
      'wandb_project': WANDB_PROJECT,
      'wandb_group': wandb_group(num_tasks, steps_per_task),
      'run_name': run_name(num_tasks, shared_repr_scale, seed),
      'log_dir': f'{root}/runs',
      'checkpoint_dir': f'{root}/checkpoints',
  }


def build_configs():
  """One from-scratch curriculum per alpha and seed. Alpha changes slowest."""
  num_tasks = selected_num_tasks()
  steps_per_task = selected_steps_per_task()
  return [
      _base_config(seed, shared_repr_scale, num_tasks, steps_per_task)
      for shared_repr_scale in selected_scales()
      for seed in selected_seeds()
  ]


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
  group.add_argument('--seeds-per-alpha', action='store_true')
  group.add_argument('--num-alphas', action='store_true')
  group.add_argument('--array-range', action='store_true')
  parser.add_argument('--complete', action='store_true')
  args = parser.parse_args()
  if args.seeds_per_alpha:
    print(len(selected_seeds()))
    return
  if args.num_alphas:
    print(len(selected_scales()))
    return
  if args.array_range:
    print(f'0-{len(selected_scales()) - 1}')
    return
  configs = build_configs()
  if args.total:
    print(len(configs))
    return
  if args.list:
    for index, config in enumerate(configs):
      print(index, config['shared_repr_scale'], config['seed'],
            config['run_name'])
    return
  if args.setting is None or args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  config = configs[args.setting]
  if args.complete:
    done = is_curriculum_complete(
        config['checkpoint_dir'], config['num_tasks'])
    print('complete' if done else 'incomplete')
    raise SystemExit(0 if done else 1)
  _emit(config)


if __name__ == '__main__':
  main()
