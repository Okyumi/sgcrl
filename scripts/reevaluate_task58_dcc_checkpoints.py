#!/usr/bin/env python3
"""Re-evaluate an existing DCC checkpoint under paired Task-5/8 success.

This program loads only ``composed_policy`` from a historical checkpoint and
runs deterministic evaluation episodes.  It does not create a learner,
replay buffer, adder, or training actor.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import numpy as np


FIXED_GOALS = {
    'sawyer_handle_press_side': np.array([-0.07, 0.68, 0.07]),
    'sawyer_window_close': np.array([0.0, 0.80, 0.2]),
}


def _load_checkpoint(path: Path):
  with path.open('rb') as handle:
    checkpoint = pickle.load(handle)
  if not isinstance(checkpoint, Mapping):
    raise TypeError(
        f'Expected checkpoint mapping at {path}, got '
        f'{type(checkpoint).__name__}.')
  if 'composed_policy' not in checkpoint:
    raise KeyError(f'Checkpoint {path} has no composed_policy entry.')
  return checkpoint


def _linear_layers(policy_params, prefix):
  layers = []
  for name, values in policy_params.items():
    if (str(name).startswith(prefix)
        and isinstance(values, Mapping)
        and 'w' in values
        and 'b' in values):
      layers.append((str(name), values))
  return sorted(layers)


def infer_actor_architecture(policy_params):
  """Infer the actor-only network settings needed to apply the checkpoint."""
  residual_layers = _linear_layers(policy_params, 'actor_body/')
  if residual_layers:
    # ResidualMLP has one input projection, actor_depth internal projections,
    # and one output projection.
    actor_depth = len(residual_layers) - 2
    if actor_depth < 4 or actor_depth % 4:
      raise ValueError(
          'Could not infer a valid residual actor depth from parameter '
          f'layers {[name for name, _ in residual_layers]}.')
    network_width = int(np.asarray(residual_layers[0][1]['b']).shape[0])
    input_width = int(np.asarray(residual_layers[0][1]['w']).shape[0])
    return {
        'use_residual': True,
        'network_width': network_width,
        'actor_depth': actor_depth,
        'hidden_layer_sizes': (256, 256),
        'input_width': input_width,
    }

  plain_layers = _linear_layers(policy_params, 'mlp/')
  if not plain_layers:
    raise ValueError(
        'Checkpoint policy has neither actor_body nor mlp linear layers; '
        f'top-level keys are {list(policy_params)}.')
  hidden_layer_sizes = tuple(
      int(np.asarray(values['b']).shape[0]) for _, values in plain_layers)
  input_width = int(np.asarray(plain_layers[0][1]['w']).shape[0])
  return {
      'use_residual': False,
      'network_width': hidden_layer_sizes[0],
      'actor_depth': 4,
      'hidden_layer_sizes': hidden_layer_sizes,
      'input_width': input_width,
  }


class _FixedVariableSource:

  def __init__(self, params):
    self._params = params

  def get_variables(self, names):
    return [self._params for _ in names]


def _validate_checkpoint_metadata(checkpoint, env_name, task_id):
  stored_env = checkpoint.get('env_name')
  if stored_env is not None and stored_env != env_name:
    raise ValueError(
        f'Checkpoint env_name is {stored_env!r}, not requested {env_name!r}.')
  stored_task = checkpoint.get('task_id')
  if stored_task is not None and int(stored_task) != task_id:
    raise ValueError(
        f'Checkpoint task_id is {stored_task}, not requested {task_id}.')
  stored_goal_mode = checkpoint.get('goal_conditioning_mode', 'full_state')
  if stored_goal_mode != 'full_state':
    raise ValueError(
        'This paired reevaluator requires the historical full_state goal; '
        f'checkpoint records {stored_goal_mode!r}.')
  stored_success_mode = checkpoint.get(
      'sawyer_success_mode', 'legacy_distance')
  if stored_success_mode != 'legacy_distance':
    raise ValueError(
        'Expected a historical legacy_distance checkpoint, but checkpoint '
        f'records {stored_success_mode!r}.')


def reevaluate(args):
  # Keep CLI parsing and --help dependency-light. These imports are available
  # in the project's contrastive_rl cluster environment, where MuJoCo runs.
  from acme import specs
  from acme import environment_loop
  from acme.agents.jax import actor_core as actor_core_lib
  from acme.agents.jax import actors
  from acme.jax import variable_utils
  import jax
  import jax.numpy as jnp

  from contrastive import networks as contrastive_networks
  from contrastive import task58_reevaluation
  from contrastive import utils as contrastive_utils

  checkpoint_path = Path(args.checkpoint).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
  checkpoint = _load_checkpoint(checkpoint_path)
  _validate_checkpoint_metadata(checkpoint, args.env_name, args.task_id)
  policy_params = jax.tree_util.tree_map(
      lambda value: jnp.asarray(value)
      if isinstance(value, np.ndarray) else value,
      checkpoint['composed_policy'])
  architecture = infer_actor_architecture(policy_params)

  start_index = int(checkpoint.get('goal_start_index', 0))
  end_index = int(checkpoint.get('goal_end_index', -1))
  task_id_for_obs = args.task_id if args.use_task_id else None
  num_tasks_for_obs = args.num_tasks if args.use_task_id else None
  # Run the historical environment and independently score both definitions
  # from every resulting state.  Environment reward cannot affect this actor.
  environment, obs_dim = contrastive_utils.make_environment(
      args.env_name, start_index, end_index, args.seed + args.task_id + 9999,
      fixed_start_end=FIXED_GOALS[args.env_name],
      task_id=task_id_for_obs, num_tasks=num_tasks_for_obs,
      sawyer_success_mode='legacy_distance')

  observation_width = int(np.prod(environment.observation_spec().shape))
  if observation_width != architecture['input_width']:
    raise ValueError(
        'Checkpoint/environment observation width mismatch: policy expects '
        f"{architecture['input_width']}, evaluator exposes {observation_width}. "
        'Check --use-task-id/--num-tasks and the source run configuration.')

  environment_spec = specs.make_environment_spec(environment)
  networks = contrastive_networks.make_networks(
      environment_spec,
      obs_dim=obs_dim,
      hidden_layer_sizes=architecture['hidden_layer_sizes'],
      use_residual=architecture['use_residual'],
      network_width=architecture['network_width'],
      actor_depth=architecture['actor_depth'])
  policy = contrastive_networks.apply_policy_and_sample(
      networks, eval_mode=True)
  actor_core = actor_core_lib.batched_feed_forward_to_actor_core(policy)
  variable_client = variable_utils.VariableClient(
      _FixedVariableSource(policy_params), '', device='cpu')
  actor = actors.GenericActor(
      actor_core,
      jax.random.PRNGKey(args.seed + args.task_id + 5000),
      variable_client,
      backend='cpu')
  observer = task58_reevaluation.PairedTask58SuccessObserver(
      obs_dim, args.env_name)
  loop = environment_loop.EnvironmentLoop(
      environment, actor, observers=[observer])

  episodes = []
  for episode_index in range(args.episodes):
    result = loop.run_episode()
    paired = {
        key: float(result[key])
        for key in (
            'legacy_success',
            'task_axis_success',
            'axis_rescued_success',
            'legacy_min_distance',
            'task_axis_min_distance',
            'legacy_reward_mismatch_steps')
    }
    paired['episode'] = episode_index
    episodes.append(paired)

  try:
    environment.close()
  except Exception:
    pass

  mismatch_steps = int(sum(
      episode['legacy_reward_mismatch_steps'] for episode in episodes))
  if mismatch_steps:
    raise RuntimeError(
        'Observation-based legacy scorer disagreed with emitted legacy reward '
        f'on {mismatch_steps} steps; refusing to report a paired comparison.')

  def _mean(key):
    return float(np.mean([episode[key] for episode in episodes]))

  payload = {
      'status': 'finished',
      'mode': 'checkpoint_reevaluation_only',
      'checkpoint': str(checkpoint_path),
      'checkpoint_task_id': args.task_id,
      'env_name': args.env_name,
      'seed': args.seed,
      'episodes': args.episodes,
      'use_task_id': args.use_task_id,
      'num_tasks': args.num_tasks,
      'goal_conditioning_mode': 'full_state',
      'source_success_mode': 'legacy_distance',
      'actor_architecture': architecture,
      'summary': {
          'legacy_success_rate': _mean('legacy_success'),
          'task_axis_success_rate': _mean('task_axis_success'),
          'axis_rescued_success_rate': _mean('axis_rescued_success'),
          'success_rate_gain': (
              _mean('task_axis_success') - _mean('legacy_success')),
          'legacy_reward_mismatch_steps': mismatch_steps,
      },
      'per_episode': episodes,
  }

  output_path = Path(args.output).expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
      json.dumps(payload, indent=2, sort_keys=True) + '\n',
      encoding='utf-8')
  print(json.dumps(payload['summary'], sort_keys=True), flush=True)
  print(f'Results written to {output_path}', flush=True)
  return payload


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description='Re-evaluate one historical DCC actor; never train.')
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument(
      '--env-name', required=True, choices=tuple(FIXED_GOALS))
  parser.add_argument('--task-id', required=True, type=int, choices=(5, 8))
  parser.add_argument('--seed', required=True, type=int)
  parser.add_argument('--episodes', type=int, default=100)
  parser.add_argument('--output', required=True)
  parser.add_argument('--use-task-id', action='store_true')
  parser.add_argument('--num-tasks', type=int, default=10)
  args = parser.parse_args(argv)
  expected_env = {
      5: 'sawyer_handle_press_side',
      8: 'sawyer_window_close',
  }[args.task_id]
  if args.env_name != expected_env:
    parser.error(
        f'--task-id={args.task_id} requires --env-name={expected_env}')
  if args.episodes <= 0:
    parser.error('--episodes must be positive')
  if args.num_tasks <= args.task_id:
    parser.error('--num-tasks must be larger than --task-id')
  return args


def main():
  # Make it explicit in process manifests that no accelerator training state
  # is allocated.  Actor inference still uses JAX's CPU backend.
  os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
  reevaluate(parse_args())


if __name__ == '__main__':
  main()
