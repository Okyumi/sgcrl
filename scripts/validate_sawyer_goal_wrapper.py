#!/usr/bin/env python3
"""Validate the custom Task-5/Task-8 wrapper against MetaWorld semantics.

This is intentionally a no-training validity gate.  It sends MetaWorld's own
scripted behavior policy through the custom goal-conditioned wrapper while
independently recomputing the native parent's success signal after each step.
It therefore tests the wrapper target, observation contract, action interface,
physics, reward, and behavior-policy solvability before a GPU experiment runs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import goal_semantics


def _native_parent(environment):
  """Find the MetaWorld parent class underneath the local subclass."""
  for parent in type(environment).__mro__[1:]:
    if (parent.__module__.startswith('metaworld.')
        and hasattr(parent, '_get_obs')
        and hasattr(parent, 'evaluate_state')):
      return parent
  raise RuntimeError(
      f'Could not locate a MetaWorld parent for {type(environment)!r}.')


def _native_observation(environment, parent) -> np.ndarray:
  """Bypass the custom ``_get_obs`` override used by the wrapper."""
  return np.asarray(parent._get_obs(environment), dtype=np.float32)


def _native_success(environment, parent, action) -> tuple[float, float]:
  """Re-evaluate the native benchmark predicate at the current state."""
  result = parent.evaluate_state(
      environment, _native_observation(environment, parent), action)
  if not isinstance(result, tuple) or len(result) != 2:
    raise RuntimeError(
        'MetaWorld evaluate_state did not return (reward, info).')
  native_reward, native_info = result
  if 'success' not in native_info:
    raise RuntimeError('MetaWorld evaluate_state info has no success key.')
  return float(native_reward), float(native_info['success'])


def _summary(values) -> float:
  return float(np.mean(values)) if values else float('nan')


def validate_task(env_name: str, *, seed: int, episodes: int,
                  max_steps: int) -> dict[str, Any]:
  """Run one task's expert-policy wrapper contract audit."""
  # Imports stay inside the runtime path so dependency-light unit tests can
  # import this file without MuJoCo, Gym, or MetaWorld installed.
  import env_utils  # pylint: disable=import-outside-toplevel
  from metaworld import policies  # pylint: disable=import-outside-toplevel

  metadata = goal_semantics.VALIDITY_TASKS[env_name]
  policy_type = getattr(policies, metadata['policy_class'])
  policy = policy_type()
  environment, obs_dim, _ = env_utils.load(
      env_name, fixed_start_end=metadata['fixed_goal'])
  if obs_dim != env_utils.STATE_DIM_UNIFIED:
    raise RuntimeError(
        f'Expected unified obs_dim={env_utils.STATE_DIM_UNIFIED}, got {obs_dim}.')
  parent = _native_parent(environment)

  custom_episode_success = []
  native_episode_success = []
  target_errors = []
  predicate_agreement = []
  native_available = []
  successful_hand_mismatch = []
  successful_gripper_mismatch = []
  successful_mechanism_distance = []
  step_count = 0

  np.random.seed(seed)
  for _ in range(episodes):
    observation = np.asarray(environment.reset(), dtype=np.float32)
    contract = goal_semantics.goal_contract_metrics(
        observation, obs_dim, np.asarray(environment._target_pos))
    target_errors.append(contract['target_linf_error'])
    custom_success = 0.0
    native_success = 0.0
    for _ in range(max_steps):
      action = np.asarray(
          policy.get_action(_native_observation(environment, parent)),
          dtype=np.float32)
      observation, reward, done, _ = environment.step(action)
      observation = np.asarray(observation, dtype=np.float32)
      _, native_step_success = _native_success(environment, parent, action)
      custom_step_success = float(float(reward) > 0.0)
      custom_success = max(custom_success, custom_step_success)
      native_success = max(native_success, native_step_success)
      predicate_agreement.append(
          float(custom_step_success == native_step_success))
      native_available.append(1.0)
      step_count += 1
      if custom_step_success > 0.5 or native_step_success > 0.5:
        successful = goal_semantics.goal_contract_metrics(
            observation, obs_dim, np.asarray(environment._target_pos))
        successful_mechanism_distance.append(
            successful['mechanism_distance'])
        successful_hand_mismatch.append(
            successful.get('hand_goal_distance', float('nan')))
        successful_gripper_mismatch.append(
            successful.get('gripper_goal_error', float('nan')))
      if done:
        break
    custom_episode_success.append(custom_success)
    native_episode_success.append(native_success)

  try:
    environment.close()
  except Exception:
    pass
  return {
      'env_name': env_name,
      'seed': seed,
      'episodes': episodes,
      'steps': step_count,
      'target_linf_error_max': float(max(target_errors, default=float('nan'))),
      'success_predicate_agreement': _summary(predicate_agreement),
      'native_success_available_fraction': _summary(native_available),
      'custom_expert_success_rate': _summary(custom_episode_success),
      'native_expert_success_rate': _summary(native_episode_success),
      'successful_mechanism_distance_mean': _summary(
          successful_mechanism_distance),
      'successful_hand_goal_distance_mean': _summary(
          successful_hand_mismatch),
      'successful_gripper_goal_error_mean': _summary(
          successful_gripper_mismatch),
  }


def _passes(result: dict[str, Any], expert_success_min: float) -> bool:
  return all((
      result['target_linf_error_max'] <= 1e-6,
      result['success_predicate_agreement'] >= 0.999,
      result['native_success_available_fraction'] >= 0.999,
      result['custom_expert_success_rate'] >= expert_success_min,
      result['native_expert_success_rate'] >= expert_success_min,
  ))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--task', action='append', choices=tuple(goal_semantics.VALIDITY_TASKS),
      help='Task to validate; repeat for both. Defaults to both tasks.')
  parser.add_argument('--seed', type=int, default=5)
  parser.add_argument('--episodes', type=int, default=50)
  parser.add_argument('--max-steps', type=int, default=150)
  parser.add_argument('--expert-success-min', type=float, default=0.80)
  parser.add_argument('--output', default='logs/goal_validity/wrapper_audit.json')
  parser.add_argument('--strict', action='store_true')
  parser.add_argument('--wandb-project', default='')
  parser.add_argument('--wandb-group', default='GOAL-WRAPPER-VALIDITY')
  args = parser.parse_args()

  tasks = args.task or list(goal_semantics.VALIDITY_TASKS)
  results = [validate_task(
      task, seed=args.seed, episodes=args.episodes,
      max_steps=args.max_steps) for task in tasks]
  payload = {
      'contract': {
          'historical': 'full_state',
          'validity_ablation': 'success_mechanism',
          'mechanism_state_indices': [4, 5, 6],
      },
      'expert_success_min': args.expert_success_min,
      'results': results,
      'all_pass': all(_passes(result, args.expert_success_min)
                      for result in results),
  }
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
  print(json.dumps(payload, indent=2))

  if args.wandb_project:
    try:
      import wandb  # pylint: disable=import-outside-toplevel
    except ImportError as error:
      raise SystemExit(
          'wandb is unavailable; activate the contrastive_rl environment.') \
          from error
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        config={
            'seed': args.seed,
            'episodes': args.episodes,
            'max_steps': args.max_steps,
            'expert_success_min': args.expert_success_min,
            'git_commit': os.environ.get('GIT_COMMIT', 'unknown'),
        },
        name=f'wrapper_audit_s{args.seed}')
    metrics = {'validity/all_pass': float(payload['all_pass'])}
    for result in results:
      prefix = ('validity/task5/' if result['env_name'] ==
                'sawyer_handle_press_side' else 'validity/task8/')
      metrics.update({
          prefix + key: float(value)
          for key, value in result.items()
          if key not in ('env_name',) and isinstance(value, (int, float))
      })
    run.log(metrics)
    run.finish()

  if args.strict and not payload['all_pass']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
