#!/usr/bin/env python3
"""Fail-closed simulator smoke test for the corrected Task-5/Task-8 wrapper.

This is intentionally evaluation-only.  It checks the observation/goal
contract at reset and after every transition, verifies the sparse success
signal independently from simulator state, and runs MetaWorld's scripted
policies to ensure that each historical fixed goal is physically solvable
inside the 150-step training horizon.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import goal_semantics
from contrastive import sawyer_success


TASKS = tuple(goal_semantics.VALIDITY_TASKS)


def _reset(environment) -> np.ndarray:
  result = environment.reset()
  if isinstance(result, tuple) and len(result) == 2:
    result = result[0]
  return np.asarray(result, dtype=np.float32)


def _step(environment, action):
  result = environment.step(action)
  if not isinstance(result, tuple):
    raise RuntimeError('Environment step did not return a tuple.')
  if len(result) == 4:
    observation, reward, done, info = result
  elif len(result) == 5:
    observation, reward, terminated, truncated, info = result
    done = bool(terminated or truncated)
  else:
    raise RuntimeError(
        f'Expected a 4- or 5-item transition; got {len(result)} items.')
  return (np.asarray(observation, dtype=np.float32), float(reward),
          bool(done), dict(info or {}))


def _native_parent(environment):
  for parent in type(environment).__mro__[1:]:
    if (parent.__module__.startswith('metaworld.')
        and hasattr(parent, '_get_obs')
        and hasattr(parent, 'evaluate_state')):
      return parent
  raise RuntimeError(
      f'No MetaWorld parent found for {type(environment).__name__}.')


def _mechanism(environment) -> np.ndarray:
  position = np.asarray(
      environment._get_pos_objects(), dtype=np.float32).reshape(-1)
  if position.size != 3:
    raise RuntimeError(
        f'Task-5/Task-8 mechanism must be 3-D; got {position.shape}.')
  return position


def _observation_errors(observation, environment, obs_dim, fixed_goal):
  if observation.shape != (2 * obs_dim,):
    raise RuntimeError(
        f'Expected observation shape {(2 * obs_dim,)}, got '
        f'{observation.shape}.')
  state_mechanism = observation[4:7]
  goal_mechanism = observation[obs_dim + 4:obs_dim + 7]
  return {
      'state_mechanism_linf_error': float(np.max(np.abs(
          state_mechanism - _mechanism(environment)))),
      'goal_mechanism_linf_error': float(np.max(np.abs(
          goal_mechanism - fixed_goal))),
      'internal_goal_linf_error': float(np.max(np.abs(
          np.asarray(environment._goal, dtype=np.float32) - fixed_goal))),
      'internal_target_linf_error': float(np.max(np.abs(
          np.asarray(environment._target_pos, dtype=np.float32)
          - fixed_goal))),
  }


def _expected_success(mechanism, fixed_goal, env_name):
  metadata = goal_semantics.VALIDITY_TASKS[env_name]
  distance = goal_semantics.success_axis_distance(
      mechanism, fixed_goal, env_name)
  return float(distance <= float(metadata['success_threshold'])), distance


def classify_task(summary, *, expert_success_min, observation_tolerance,
                  goal_tolerance, zero_action_tolerance):
  """Return a deterministic pass/fail decision and all failed gates."""
  gates = {
      'reported_training_horizon_is_150':
          summary['reported_training_horizon'] == 150,
      'reset_is_not_already_successful':
          summary['reset_success_count'] == 0,
      'reset_and_rollout_observations_match_simulator':
          summary['state_mechanism_linf_error_max'] <= observation_tolerance,
      'exposed_goal_matches_fixed_goal':
          summary['goal_mechanism_linf_error_max'] <= goal_tolerance,
      'internal_goal_matches_fixed_goal':
          summary['internal_goal_linf_error_max'] <= goal_tolerance,
      'internal_target_matches_fixed_goal':
          summary['internal_target_linf_error_max'] <= goal_tolerance,
      'zero_action_has_no_reset_jump':
          summary['zero_action_mechanism_displacement_max']
          <= zero_action_tolerance,
      'reward_matches_axis_predicate':
          summary['reward_axis_mismatch_steps'] == 0,
      'info_matches_axis_predicate':
          summary['info_axis_mismatch_steps'] == 0,
      'scripted_policy_solves_by_training_horizon':
          summary['success_by_training_horizon_rate'] >= expert_success_min,
  }
  failed = [name for name, passed in gates.items() if not passed]
  return {'passed': not failed, 'failed_gates': failed, 'gates': gates}


def _update_maxima(maxima: dict[str, float], values: dict[str, float]):
  for key, value in values.items():
    maxima[key] = max(maxima.get(key, 0.0), float(value))


def smoke_task(env_name: str, *, seeds: Sequence[int], episodes: int,
               training_horizon: int, max_steps: int,
               expert_success_min: float, observation_tolerance: float,
               goal_tolerance: float,
               zero_action_tolerance: float) -> dict[str, Any]:
  """Exercise one corrected wrapper with independent simulator checks."""
  import env_utils  # pylint: disable=import-outside-toplevel
  from metaworld import policies  # pylint: disable=import-outside-toplevel

  metadata = goal_semantics.VALIDITY_TASKS[env_name]
  fixed_goal = np.asarray(metadata['fixed_goal'], dtype=np.float32)
  policy_type = getattr(policies, metadata['policy_class'])
  rows = []
  maxima: dict[str, float] = {}
  reset_success_count = 0
  reward_axis_mismatch_steps = 0
  info_axis_mismatch_steps = 0
  success_by_training_horizon_count = 0
  success_by_max_steps_count = 0
  post_horizon_success_count = 0
  zero_displacements = []
  reported_horizons = set()

  for seed in seeds:
    np.random.seed(seed)
    environment, obs_dim, reported_horizon = env_utils.load(
        env_name,
        fixed_start_end=fixed_goal,
        sawyer_success_mode='corrected')
    reported_horizons.add(int(reported_horizon))
    parent = _native_parent(environment)
    try:
      for episode in range(episodes):
        # A separate reset/zero transition catches the Task-8 stale-site bug
        # without changing the scripted-policy trajectory being evaluated.
        zero_reset_observation = _reset(environment)
        zero_before = _mechanism(environment)
        zero_errors = _observation_errors(
            zero_reset_observation, environment, obs_dim, fixed_goal)
        zero_expected, zero_initial_distance = _expected_success(
            zero_before, fixed_goal, env_name)
        reset_success_count += int(zero_expected)
        zero_observation, zero_reward, _, zero_info = _step(
            environment, np.zeros(environment.action_space.shape,
                                  dtype=np.float32))
        zero_after = _mechanism(environment)
        zero_displacements.append(float(np.max(np.abs(
            zero_after - zero_before))))
        zero_step_errors = _observation_errors(
            zero_observation, environment, obs_dim, fixed_goal)
        _update_maxima(maxima, zero_errors)
        _update_maxima(maxima, zero_step_errors)
        zero_step_expected, _ = _expected_success(
            zero_after, fixed_goal, env_name)
        reward_axis_mismatch_steps += int(
            float(zero_reward > 0.0) != zero_step_expected)
        info_axis_mismatch_steps += int(
            float(zero_info.get('success', -1.0)) != zero_step_expected)

        observation = _reset(environment)
        initial_mechanism = _mechanism(environment)
        initial_errors = _observation_errors(
            observation, environment, obs_dim, fixed_goal)
        _update_maxima(maxima, initial_errors)
        initial_success, initial_distance = _expected_success(
            initial_mechanism, fixed_goal, env_name)
        reset_success_count += int(initial_success)
        policy = policy_type()
        first_success_step = None
        min_axis_distance = initial_distance
        steps = 0

        for step_index in range(1, max_steps + 1):
          policy_observation = sawyer_success.native_observation(
              environment, parent)
          action = np.asarray(
              policy.get_action(policy_observation), dtype=np.float32)
          action = np.clip(
              action, environment.action_space.low,
              environment.action_space.high)
          observation, reward, done, info = _step(environment, action)
          steps = step_index
          errors = _observation_errors(
              observation, environment, obs_dim, fixed_goal)
          _update_maxima(maxima, errors)
          expected, axis_distance = _expected_success(
              _mechanism(environment), fixed_goal, env_name)
          min_axis_distance = min(min_axis_distance, axis_distance)
          reward_axis_mismatch_steps += int(
              float(reward > 0.0) != expected)
          info_axis_mismatch_steps += int(
              float(info.get('success', -1.0)) != expected)
          if expected and first_success_step is None:
            first_success_step = step_index
            break
          if done:
            break

        solved_by_horizon = (
            first_success_step is not None
            and first_success_step <= training_horizon)
        solved_by_max = first_success_step is not None
        success_by_training_horizon_count += int(solved_by_horizon)
        success_by_max_steps_count += int(solved_by_max)
        post_horizon_success_count += int(
            solved_by_max and not solved_by_horizon)
        rows.append({
            'seed': int(seed),
            'episode': episode,
            'zero_reset_axis_distance': zero_initial_distance,
            'initial_axis_distance': initial_distance,
            'minimum_axis_distance': min_axis_distance,
            'first_success_step': first_success_step,
            'steps': steps,
        })
    finally:
      try:
        environment.close()
      except Exception:
        pass

  total = len(rows)
  if reported_horizons != {training_horizon}:
    reported_horizon = (-1 if len(reported_horizons) != 1
                        else next(iter(reported_horizons)))
  else:
    reported_horizon = training_horizon
  summary = {
      'env_name': env_name,
      'success_axis_state_indices': list(
          metadata['success_state_indices']),
      'success_threshold': float(metadata['success_threshold']),
      'episodes': total,
      'reported_training_horizon': int(reported_horizon),
      'reset_success_count': reset_success_count,
      'zero_action_mechanism_displacement_max': max(
          zero_displacements, default=float('nan')),
      'reward_axis_mismatch_steps': reward_axis_mismatch_steps,
      'info_axis_mismatch_steps': info_axis_mismatch_steps,
      'success_by_training_horizon_rate': (
          success_by_training_horizon_count / max(total, 1)),
      'success_by_max_steps_rate': success_by_max_steps_count / max(total, 1),
      'post_horizon_success_rate': post_horizon_success_count / max(total, 1),
      **{key + '_max': value for key, value in maxima.items()},
  }
  summary['classification'] = classify_task(
      summary,
      expert_success_min=expert_success_min,
      observation_tolerance=observation_tolerance,
      goal_tolerance=goal_tolerance,
      zero_action_tolerance=zero_action_tolerance)
  return {'summary': summary, 'episodes': rows}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seeds', type=int, nargs='+', default=(5, 6, 7))
  parser.add_argument('--episodes', type=int, default=5)
  parser.add_argument('--training-horizon', type=int, default=150)
  parser.add_argument('--max-steps', type=int, default=200)
  parser.add_argument('--expert-success-min', type=float, default=0.80)
  parser.add_argument('--observation-tolerance', type=float, default=1e-5)
  parser.add_argument('--goal-tolerance', type=float, default=1e-6)
  parser.add_argument('--zero-action-tolerance', type=float, default=0.02)
  parser.add_argument(
      '--output', type=Path,
      default=Path('logs/wrapper_smoke/task58_corrected_wrapper.json'))
  args = parser.parse_args()
  if args.max_steps < args.training_horizon:
    parser.error('--max-steps must be at least --training-horizon.')

  results = {
      'protocol': 'task58_corrected_wrapper_smoke_v1',
      'success_mode': 'corrected',
      'seeds': args.seeds,
      'episodes_per_seed': args.episodes,
      'training_horizon': args.training_horizon,
      'max_steps': args.max_steps,
      'tasks': {},
  }
  for env_name in TASKS:
    results['tasks'][env_name] = smoke_task(
        env_name,
        seeds=args.seeds,
        episodes=args.episodes,
        training_horizon=args.training_horizon,
        max_steps=args.max_steps,
        expert_success_min=args.expert_success_min,
        observation_tolerance=args.observation_tolerance,
        goal_tolerance=args.goal_tolerance,
        zero_action_tolerance=args.zero_action_tolerance)

  results['passed'] = all(
      result['summary']['classification']['passed']
      for result in results['tasks'].values())
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(results, indent=2, sort_keys=True) + '\n',
      encoding='utf-8')
  print(json.dumps({
      'output': str(args.output),
      'passed': results['passed'],
      'tasks': {
          env_name: result['summary']
          for env_name, result in results['tasks'].items()
      },
  }, indent=2, sort_keys=True))
  if not results['passed']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
