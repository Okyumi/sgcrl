#!/usr/bin/env python3
"""Paired positive controls for Sawyer Task-5/Task-8 goal semantics.

For each official MetaWorld ML1 task/reset this script compares:

1. the untouched native environment with MetaWorld's scripted policy;
2. the custom goal-conditioned wrapper preserving the native reset target;
3. the custom wrapper overwriting that target with the historical fixed goal.

It also replays the exact native action sequence through both wrappers. The
replay separates a target/reward mismatch from changes in policy observations
or an invalid scripted-policy adapter. Imports requiring MuJoCo/MetaWorld are
kept inside runtime functions so the decision logic remains dependency-light.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import goal_semantics


CONDITIONS = (
    'native_official',
    'wrapper_native_target_policy',
    'wrapper_fixed_target_policy',
    'wrapper_native_target_replay',
    'wrapper_fixed_target_replay',
)

DECISION_CODES = {
    'fixed_target_valid': 0,
    'fixed_global_target_misaligned': 1,
    'custom_wrapper_invalid': 2,
    'native_positive_control_failed': 3,
    'inconclusive': 4,
}


def _mean(values: Iterable[float]) -> float:
  finite = [float(value) for value in values if math.isfinite(float(value))]
  return float(np.mean(finite)) if finite else float('nan')


def _maximum(values: Iterable[float]) -> float:
  finite = [float(value) for value in values if math.isfinite(float(value))]
  return max(finite) if finite else float('nan')


def _minimum(values: Iterable[float]) -> float:
  finite = [float(value) for value in values if math.isfinite(float(value))]
  return min(finite) if finite else float('nan')


def _as_float_array(value) -> np.ndarray:
  return np.asarray(value, dtype=np.float32).copy()


def _step_result(environment, action):
  """Normalize old Gym and Gymnasium step signatures."""
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
        f'Expected a 4- or 5-tuple from step, received {len(result)} items.')
  return observation, float(reward), bool(done), dict(info or {})


def _reset_result(environment):
  """Normalize old Gym and Gymnasium reset signatures."""
  result = environment.reset()
  if isinstance(result, tuple) and len(result) == 2:
    return result[0]
  return result


def _native_parent(environment):
  """Find the MetaWorld class below a local goal-conditioned subclass."""
  for parent in type(environment).__mro__[1:]:
    if (parent.__module__.startswith('metaworld.')
        and hasattr(parent, '_get_obs')
        and hasattr(parent, 'evaluate_state')):
      return parent
  raise RuntimeError(
      f'Could not locate a MetaWorld parent for {type(environment)!r}.')


def _native_observation(environment, parent) -> np.ndarray:
  """Obtain the raw observation expected by MetaWorld scripted policies."""
  return _as_float_array(parent._get_obs(environment))


def _mechanism_position(environment) -> np.ndarray:
  position = _as_float_array(environment._get_pos_objects()).reshape(-1)
  if position.size < 3:
    raise RuntimeError(
        f'Expected a 3-D mechanism position, got shape {position.shape}.')
  return position[-3:]


def _last_rand_vec(environment) -> np.ndarray:
  value = getattr(environment, '_last_rand_vec', None)
  if value is None:
    raise RuntimeError(
        'MetaWorld environment has no _last_rand_vec after set_task/reset.')
  return _as_float_array(value).reshape(-1)


def _nearest_site(environment, target: np.ndarray) -> dict[str, Any]:
  """Describe the rendered/physical site nearest a target, when available."""
  names = getattr(environment.model, 'site_names', ())
  best_name = ''
  best_position = None
  best_distance = float('inf')
  for raw_name in names:
    name = (raw_name.decode('utf-8') if isinstance(raw_name, bytes)
            else str(raw_name))
    try:
      position = _as_float_array(
          environment._get_site_pos(name)).reshape(-1)[-3:]
    except Exception:
      continue
    distance = _distance(position, target)
    if distance < best_distance:
      best_name = name
      best_position = position
      best_distance = distance
  return {
      'name': best_name,
      'position': (best_position.tolist()
                   if best_position is not None else []),
      'distance': (best_distance if best_position is not None
                   else float('nan')),
  }


def _evaluate_native(environment, parent, action) -> tuple[float, float]:
  result = parent.evaluate_state(
      environment, _native_observation(environment, parent), action)
  if not isinstance(result, tuple) or len(result) != 2:
    raise RuntimeError(
        'MetaWorld evaluate_state did not return (reward, info).')
  reward, info = result
  if 'success' not in info:
    raise RuntimeError('MetaWorld evaluate_state info has no success key.')
  return float(reward), float(info['success'])


def _set_task_and_reset(environment, task) -> np.ndarray:
  """Apply one official ML1 task and reset to its frozen random vector."""
  environment.set_task(task)
  return _as_float_array(_reset_result(environment))


def _distance(position: np.ndarray, target: np.ndarray) -> float:
  return float(np.linalg.norm(
      _as_float_array(position) - _as_float_array(target)))


def _rollout(
    environment,
    parent,
    *,
    policy,
    max_steps: int,
    native_target: np.ndarray,
    fixed_target: np.ndarray,
    success_threshold: float,
    replay_actions: Optional[Sequence[np.ndarray]] = None,
    reference_trajectory: Optional[Sequence[np.ndarray]] = None,
) -> dict[str, Any]:
  """Run a policy or replay actions and retain independent success signals."""
  actions = []
  trajectory = [_mechanism_position(environment)]
  native_distances = [_distance(trajectory[-1], native_target)]
  fixed_distances = [_distance(trajectory[-1], fixed_target)]
  native_info_success = 0.0
  positive_reward_success = 0.0
  action_clip_count = 0
  native_rewards = []
  wrapper_rewards = []

  if replay_actions is None and policy is None:
    raise ValueError('A rollout needs either a policy or replay actions.')
  horizon = (min(max_steps, len(replay_actions))
             if replay_actions is not None else max_steps)
  for step_index in range(horizon):
    if replay_actions is None:
      action = _as_float_array(
          policy.get_action(_native_observation(environment, parent)))
    else:
      action = _as_float_array(replay_actions[step_index])
    clipped = np.clip(
        action, environment.action_space.low, environment.action_space.high)
    action_clip_count += int(np.max(np.abs(action - clipped)) > 1e-7)
    action = _as_float_array(clipped)
    actions.append(action)

    _, reward, done, info = _step_result(environment, action)
    wrapper_rewards.append(reward)
    if 'success' in info:
      step_native_reward = reward
      step_native_success = float(info['success'])
    else:
      step_native_reward, step_native_success = _evaluate_native(
          environment, parent, action)
    native_rewards.append(step_native_reward)
    native_info_success = max(native_info_success, step_native_success)
    positive_reward_success = max(
        positive_reward_success, float(reward > 0.0))

    mechanism = _mechanism_position(environment)
    trajectory.append(mechanism)
    native_distances.append(_distance(mechanism, native_target))
    fixed_distances.append(_distance(mechanism, fixed_target))
    if done:
      break

  trajectory_error = float('nan')
  if reference_trajectory is not None:
    count = min(len(trajectory), len(reference_trajectory))
    if count:
      trajectory_error = max(float(np.max(np.abs(
          trajectory[index] - reference_trajectory[index])))
                             for index in range(count))

  return {
      'steps': len(actions),
      'native_info_success': native_info_success,
      'positive_reward_success': positive_reward_success,
      'native_target_success': float(
          min(native_distances) < success_threshold),
      'fixed_target_success': float(
          min(fixed_distances) < success_threshold),
      'initial_native_distance': native_distances[0],
      'minimum_native_distance': min(native_distances),
      'final_native_distance': native_distances[-1],
      'initial_fixed_distance': fixed_distances[0],
      'minimum_fixed_distance': min(fixed_distances),
      'final_fixed_distance': fixed_distances[-1],
      'action_clip_fraction': action_clip_count / max(len(actions), 1),
      'native_reward_max': _maximum(native_rewards),
      'wrapper_reward_max': _maximum(wrapper_rewards),
      'trajectory_linf_error_vs_native': trajectory_error,
      '_actions': actions,
      '_trajectory': trajectory,
  }


def _strip_private(result: dict[str, Any]) -> dict[str, Any]:
  return {key: value for key, value in result.items()
          if not key.startswith('_')}


def _summarize_conditions(
    episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
  summary = {}
  for condition in CONDITIONS:
    rows = [episode['conditions'][condition] for episode in episodes]
    keys = sorted({key for row in rows for key, value in row.items()
                   if isinstance(value, (int, float))})
    summary[condition] = {
        key + '_mean': _mean(row[key] for row in rows if key in row)
        for key in keys
    }
    for key in (
        'minimum_native_distance',
        'minimum_fixed_distance',
        'trajectory_linf_error_vs_native',
    ):
      values = [row[key] for row in rows if key in row]
      if values:
        summary[condition][key + '_max'] = _maximum(values)
        summary[condition][key + '_min'] = _minimum(values)
  return summary


def classify_task(
    summary: dict[str, Any],
    *,
    expert_success_min: float,
    fixed_success_max: float,
    trajectory_tolerance: float,
    target_tolerance: float,
    success_threshold: float,
) -> dict[str, Any]:
  """Classify which validity boundary failed using pre-registered gates."""
  native = summary['conditions']['native_official']
  wrapper_native = summary['conditions']['wrapper_native_target_policy']
  wrapper_native_replay = summary[
      'conditions']['wrapper_native_target_replay']
  wrapper_fixed = summary['conditions']['wrapper_fixed_target_policy']
  wrapper_fixed_replay = summary['conditions']['wrapper_fixed_target_replay']
  pairing = summary['pairing']

  native_control_pass = (
      native['native_info_success_mean'] >= expert_success_min
      and native['native_target_success_mean'] >= expert_success_min)
  wrapper_native_control_pass = (
      wrapper_native['positive_reward_success_mean'] >= expert_success_min
      and wrapper_native['native_info_success_mean'] >= expert_success_min
      and wrapper_native['native_target_success_mean'] >= expert_success_min
      and wrapper_native_replay['positive_reward_success_mean'] >=
      expert_success_min
      and wrapper_native_replay['native_info_success_mean'] >=
      expert_success_min
      and wrapper_native_replay['native_target_success_mean'] >=
      expert_success_min
      and pairing['native_target_pair_linf_error_max'] <= target_tolerance
      and pairing['rand_vec_pair_linf_error_max'] <= target_tolerance
      and pairing['initial_mechanism_pair_linf_error_max'] <=
      trajectory_tolerance
      and wrapper_native_replay[
          'trajectory_linf_error_vs_native_max'] <= trajectory_tolerance)
  fixed_target_valid = (
      wrapper_fixed['positive_reward_success_mean'] >= expert_success_min
      and wrapper_fixed_replay['fixed_target_success_mean'] >=
      expert_success_min)
  fixed_replay_reaches_native_endpoint = (
      wrapper_fixed_replay['native_target_success_mean'] >=
      expert_success_min
      and wrapper_fixed_replay[
          'trajectory_linf_error_vs_native_max'] <= trajectory_tolerance)
  meaningful_target_mismatch = (
      pairing['fixed_to_native_target_distance_mean'] > success_threshold)
  fixed_global_target_misaligned = (
      native_control_pass
      and wrapper_native_control_pass
      and wrapper_fixed['positive_reward_success_mean'] <= fixed_success_max
      and wrapper_fixed_replay['fixed_target_success_mean'] <=
      fixed_success_max
      and fixed_replay_reaches_native_endpoint
      and meaningful_target_mismatch)

  if not native_control_pass:
    decision = 'native_positive_control_failed'
  elif not wrapper_native_control_pass:
    decision = 'custom_wrapper_invalid'
  elif fixed_global_target_misaligned:
    decision = 'fixed_global_target_misaligned'
  elif fixed_target_valid:
    decision = 'fixed_target_valid'
  else:
    decision = 'inconclusive'

  return {
      'decision': decision,
      'decision_code': DECISION_CODES[decision],
      'native_control_pass': native_control_pass,
      'wrapper_native_control_pass': wrapper_native_control_pass,
      'fixed_target_valid': fixed_target_valid,
      'fixed_global_target_misaligned': fixed_global_target_misaligned,
      'fixed_replay_reaches_native_endpoint':
          fixed_replay_reaches_native_endpoint,
      'meaningful_target_mismatch': meaningful_target_mismatch,
  }


def summarize_task(
    env_name: str,
    episodes: Sequence[dict[str, Any]],
    *,
    expert_success_min: float,
    fixed_success_max: float,
    trajectory_tolerance: float,
    target_tolerance: float,
) -> dict[str, Any]:
  metadata = goal_semantics.VALIDITY_TASKS[env_name]
  condition_summary = _summarize_conditions(episodes)
  pairing = {
      'native_target_pair_linf_error_max': _maximum(
          episode['pairing']['native_target_pair_linf_error']
          for episode in episodes),
      'initial_mechanism_pair_linf_error_max': _maximum(
          episode['pairing']['initial_mechanism_pair_linf_error']
          for episode in episodes),
      'rand_vec_pair_linf_error_max': _maximum(
          episode['pairing']['rand_vec_pair_linf_error']
          for episode in episodes),
      'fixed_to_native_target_distance_mean': _mean(
          episode['pairing']['fixed_to_native_target_distance']
          for episode in episodes),
      'fixed_to_native_target_distance_min': _minimum(
          episode['pairing']['fixed_to_native_target_distance']
          for episode in episodes),
      'fixed_to_native_target_distance_max': _maximum(
          episode['pairing']['fixed_to_native_target_distance']
          for episode in episodes),
  }
  summary = {
      'env_name': env_name,
      'episodes': len(episodes),
      'success_threshold': float(metadata['success_threshold']),
      'conditions': condition_summary,
      'pairing': pairing,
  }
  summary['classification'] = classify_task(
      summary,
      expert_success_min=expert_success_min,
      fixed_success_max=fixed_success_max,
      trajectory_tolerance=trajectory_tolerance,
      target_tolerance=target_tolerance,
      success_threshold=float(metadata['success_threshold']))
  return summary


def _make_benchmark(native_env_name: str, seed: int):
  import metaworld  # pylint: disable=import-outside-toplevel
  np.random.seed(seed)
  try:
    return metaworld.ML1(native_env_name, seed=seed)
  except TypeError:
    np.random.seed(seed)
    return metaworld.ML1(native_env_name)


def _make_native_environment(benchmark, native_env_name: str):
  environment = benchmark.train_classes[native_env_name]()
  return environment, type(environment)


def _make_custom_environment(env_name: str, fixed_goal):
  import env_utils  # pylint: disable=import-outside-toplevel
  environment, obs_dim, _ = env_utils.load(
      env_name, fixed_start_end=fixed_goal)
  if obs_dim != env_utils.STATE_DIM_UNIFIED:
    raise RuntimeError(
        f'Expected unified obs_dim={env_utils.STATE_DIM_UNIFIED}, got '
        f'{obs_dim}.')
  return environment, _native_parent(environment)


def _select_tasks(benchmark, native_env_name: str, episodes: int,
                  seed: int):
  tasks = [task for task in benchmark.train_tasks
           if getattr(task, 'env_name', native_env_name) == native_env_name]
  if not tasks:
    raise RuntimeError(f'ML1 returned no train tasks for {native_env_name}.')
  rng = np.random.RandomState(seed)
  order = rng.permutation(len(tasks))
  return [tasks[order[index % len(order)]] for index in range(episodes)]


def _close_all(environments):
  for environment in environments:
    try:
      environment.close()
    except Exception:
      pass


def audit_task(
    env_name: str,
    *,
    seed: int,
    episodes: int,
    max_steps: int,
    expert_success_min: float,
    fixed_success_max: float,
    trajectory_tolerance: float,
    target_tolerance: float,
) -> dict[str, Any]:
  """Run all paired controls for one task."""
  from metaworld import policies  # pylint: disable=import-outside-toplevel

  metadata = goal_semantics.VALIDITY_TASKS[env_name]
  native_env_name = metadata['native_env_name']
  fixed_goal = _as_float_array(metadata['fixed_goal'])
  benchmark = _make_benchmark(native_env_name, seed)
  tasks = _select_tasks(benchmark, native_env_name, episodes, seed)

  native_env, native_parent = _make_native_environment(
      benchmark, native_env_name)
  wrapper_native, wrapper_native_parent = _make_custom_environment(
      env_name, None)
  wrapper_fixed, wrapper_fixed_parent = _make_custom_environment(
      env_name, fixed_goal)
  environments = (native_env, wrapper_native, wrapper_fixed)
  policy_type = getattr(policies, metadata['policy_class'])
  episode_results = []

  try:
    for episode_index, task in enumerate(tasks):
      _set_task_and_reset(native_env, task)
      native_target = _as_float_array(native_env._target_pos)
      native_rand_vec = _last_rand_vec(native_env)
      native_initial_mechanism = _mechanism_position(native_env)
      native_nearest_site = _nearest_site(native_env, native_target)
      native_policy_result = _rollout(
          native_env,
          native_parent,
          policy=policy_type(),
          max_steps=max_steps,
          native_target=native_target,
          fixed_target=fixed_goal,
          success_threshold=float(metadata['success_threshold']))
      native_actions = native_policy_result['_actions']
      native_trajectory = native_policy_result['_trajectory']

      _set_task_and_reset(wrapper_native, task)
      wrapper_native_target = _as_float_array(wrapper_native._target_pos)
      wrapper_native_rand_vec = _last_rand_vec(wrapper_native)
      wrapper_native_initial = _mechanism_position(wrapper_native)
      wrapper_native_policy_result = _rollout(
          wrapper_native,
          wrapper_native_parent,
          policy=policy_type(),
          max_steps=max_steps,
          native_target=native_target,
          fixed_target=fixed_goal,
          success_threshold=float(metadata['success_threshold']))

      _set_task_and_reset(wrapper_fixed, task)
      wrapper_fixed_rand_vec = _last_rand_vec(wrapper_fixed)
      wrapper_fixed_initial = _mechanism_position(wrapper_fixed)
      wrapper_fixed_nearest_native_site = _nearest_site(
          wrapper_fixed, native_target)
      wrapper_fixed_nearest_fixed_site = _nearest_site(
          wrapper_fixed, fixed_goal)
      wrapper_fixed_policy_result = _rollout(
          wrapper_fixed,
          wrapper_fixed_parent,
          policy=policy_type(),
          max_steps=max_steps,
          native_target=native_target,
          fixed_target=fixed_goal,
          success_threshold=float(metadata['success_threshold']))

      _set_task_and_reset(wrapper_native, task)
      wrapper_native_replay_result = _rollout(
          wrapper_native,
          wrapper_native_parent,
          policy=None,
          max_steps=max_steps,
          native_target=native_target,
          fixed_target=fixed_goal,
          success_threshold=float(metadata['success_threshold']),
          replay_actions=native_actions,
          reference_trajectory=native_trajectory)

      _set_task_and_reset(wrapper_fixed, task)
      wrapper_fixed_replay_result = _rollout(
          wrapper_fixed,
          wrapper_fixed_parent,
          policy=None,
          max_steps=max_steps,
          native_target=native_target,
          fixed_target=fixed_goal,
          success_threshold=float(metadata['success_threshold']),
          replay_actions=native_actions,
          reference_trajectory=native_trajectory)

      episode_results.append({
          'episode': episode_index,
          'pairing': {
              'native_target_pair_linf_error': float(np.max(np.abs(
                  native_target - wrapper_native_target))),
              'initial_mechanism_pair_linf_error': max(
                  float(np.max(np.abs(
                      native_initial_mechanism - wrapper_native_initial))),
                  float(np.max(np.abs(
                      native_initial_mechanism - wrapper_fixed_initial)))),
              'rand_vec_pair_linf_error': max(
                  float(np.max(np.abs(
                      native_rand_vec - wrapper_native_rand_vec))),
                  float(np.max(np.abs(
                      native_rand_vec - wrapper_fixed_rand_vec)))),
              'fixed_to_native_target_distance': _distance(
                  fixed_goal, native_target),
          },
          'native_target': native_target.tolist(),
          'fixed_target': fixed_goal.tolist(),
          'native_rand_vec': native_rand_vec.tolist(),
          'sites': {
              'native_nearest_to_native_target': native_nearest_site,
              'fixed_wrapper_nearest_to_native_target':
                  wrapper_fixed_nearest_native_site,
              'fixed_wrapper_nearest_to_fixed_target':
                  wrapper_fixed_nearest_fixed_site,
          },
          'conditions': {
              'native_official': _strip_private(native_policy_result),
              'wrapper_native_target_policy': _strip_private(
                  wrapper_native_policy_result),
              'wrapper_fixed_target_policy': _strip_private(
                  wrapper_fixed_policy_result),
              'wrapper_native_target_replay': _strip_private(
                  wrapper_native_replay_result),
              'wrapper_fixed_target_replay': _strip_private(
                  wrapper_fixed_replay_result),
          },
      })
  finally:
    _close_all(environments)

  return {
      'env_name': env_name,
      'native_env_name': native_env_name,
      'seed': seed,
      'episodes': episode_results,
      'summary': summarize_task(
          env_name,
          episode_results,
          expert_success_min=expert_success_min,
          fixed_success_max=fixed_success_max,
          trajectory_tolerance=trajectory_tolerance,
          target_tolerance=target_tolerance),
  }


def _wandb_metrics(payload: dict[str, Any]) -> dict[str, float]:
  metrics = {
      'positive_control/all_current_wrappers_valid': float(
          payload['all_current_wrappers_valid']),
      'positive_control/any_fixed_target_misaligned': float(
          payload['any_fixed_target_misaligned']),
  }
  for result in payload['results']:
    task = ('task5' if result['env_name'] == 'sawyer_handle_press_side'
            else 'task8')
    summary = result['summary']
    classification = summary['classification']
    prefix = f'positive_control/{task}/'
    metrics.update({
        prefix + 'decision_code': float(classification['decision_code']),
        prefix + 'native_control_pass': float(
            classification['native_control_pass']),
        prefix + 'wrapper_native_control_pass': float(
            classification['wrapper_native_control_pass']),
        prefix + 'fixed_target_valid': float(
            classification['fixed_target_valid']),
        prefix + 'fixed_global_target_misaligned': float(
            classification['fixed_global_target_misaligned']),
    })
    for key, value in summary['pairing'].items():
      metrics[prefix + 'pairing/' + key] = float(value)
    for condition, values in summary['conditions'].items():
      for key, value in values.items():
        metrics[prefix + condition + '/' + key] = float(value)
  return metrics


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--task', action='append', choices=tuple(goal_semantics.VALIDITY_TASKS),
      help='Task to validate; repeat for both. Defaults to both tasks.')
  parser.add_argument('--seed', type=int, default=5)
  parser.add_argument('--episodes', type=int, default=50)
  parser.add_argument('--max-steps', type=int, default=150)
  parser.add_argument('--expert-success-min', type=float, default=0.80)
  parser.add_argument('--fixed-success-max', type=float, default=0.20)
  parser.add_argument('--trajectory-tolerance', type=float, default=1e-5)
  parser.add_argument('--target-tolerance', type=float, default=1e-6)
  parser.add_argument(
      '--output',
      default='logs/goal_validity/positive_controls_seed5.json')
  parser.add_argument('--wandb-project', default='')
  parser.add_argument(
      '--wandb-group', default='GOAL-WRAPPER-POSITIVE-CONTROLS-V2')
  parser.add_argument(
      '--strict-current-wrapper', action='store_true',
      help='Exit nonzero unless the historical fixed-target wrapper validates.')
  args = parser.parse_args()

  tasks = args.task or list(goal_semantics.VALIDITY_TASKS)
  results = [audit_task(
      task,
      seed=args.seed,
      episodes=args.episodes,
      max_steps=args.max_steps,
      expert_success_min=args.expert_success_min,
      fixed_success_max=args.fixed_success_max,
      trajectory_tolerance=args.trajectory_tolerance,
      target_tolerance=args.target_tolerance) for task in tasks]
  payload = {
      'audit_version': 2,
      'seed': args.seed,
      'episodes_per_task': args.episodes,
      'max_steps': args.max_steps,
      'gates': {
          'expert_success_min': args.expert_success_min,
          'fixed_success_max': args.fixed_success_max,
          'trajectory_tolerance': args.trajectory_tolerance,
          'target_tolerance': args.target_tolerance,
      },
      'results': results,
      'all_current_wrappers_valid': all(
          result['summary']['classification']['fixed_target_valid']
          for result in results),
      'any_fixed_target_misaligned': any(
          result['summary']['classification'][
              'fixed_global_target_misaligned']
          for result in results),
  }

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
  print(json.dumps({
      'seed': payload['seed'],
      'all_current_wrappers_valid': payload['all_current_wrappers_valid'],
      'any_fixed_target_misaligned': payload[
          'any_fixed_target_misaligned'],
      'decisions': {
          result['env_name']:
              result['summary']['classification']['decision']
          for result in results
      },
      'output': str(output),
  }, indent=2))

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
            'audit_version': 2,
            'seed': args.seed,
            'episodes': args.episodes,
            'max_steps': args.max_steps,
            'expert_success_min': args.expert_success_min,
            'fixed_success_max': args.fixed_success_max,
            'trajectory_tolerance': args.trajectory_tolerance,
            'target_tolerance': args.target_tolerance,
            'git_commit': os.environ.get('GIT_COMMIT', 'unknown'),
        },
        name=f'goal_positive_controls_s{args.seed}')
    run.log(_wandb_metrics(payload))
    run.summary['positive_control/decisions'] = {
        result['env_name']:
            result['summary']['classification']['decision']
        for result in results
    }
    run.finish()

  if args.strict_current_wrapper and not payload['all_current_wrappers_valid']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
