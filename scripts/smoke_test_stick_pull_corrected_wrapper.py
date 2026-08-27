#!/usr/bin/env python3
"""Evaluation-only Stick-Pull success and reachable-goal audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import sawyer_success


FIXED_TARGET = np.array([0.41, 0.54, 0.02], dtype=np.float32)


def insertion_metrics(handle, stick_end):
  """Return MetaWorld's insertion predicate and a signed scalar margin."""
  handle = np.asarray(handle, dtype=np.float32)
  stick_end = np.asarray(stick_end, dtype=np.float32)
  margins = np.array([
      stick_end[0] - handle[0],
      0.040 - abs(stick_end[1] - handle[1]),
      0.060 - abs(stick_end[2] - handle[2]),
  ], dtype=np.float32)
  return {
      'inserted': bool(np.all(margins >= 0.0)),
      # One continuous coordinate fits the wrapper's existing state[10]
      # padding slot and is >= 0 exactly when all insertion gates pass.
      'signed_insertion_margin': float(np.min(margins)),
      'axis_margins': margins.tolist(),
  }


def classify_summary(summary, *, expert_success_min=0.8):
  gates = {
      'reported_training_horizon_is_150':
          summary['reported_training_horizon'] == 150,
      'reset_is_not_already_successful':
          summary['reset_success_count'] == 0,
      'reward_matches_official_predicate':
          summary['reward_mismatch_steps'] == 0,
      'info_matches_official_predicate':
          summary['info_mismatch_steps'] == 0,
      'scripted_policy_solves_by_training_horizon':
          summary['success_rate'] >= expert_success_min,
      'captured_successful_goal_exists':
          summary['captured_successful_state'] is not None,
  }
  failed = [name for name, passed in gates.items() if not passed]
  return {'passed': not failed, 'failed_gates': failed, 'gates': gates}


def _reset(environment):
  result = environment.reset()
  if isinstance(result, tuple) and len(result) == 2:
    result = result[0]
  return np.asarray(result, dtype=np.float32)


def _step(environment, action):
  result = environment.step(action)
  if len(result) == 4:
    observation, reward, done, info = result
  elif len(result) == 5:
    observation, reward, terminated, truncated, info = result
    done = bool(terminated or truncated)
  else:
    raise RuntimeError(f'Unexpected transition length {len(result)}.')
  return (np.asarray(observation, dtype=np.float32), float(reward),
          bool(done), dict(info or {}))


def _native_parent(environment):
  for parent in type(environment).__mro__[1:]:
    if (parent.__module__.startswith('metaworld.')
        and hasattr(parent, '_get_obs')
        and hasattr(parent, 'evaluate_state')):
      return parent
  raise RuntimeError('No MetaWorld parent found for Stick-Pull wrapper.')


def _state_and_success(environment, observation):
  state10 = np.asarray(observation[:10], dtype=np.float32)
  handle = state10[7:10]
  stick_end = np.asarray(
      environment._get_site_pos('stick_end'), dtype=np.float32)
  insertion = insertion_metrics(handle, stick_end)
  handle_distance = float(np.linalg.norm(handle - FIXED_TARGET))
  success = bool(handle_distance <= 0.12 and insertion['inserted'])
  state11 = np.concatenate([
      state10,
      np.array([insertion['signed_insertion_margin']], dtype=np.float32),
  ])
  return state11, success, handle_distance, stick_end, insertion


def run_audit(*, seeds, episodes, training_horizon, expert_success_min):
  import env_utils  # pylint: disable=import-outside-toplevel
  from metaworld import policies  # pylint: disable=import-outside-toplevel

  total_episodes = 0
  successes = 0
  reset_success_count = 0
  reward_mismatch_steps = 0
  info_mismatch_steps = 0
  legacy_positive_steps = 0
  legacy_false_positive_steps = 0
  captured_successful_state = None
  captured_details = None
  reported_horizons = set()

  for seed in seeds:
    np.random.seed(seed)
    environment, obs_dim, reported_horizon = env_utils.load(
        'sawyer_stick_pull',
        fixed_start_end=FIXED_TARGET,
        sawyer_success_mode='corrected')
    if obs_dim != 11:
      raise RuntimeError(f'Expected unified obs_dim=11, got {obs_dim}.')
    reported_horizons.add(int(reported_horizon))
    parent = _native_parent(environment)
    try:
      for episode in range(episodes):
        observation = _reset(environment)
        _, reset_success, _, _, _ = _state_and_success(
            environment, observation)
        reset_success_count += int(reset_success)
        policy = policies.SawyerStickPullV2Policy()
        episode_success = False
        for step in range(1, training_horizon + 1):
          policy_observation = sawyer_success.native_observation(
              environment, parent)
          action = np.asarray(
              policy.get_action(policy_observation), dtype=np.float32)
          action = np.clip(
              action, environment.action_space.low,
              environment.action_space.high)
          observation, reward, done, info = _step(environment, action)
          (state11, expected_success, handle_distance, stick_end,
           insertion) = _state_and_success(environment, observation)
          reward_mismatch_steps += int(
              bool(reward > 0.0) != expected_success)
          info_mismatch_steps += int(
              bool(info.get('success', False)) != expected_success)
          legacy_positive = handle_distance <= 0.12
          legacy_positive_steps += int(legacy_positive)
          legacy_false_positive_steps += int(
              legacy_positive and not expected_success)
          if expected_success:
            episode_success = True
            if captured_successful_state is None:
              captured_successful_state = state11.tolist()
              captured_details = {
                  'seed': int(seed),
                  'episode': int(episode),
                  'step': int(step),
                  'handle_target_distance': handle_distance,
                  'stick_end': stick_end.tolist(),
                  **insertion,
              }
          if done:
            break
        successes += int(episode_success)
        total_episodes += 1
    finally:
      try:
        environment.close()
      except Exception:
        pass

  summary = {
      'env_name': 'sawyer_stick_pull',
      'fixed_target': FIXED_TARGET.tolist(),
      'episodes': total_episodes,
      'reported_training_horizon': (
          next(iter(reported_horizons))
          if len(reported_horizons) == 1 else -1),
      'reset_success_count': reset_success_count,
      'success_rate': successes / max(total_episodes, 1),
      'reward_mismatch_steps': reward_mismatch_steps,
      'info_mismatch_steps': info_mismatch_steps,
      'legacy_positive_steps': legacy_positive_steps,
      'legacy_false_positive_steps': legacy_false_positive_steps,
      'legacy_false_positive_fraction': (
          legacy_false_positive_steps / max(legacy_positive_steps, 1)),
      # Proposed corrected state/goal semantics:
      # hand(3), gripper(1), stick COM(3), handle(3), insertion margin(1).
      'captured_successful_state_semantics': (
          'hand_xyz, gripper, stick_com_xyz, handle_xyz, '
          'signed_insertion_margin'),
      'captured_successful_state': captured_successful_state,
      'captured_successful_state_details': captured_details,
  }
  summary['classification'] = classify_summary(
      summary, expert_success_min=expert_success_min)
  return summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seeds', type=int, nargs='+', default=(5, 6, 7))
  parser.add_argument('--episodes', type=int, default=5)
  parser.add_argument('--training-horizon', type=int, default=150)
  parser.add_argument('--expert-success-min', type=float, default=0.8)
  parser.add_argument(
      '--output', type=Path,
      default=Path('logs/wrapper_smoke/stick_pull_corrected_wrapper.json'))
  args = parser.parse_args()
  summary = run_audit(
      seeds=args.seeds,
      episodes=args.episodes,
      training_horizon=args.training_horizon,
      expert_success_min=args.expert_success_min)
  result = {
      'protocol': 'stick_pull_corrected_wrapper_smoke_v1',
      'passed': summary['classification']['passed'],
      'summary': summary,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  print(json.dumps({'output': str(args.output), **result}, indent=2,
                   sort_keys=True))
  if not result['passed']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
