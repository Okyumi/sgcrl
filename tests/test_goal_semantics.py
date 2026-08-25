#!/usr/bin/env python3
"""Dependency-light tests for the Task-5/Task-8 goal validity sequence."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import counterfactual_outcomes
from contrastive import goal_semantics
from experiment_configs_goal_semantics import build_configs


def test_goal_slices():
  assert goal_semantics.resolve_goal_slice(
      'full_state', 'sawyer_handle_press_side') == (0, -1)
  assert goal_semantics.resolve_goal_slice(
      'success_mechanism', 'sawyer_handle_press_side') == (4, 7)
  assert goal_semantics.resolve_goal_slice(
      'success_mechanism', 'sawyer_window_close') == (4, 7)
  assert goal_semantics.resolve_goal_slice(
      'native_success_axis', 'sawyer_handle_press_side') == (6, 7)
  assert goal_semantics.resolve_goal_slice(
      'native_success_axis', 'sawyer_window_close') == (4, 5)


def test_native_success_axis_matches_official_task_coordinates():
  position = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
  target = np.asarray([0.0, 0.0, 0.35], dtype=np.float32)
  handle_distance = goal_semantics.success_axis_distance(
      position, target, 'sawyer_handle_press_side')
  window_distance = goal_semantics.success_axis_distance(
      position, target, 'sawyer_window_close')
  assert abs(handle_distance - 0.05) < 1e-6
  assert abs(window_distance - 0.10) < 1e-6
  full_state = np.zeros(11, dtype=np.float32)
  full_state[4:7] = position
  assert np.allclose(
      goal_semantics.success_axis_from_state(
          full_state, 'sawyer_handle_press_side'), [0.3])
  assert np.allclose(
      goal_semantics.success_axis_from_goal(
          target, 'sawyer_window_close'), [0.0])


def test_mechanism_distance_for_both_contracts():
  state = np.zeros(11, dtype=np.float32)
  state[4:7] = [1.0, 2.0, 3.0]
  full_goal = np.zeros(11, dtype=np.float32)
  full_goal[4:7] = [1.0, 2.0, 4.0]
  compact_goal = np.asarray([1.0, 2.0, 4.0], dtype=np.float32)
  _, full_distance = counterfactual_outcomes.goal_distances(
      np.concatenate([state, full_goal]), 11)
  compact_full, compact_distance = counterfactual_outcomes.goal_distances(
      np.concatenate([state, compact_goal]), 11)
  assert abs(full_distance - 1.0) < 1e-6
  assert abs(compact_distance - 1.0) < 1e-6
  assert abs(compact_full - compact_distance) < 1e-6


def test_goal_contract_metrics_expose_nuisance_mismatch():
  state = np.zeros(11, dtype=np.float32)
  state[:3] = [0.1, 0.2, 0.3]
  goal = np.zeros(11, dtype=np.float32)
  goal[:3] = [0.1, 0.2, 0.4]
  goal[3] = 0.4
  goal[4:7] = [0.5, 0.6, 0.7]
  metrics = goal_semantics.goal_contract_metrics(
      np.concatenate([state, goal]), 11, goal[4:7])
  assert metrics['target_linf_error'] == 0.0
  assert metrics['hand_goal_distance'] > 0.0
  assert abs(metrics['gripper_goal_error'] - 0.4) < 1e-6


def test_config_matrix_and_hot_path():
  configs = build_configs()
  assert len(configs) == 8
  assert {config['seed'] for config in configs} == {5, 6}
  assert {config['goal_conditioning_mode'] for config in configs} == {
      'full_state', 'success_mechanism'}
  for config in configs:
    assert config['steps_per_task'] == 1_000_000
    assert config['counterfactual_rank_interval_steps'] == 0
    assert config['counterfactual_oracle_interval_steps'] == 0
    assert config['action_landscape_diagnostic_interval_steps'] == 0
    assert config['shortcut_diagnostic_interval'] == 0
    assert config['profile_runtime'] is True


def test_promotion_is_guarded():
  env = dict(os.environ)
  env.pop('GOAL_VALIDITY_PROMOTED', None)
  result = subprocess.run(
      [sys.executable, 'experiment_configs_goal_semantics_promotion.py',
       '--total'], env=env, capture_output=True, text=True, check=False)
  assert result.returncode != 0
  assert 'GOAL_VALIDITY_PROMOTED=true' in result.stderr


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'goal-semantics dependency-light tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
