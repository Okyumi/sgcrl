#!/usr/bin/env python3
"""Dependency-light checks for corrected Task-5/Task-8 full-state goals."""
from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _array_dict(name):
  tree = ast.parse((REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8'))
  for node in tree.body:
    if (isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)):
      result = {}
      for key, value in zip(node.value.keys, node.value.values):
        result[ast.literal_eval(key)] = ast.literal_eval(value.args[0])
      return result
  raise AssertionError(f'No {name} assignment found.')


def test_captured_goals_are_complete_and_successful():
  targets = _array_dict('TASK58_FIXED_MECHANISM_TARGETS')
  goals = _array_dict('TASK58_REACHABLE_SUCCESS_GOALS')
  assert set(goals) == set(targets) == {
      'sawyer_handle_press_side', 'sawyer_window_close'}
  assert all(len(goal) == 7 for goal in goals.values())
  assert abs(goals['sawyer_handle_press_side'][6] - targets[
      'sawyer_handle_press_side'][2]) <= 0.02
  assert abs(goals['sawyer_window_close'][4] - targets[
      'sawyer_window_close'][0]) <= 0.05


def test_captured_goals_remove_unreachable_gripper_value():
  goals = _array_dict('TASK58_REACHABLE_SUCCESS_GOALS')
  for goal in goals.values():
    assert abs(goal[3] - 0.29668) < 1e-4
    assert abs(goal[3] - 0.4) > 0.10


def test_legacy_mode_keeps_historical_synthetic_goal():
  source = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  selector = source.split(
      'def _task58_reachable_success_goal(', 1)[1].split(
          'def _set_sawyer_success_mode(', 1)[0]
  assert "sawyer_success_mode == 'legacy_distance'" in selector
  assert 'return None' in selector
  for class_name, next_class in (
      ('SawyerHandlePressSide', 'SawyerPush'),
      ('SawyerWindowClose', 'SawyerPegUnplugSide'),
  ):
    wrapper = source.split(f'class {class_name}(', 1)[1].split(
        f'class {next_class}(', 1)[0]
    assert 'if self._fixed_goal_state is not None:' in wrapper
    assert '[0.4]' in wrapper


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Task-5/Task-8 reachable-goal tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
