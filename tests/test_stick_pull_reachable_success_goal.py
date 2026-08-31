#!/usr/bin/env python3
"""Dependency-light checks for the corrected Stick-Pull goal contract."""
from __future__ import annotations

import ast
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _array(name):
  tree = ast.parse((REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8'))
  for node in tree.body:
    if (isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in node.targets)):
      return ast.literal_eval(node.value.args[0])
  raise AssertionError(f'No {name} assignment found.')


def test_captured_goal_is_complete_and_robustly_successful():
  target = _array('STICK_PULL_FIXED_TARGET')
  goal = _array('STICK_PULL_REACHABLE_SUCCESS_GOAL')
  assert len(target) == 3
  assert len(goal) == 11
  handle_distance = math.dist(goal[7:10], target)
  assert handle_distance <= 0.12
  assert goal[10] >= 0.0
  assert min(0.12 - handle_distance, goal[10]) > 0.007


def test_corrected_wrapper_exposes_insertion_margin_and_captured_goal():
  source = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  wrapper = source.split('class SawyerStickPull(', 1)[1].split(
      'class SawyerHandlePressSide(', 1)[0]
  assert 'fixed_goal_state=None' in wrapper
  assert 'sawyer_success.stick_pull_insertion_margin(' in wrapper
  assert 'goal = self._fixed_goal_state' in wrapper


def test_legacy_mode_retains_the_historical_ten_coordinate_contract():
  source = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  selector = source.split(
      'def _stick_pull_reachable_success_goal(', 1)[1].split(
          'def _set_sawyer_success_mode(', 1)[0]
  assert "sawyer_success_mode == 'legacy_distance'" in selector
  assert 'return None' in selector
  wrapper = source.split('class SawyerStickPull(', 1)[1].split(
      'class SawyerHandlePressSide(', 1)[0]
  assert 'Legacy goal mirrors historical state semantics and padding.' in wrapper


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Stick-Pull reachable-goal tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
