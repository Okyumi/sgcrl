#!/usr/bin/env python3
"""Dependency-light checks for the direct Task-5/Task-8 reward fix."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import sawyer_success


class _Environment:

  def __init__(self, mode='corrected'):
    self._sawyer_success_mode = mode


def test_task5_uses_z_only():
  result = sawyer_success.task_axis_sparse_reward(
      _Environment(), [9.0, -4.0, 0.071], [0.0, 0.0, 0.070],
      axis=2, threshold=0.02)
  reward, info = result
  assert reward == 1.0
  assert info['success_axis_index'] == 2
  assert info['success_axis_distance'] < 0.02


def test_task8_uses_x_only():
  result = sawyer_success.task_axis_sparse_reward(
      _Environment(), [0.04, 8.0, -3.0], [0.0, 0.0, 0.0],
      axis=0, threshold=0.05)
  reward, info = result
  assert reward == 1.0
  assert info['success_axis_index'] == 0


def test_axis_threshold_is_inclusive_and_failure_is_zero():
  reward, _ = sawyer_success.task_axis_sparse_reward(
      _Environment(), [0.05], [0.0], axis=0, threshold=0.05)
  assert reward == 1.0
  reward, _ = sawyer_success.task_axis_sparse_reward(
      _Environment(), [0.051], [0.0], axis=0, threshold=0.05)
  assert reward == 0.0


def test_other_modes_keep_their_existing_paths():
  for mode in ('legacy_distance', 'native_info'):
    assert sawyer_success.task_axis_sparse_reward(
        _Environment(mode), [0.0], [0.0], axis=0,
        threshold=0.05) is None


def test_wrapper_wiring_is_task_specific():
  source = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  assert "self, handle_pos, self._goal, axis=2, threshold=0.02" in source
  assert "self, handle_pos, self._goal, axis=0, threshold=0.05" in source
  assert 'task_axis success is defined only for' in source


def test_corrected_is_the_default_and_has_separate_checkpoint_identity():
  source = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  env_source = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  utils_source = (REPO_ROOT / 'contrastive' / 'utils.py').read_text(
      encoding='utf-8')
  assert "'sawyer_success_mode', 'corrected'" in source
  assert "('corrected', 'legacy_distance', 'task_axis', 'native_info')" in source
  assert "sawyer_success_mode='corrected'" in env_source
  assert "sawyer_success_mode='corrected'" in utils_source
  assert "config_key += f'_success_{sawyer_success_mode}'" in source


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Task-5/Task-8 axis-reward tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
