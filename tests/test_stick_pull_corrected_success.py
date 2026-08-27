#!/usr/bin/env python3
"""Dependency-light checks for corrected Stick-Pull success semantics."""
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


def _reward(handle, stick_end, mode='corrected'):
  result = sawyer_success.stick_pull_sparse_reward(
      _Environment(mode), handle, [0.41, 0.54, 0.02], stick_end)
  if result is None:
    return None
  return result


def test_target_distance_alone_is_not_success():
  reward, info = _reward(
      [0.41, 0.54, 0.02],
      # Behind and far above the handle: direct-push false positive.
      [0.30, 0.54, 0.20])
  assert reward == 0.0
  assert info['handle_target_distance'] == 0.0
  assert info['stick_is_inserted'] == 0.0


def test_target_and_inserted_stick_are_success():
  reward, info = _reward(
      [0.41, 0.54, 0.02],
      [0.45, 0.56, 0.07])
  assert reward == 1.0
  assert info['success'] == 1.0
  assert info['stick_is_inserted'] == 1.0


def test_insertion_tolerances_are_inclusive():
  reward, _ = _reward(
      [0.41, 0.54, 0.02],
      [0.41, 0.58, 0.08])
  assert reward == 1.0


def test_handle_must_also_reach_target():
  reward, info = _reward(
      [0.20, 0.54, 0.02],
      [0.25, 0.54, 0.02])
  assert reward == 0.0
  assert info['stick_is_inserted'] == 1.0
  assert info['handle_target_distance'] > 0.12


def test_legacy_and_native_paths_remain_separate():
  for mode in ('legacy_distance', 'native_info', 'task_axis'):
    assert _reward(
        [0.41, 0.54, 0.02], [0.45, 0.54, 0.02], mode=mode) is None


def test_original_wrapper_queries_the_stick_endpoint():
  source = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  wrapper = source.split('class SawyerStickPull(', 1)[1].split(
      'class SawyerHandlePressSide(', 1)[0]
  assert "self._get_site_pos('stick_end')" in wrapper
  assert 'sawyer_success.stick_pull_sparse_reward(' in wrapper


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'corrected Stick-Pull success tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
