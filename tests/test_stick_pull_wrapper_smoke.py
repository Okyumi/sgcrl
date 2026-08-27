#!/usr/bin/env python3
"""Dependency-light checks for the Stick-Pull simulator audit."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_test_stick_pull_corrected_wrapper import classify_summary
from scripts.smoke_test_stick_pull_corrected_wrapper import insertion_metrics


def _summary():
  return {
      'reported_training_horizon': 150,
      'reset_success_count': 0,
      'reward_mismatch_steps': 0,
      'info_mismatch_steps': 0,
      'success_rate': 1.0,
      'captured_successful_state': [0.0] * 11,
  }


def test_insertion_margin_matches_all_three_official_gates():
  inside = insertion_metrics(
      [0.41, 0.54, 0.02], [0.45, 0.56, 0.07])
  assert inside['inserted']
  assert inside['signed_insertion_margin'] >= 0.0
  outside = insertion_metrics(
      [0.41, 0.54, 0.02], [0.45, 0.59, 0.07])
  assert not outside['inserted']
  assert outside['signed_insertion_margin'] < 0.0


def test_clean_summary_passes():
  result = classify_summary(_summary())
  assert result['passed']
  assert result['failed_gates'] == []


def test_reward_failure_and_missing_goal_fail_independently():
  summary = _summary()
  summary['reward_mismatch_steps'] = 1
  summary['captured_successful_state'] = None
  result = classify_summary(summary)
  assert not result['passed']
  assert 'reward_matches_official_predicate' in result['failed_gates']
  assert 'captured_successful_goal_exists' in result['failed_gates']


def test_launcher_is_evaluation_only():
  source = (REPO_ROOT / 'DRAFT_stick_pull_wrapper_smoke.sh').read_text(
      encoding='utf-8')
  assert 'scripts/smoke_test_stick_pull_corrected_wrapper.py' in source
  assert '--seeds 5 6 7' in source
  assert '--episodes 5' in source
  assert '--training-horizon 150' in source
  assert 'run_continual_contrastive.py' not in source


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Stick-Pull wrapper smoke tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
