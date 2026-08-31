#!/usr/bin/env python3
"""Dependency-light checks for the Stick-Pull simulator audit."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_test_stick_pull_corrected_wrapper import classify_summary
from scripts.smoke_test_stick_pull_corrected_wrapper import full_goal_errors
from scripts.smoke_test_stick_pull_corrected_wrapper import insertion_metrics


def _summary():
  return {
      'reported_training_horizon': 150,
      'reset_success_count': 0,
      'reward_mismatch_steps': 0,
      'info_mismatch_steps': 0,
      'state_insertion_margin_error_max': 0.0,
      'exposed_goal_linf_error_max': 0.0,
      'success_rate': 1.0,
      'captured_successful_state': [0.0] * 11,
  }


def test_full_goal_errors_reports_component_distances():
  observation = np.array([
      1.0, 2.0, 3.0, 0.5,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
      0.1,
      1.1, 2.1, 3.1, 0.6,
      4.2, 5.2, 6.2,
      7.3, 8.3, 9.3,
      0.2,
  ], dtype=np.float32)
  metrics = full_goal_errors(observation, obs_dim=11)
  assert np.isclose(metrics['full_goal_linf_error'], 0.3)
  assert metrics['hand_goal_l2_error'] > 0.0
  assert metrics['stick_com_goal_l2_error'] > 0.0
  assert metrics['handle_goal_l2_error'] > 0.0
  assert np.isclose(metrics['insertion_margin_goal_abs_error'], 0.1)
  assert not metrics['bitwise_equal']


def test_summary_with_goal_revisit_metrics_still_passes():
  summary = _summary()
  summary.update({
      'successful_state_samples': 12,
      'full_goal_visits_within_1e-2': 4,
      'full_goal_linf_error_at_success_min': 0.01,
      'full_goal_linf_error_at_success_mean': 0.15,
      'full_goal_linf_error_at_success_max': 0.4,
      'minimum_any_state_full_goal_linf_error': 0.005,
  })
  result = classify_summary(summary)
  assert result['passed']


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
