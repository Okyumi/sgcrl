#!/usr/bin/env python3
"""Dependency-light tests for the Task-5/Task-8 simulator smoke gate."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_test_task58_corrected_wrapper import classify_task


def _summary():
  return {
      'reported_training_horizon': 150,
      'reset_success_count': 0,
      'state_mechanism_linf_error_max': 0.0,
      'goal_mechanism_linf_error_max': 0.0,
      'internal_goal_linf_error_max': 0.0,
      'internal_target_linf_error_max': 0.0,
      'zero_action_mechanism_displacement_max': 0.0,
      'reward_axis_mismatch_steps': 0,
      'info_axis_mismatch_steps': 0,
      'success_by_training_horizon_rate': 1.0,
  }


def _classify(summary):
  return classify_task(
      summary,
      expert_success_min=0.8,
      observation_tolerance=1e-5,
      goal_tolerance=1e-6,
      zero_action_tolerance=0.02)


def test_clean_summary_passes():
  result = _classify(_summary())
  assert result['passed']
  assert result['failed_gates'] == []


def test_stale_reset_transition_fails_closed():
  summary = _summary()
  summary['zero_action_mechanism_displacement_max'] = 0.2
  result = _classify(summary)
  assert not result['passed']
  assert 'zero_action_has_no_reset_jump' in result['failed_gates']


def test_reward_and_solvability_fail_independently():
  summary = _summary()
  summary['reward_axis_mismatch_steps'] = 1
  summary['success_by_training_horizon_rate'] = 0.6
  result = _classify(summary)
  assert not result['passed']
  assert 'reward_matches_axis_predicate' in result['failed_gates']
  assert 'scripted_policy_solves_by_training_horizon' in result['failed_gates']


def test_launcher_is_short_evaluation_only_gate():
  source = (REPO_ROOT / 'DRAFT_task58_wrapper_smoke.sh').read_text(
      encoding='utf-8')
  assert 'scripts/smoke_test_task58_corrected_wrapper.py' in source
  assert '--seeds 5 6 7' in source
  assert '--episodes 5' in source
  assert '--training-horizon 150' in source
  assert '--max-steps 200' in source
  assert 'run_continual_contrastive.py' not in source


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Task-5/Task-8 wrapper smoke tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
