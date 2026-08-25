#!/usr/bin/env python3
"""Dependency-light tests for the paired Sawyer wrapper controls."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import goal_semantics
from experiment_configs_goal_wrapper_positive_controls import build_configs
from scripts import audit_sawyer_goal_positive_controls as audit
from scripts import evaluate_goal_wrapper_positive_controls as evaluator


def _condition(
    *,
    native_success=1.0,
    positive_success=1.0,
    native_axis_success=1.0,
    fixed_axis_success=0.0,
    trajectory_error=0.0,
):
  return {
      'native_info_success_mean': native_success,
      'positive_reward_success_mean': positive_success,
      'native_axis_success_mean': native_axis_success,
      'fixed_axis_success_mean': fixed_axis_success,
      'native_info_axis_mismatch_fraction_mean': 0.0,
      'reward_native_info_mismatch_fraction_mean': 0.0,
      'evaluate_state_fallback_fraction_mean': 1.0,
      'trajectory_linf_error_vs_native_max': trajectory_error,
  }


def _summary():
  return {
      'conditions': {
          'native_official': _condition(),
          'wrapper_native_target_policy': _condition(),
          'wrapper_native_target_replay': _condition(),
          'wrapper_fixed_target_policy': _condition(
              native_success=0.0, positive_success=0.0),
          'wrapper_fixed_target_replay': _condition(
              native_success=0.0,
              positive_success=0.0,
              native_axis_success=1.0,
              fixed_axis_success=0.0),
      },
      'pairing': {
          'native_target_pair_linf_error_max': 0.0,
          'rand_vec_pair_linf_error_max': 0.0,
          'initial_mechanism_pair_linf_error_max': 0.0,
          'fixed_to_native_target_distance_mean': 0.10,
          'fixed_to_native_success_axis_distance_mean': 0.10,
      },
  }


def _classify(summary):
  return audit.classify_task(
      summary,
      expert_success_min=0.80,
      fixed_success_max=0.20,
      trajectory_tolerance=1e-5,
      target_tolerance=1e-6,
      success_threshold=0.02)


def test_metadata_names_untouched_native_tasks():
  assert goal_semantics.VALIDITY_TASKS[
      'sawyer_handle_press_side']['native_env_name'] == (
          'handle-press-side-v2')
  assert goal_semantics.VALIDITY_TASKS[
      'sawyer_window_close']['native_env_name'] == 'window-close-v2'


def test_fixed_target_misalignment_signature():
  result = _classify(_summary())
  assert result['decision'] == 'fixed_global_target_misaligned'
  assert result['fixed_replay_reaches_native_endpoint'] is True
  assert result['fixed_target_valid'] is False


def test_native_positive_control_failure_takes_precedence():
  summary = _summary()
  summary['conditions']['native_official']['native_info_success_mean'] = 0.0
  summary['conditions']['native_official']['native_axis_success_mean'] = 0.0
  result = _classify(summary)
  assert result['decision'] == 'native_positive_control_failed'


def test_native_info_axis_disagreement_is_audit_error():
  summary = _summary()
  summary['conditions']['native_official']['native_axis_success_mean'] = 0.0
  summary['conditions']['native_official'][
      'native_info_axis_mismatch_fraction_mean'] = 1.0
  result = _classify(summary)
  assert result['decision'] == 'audit_metric_inconsistent'


def test_native_target_wrapper_failure_is_separate():
  summary = _summary()
  summary['conditions'][
      'wrapper_native_target_policy']['positive_reward_success_mean'] = 0.0
  result = _classify(summary)
  assert result['decision'] == 'custom_wrapper_invalid'


def test_valid_fixed_target_can_pass():
  summary = _summary()
  summary['conditions'][
      'wrapper_fixed_target_policy']['positive_reward_success_mean'] = 1.0
  summary['conditions'][
      'wrapper_fixed_target_policy']['native_info_success_mean'] = 1.0
  summary['conditions'][
      'wrapper_fixed_target_policy']['fixed_axis_success_mean'] = 1.0
  summary['conditions'][
      'wrapper_fixed_target_replay']['positive_reward_success_mean'] = 1.0
  summary['conditions'][
      'wrapper_fixed_target_replay']['native_info_success_mean'] = 1.0
  summary['conditions'][
      'wrapper_fixed_target_replay']['fixed_axis_success_mean'] = 1.0
  result = _classify(summary)
  assert result['decision'] == 'fixed_target_valid'


def test_config_matrix():
  configs = build_configs()
  assert len(configs) == 3
  assert [config['seed'] for config in configs] == [5, 6, 7]
  assert all(config['episodes'] == 50 for config in configs)
  assert all(config['max_steps'] == 150 for config in configs)
  assert all(config['wandb_group'] ==
             'GOAL-WRAPPER-POSITIVE-CONTROLS-V4'
             for config in configs)


def test_three_seed_aggregation_blocks_promotion_on_confirmed_issue():
  payloads = []
  for seed in (5, 6, 7):
    results = []
    for env_name in (
        'sawyer_handle_press_side', 'sawyer_window_close'):
      summary = _summary()
      summary['classification'] = _classify(summary)
      results.append({'env_name': env_name, 'summary': summary})
    payloads.append({
        'audit_version': 4,
        'seed': seed,
        'results': results,
    })
  report = evaluator.aggregate(payloads)
  assert report['conclusion'] == 'fixed_global_targets_invalid'
  assert report['promotion_allowed'] is False
  assert report['all_fixed_misalignment_confirmed'] is True


def test_launcher_activates_project_environment():
  launcher = (
      REPO_ROOT / 'DRAFT_goal_wrapper_positive_controls.sh').read_text(
          encoding='utf-8')
  assert 'conda activate contrastive_rl' in launcher
  assert 'audit_sawyer_goal_positive_controls.py' in launcher
  assert '#SBATCH --array=0-2' in launcher
  assert '#SBATCH --gres=gpu:1' in launcher
  assert 'MUJOCO_GL=egl' in launcher
  assert '--strict-current-wrapper' not in launcher


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'goal-wrapper positive-control tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
