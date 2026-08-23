"""Dependency-light checks for the staged Outcome-Calibrated DCC sweep."""
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_configs_outcome_falsification import build_configs
from scripts.evaluate_outcome_falsification import (
    ACTION_SHUFFLE_RETENTION_KEY,
    FIXED_STATE_ACTION_STD_KEY,
    SAME_STATE_SPEARMAN_KEY,
)


def test_stage_layout():
  configs = build_configs()
  assert len(configs) == 12
  assert Counter(c['falsification_stage'] for c in configs) == {
      1: 4, 2: 4, 3: 4}
  for stage in (1, 2, 3):
    stage_configs = [
        config for config in configs
        if config['falsification_stage'] == stage]
    assert Counter(config['seed'] for config in stage_configs) == {5: 2, 6: 2}
    assert {config['single_task'] for config in stage_configs} == {
        'sawyer_handle_press_side', 'sawyer_window_close'}
    assert all(config['steps_per_task'] == 1_000_000
               for config in stage_configs)
    assert all(config['action_landscape_diagnostic_interval_steps'] == 250_000
               for config in stage_configs)


def test_isolated_stage_changes():
  configs = build_configs()
  s1 = configs[0]
  s2 = configs[4]
  s3 = configs[8]
  assert s1['action_effect_actor_mode'] == 'effect_only'
  assert s1['action_effect_target_mode'] == 'psi_one_step'
  assert s1['success_bc_weight'] == 0.0
  assert s2['action_effect_target_mode'] == 'raw_horizon'
  assert s2['outcome_horizon'] == 25
  assert s2['success_bc_weight'] == 0.0
  assert s3['action_effect_target_mode'] == 'raw_horizon'
  assert s3['outcome_horizon'] == 25
  assert s3['success_bc_weight'] > 0.0


def test_wandb_metric_keys_match_logged_namespaces():
  assert SAME_STATE_SPEARMAN_KEY == (
      'learner/action_landscape/'
      'score_vs_rollout_mechanism_progress_spearman')
  assert FIXED_STATE_ACTION_STD_KEY == (
      'learner/outcome/fixed_state_action_std')
  assert ACTION_SHUFFLE_RETENTION_KEY == (
      'learner/outcome/action_shuffle_retention')


if __name__ == '__main__':
  test_stage_layout()
  test_isolated_stage_changes()
  test_wandb_metric_keys_match_logged_namespaces()
  print('outcome falsification checks passed')
