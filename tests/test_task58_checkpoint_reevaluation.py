#!/usr/bin/env python3
"""Dependency-light tests for Task-5/Task-8 checkpoint reevaluation."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import task58_reevaluation as reevaluation
from experiment_configs_task58_checkpoint_reeval import build_configs
from scripts.reevaluate_task58_dcc_checkpoints import infer_actor_architecture
from scripts.summarize_task58_checkpoint_reevaluation import summarize


def _observation(mechanism, goal, obs_dim=11):
  observation = np.zeros(obs_dim * 2, dtype=np.float32)
  observation[4:7] = mechanism
  observation[obs_dim + 4:obs_dim + 7] = goal
  return observation


def test_task5_axis_can_rescue_a_legacy_false_negative():
  observation = _observation(
      mechanism=[9.0, -4.0, 0.071], goal=[0.0, 0.0, 0.070])
  legacy, axis = reevaluation.success_flags(
      observation, 11, 'sawyer_handle_press_side')
  assert legacy is False
  assert axis is True


def test_task8_axis_can_rescue_a_legacy_false_negative():
  observation = _observation(
      mechanism=[0.04, 8.0, -3.0], goal=[0.0, 0.0, 0.0])
  legacy, axis = reevaluation.success_flags(
      observation, 11, 'sawyer_window_close')
  assert legacy is False
  assert axis is True


def test_axis_failure_remains_failure():
  observation = _observation(
      mechanism=[0.051, 0.0, 0.0], goal=[0.0, 0.0, 0.0])
  legacy, axis = reevaluation.success_flags(
      observation, 11, 'sawyer_window_close')
  assert legacy is False
  assert axis is False


def test_task_id_layout_keeps_mechanism_offsets_aligned():
  obs_dim = 21
  observation = np.zeros(obs_dim * 2, dtype=np.float32)
  observation[4:7] = [0.04, 9.0, 9.0]
  observation[obs_dim + 4:obs_dim + 7] = [0.0, 0.0, 0.0]
  assert reevaluation.success_flags(
      observation, obs_dim, 'sawyer_window_close') == (False, True)


def test_config_matrix_targets_six_historical_checkpoints():
  configs = build_configs()
  assert len(configs) == 6
  assert {(config['task_id'], config['seed']) for config in configs} == {
      (task_id, seed) for task_id in (5, 8) for seed in (5, 6, 7)
  }
  assert all(config['episodes'] == 100 for config in configs)
  assert all(config['checkpoint_relative'].startswith(
      'actor_reset_critic_decomposed_tid_False_heads_True/seed_')
             for config in configs)
  assert all('_dyn1.000_pt256x4' not in config['checkpoint_relative']
             for config in configs)
  assert all(config['checkpoint_relative'].endswith(
      f"task_{config['task_id']}.pkl") for config in configs)


def test_residual_actor_architecture_is_inferred_from_checkpoint():
  params = {
      'actor_body/linear': {
          'w': np.zeros((22, 1024)), 'b': np.zeros((1024,))},
  }
  for index in range(1, 6):
    params[f'actor_body/~/linear_{index}'] = {
        'w': np.zeros((1024, 1024)), 'b': np.zeros((1024,))}
  architecture = infer_actor_architecture(params)
  assert architecture['use_residual'] is True
  assert architecture['input_width'] == 22
  assert architecture['network_width'] == 1024
  assert architecture['actor_depth'] == 4


def test_evaluator_is_policy_only_and_paired():
  source = (REPO_ROOT / 'scripts' /
            'reevaluate_task58_dcc_checkpoints.py').read_text(
                encoding='utf-8')
  assert "checkpoint['composed_policy']" in source
  assert 'eval_mode=True' in source
  assert "sawyer_success_mode='legacy_distance'" in source
  assert 'PairedTask58SuccessObserver' in source
  assert 'train_single_task' not in source
  assert 'reverb' not in source.lower()


def test_six_result_summary_preserves_paired_seed_values():
  payloads = []
  for task_id, env_name in (
      (5, 'sawyer_handle_press_side'),
      (8, 'sawyer_window_close')):
    for seed in (5, 6, 7):
      legacy = 0.1 if task_id == 5 else 0.2
      axis = legacy + 0.3
      payloads.append({
          'status': 'finished',
          'env_name': env_name,
          'seed': seed,
          'summary': {
              'legacy_success_rate': legacy,
              'task_axis_success_rate': axis,
              'axis_rescued_success_rate': 0.3,
          },
      })
  result = summarize(payloads)
  assert result['num_checkpoints'] == 6
  assert result['conclusion'] == 'axis_metric_changes_reported_performance'
  assert np.isclose(result['tasks']['sawyer_handle_press_side'][
      'success_rate_gain_mean'], 0.3)


def test_launcher_is_evaluation_only():
  launcher = (REPO_ROOT / 'DRAFT_task58_checkpoint_reeval.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-2' in launcher
  assert 'reevaluate_task58_dcc_checkpoints.py' in launcher
  assert 'run_continual_contrastive.py' not in launcher
  assert 'STEPS_PER_TASK' not in launcher


def test_training_launcher_was_withdrawn():
  assert not (REPO_ROOT / 'DRAFT_task58_axis_reward.sh').exists()
  assert not (REPO_ROOT / 'experiment_configs_task58_axis_reward.py').exists()


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Task-5/Task-8 checkpoint reevaluation tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
