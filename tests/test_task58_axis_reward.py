#!/usr/bin/env python3
"""Dependency-light checks for the direct Task-5/Task-8 reward fix."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import sawyer_success
from experiment_configs_task58_axis_reward import build_configs


class _Environment:

  def __init__(self, mode='task_axis'):
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


def test_runner_and_checkpoint_identity_accept_task_axis():
  source = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  assert "('legacy_distance', 'task_axis', 'native_info')" in source
  assert "config_key += f'_success_{sawyer_success_mode}'" in source


def test_minimal_config_matrix():
  configs = build_configs()
  assert len(configs) == 4
  assert {(config['single_task'], config['seed']) for config in configs} == {
      ('sawyer_handle_press_side', 5),
      ('sawyer_handle_press_side', 6),
      ('sawyer_window_close', 5),
      ('sawyer_window_close', 6),
  }
  assert all(config['critic_mode'] == 'decomposed' for config in configs)
  assert all(config['actor_mode'] == 'reset' for config in configs)
  assert all(config['base_steps'] == 1_000_000 for config in configs)
  assert all(config['sawyer_success_mode'] == 'task_axis'
             for config in configs)
  assert all(config['counterfactual_rank_interval_steps'] == 0
             for config in configs)
  assert all(config['counterfactual_oracle_interval_steps'] == 0
             for config in configs)


def test_launcher_runs_only_the_four_cells():
  launcher = (REPO_ROOT / 'DRAFT_task58_axis_reward.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-1' in launcher
  assert 'export CONFIG_LIMIT=4' in launcher
  assert 'export TASKS_PER_GPU=2' in launcher
  assert 'experiment_configs_task58_axis_reward.py' in launcher


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Task-5/Task-8 axis-reward tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
