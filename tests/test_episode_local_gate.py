#!/usr/bin/env python3
"""Whole-episode Success-BC with the local-σ_s critic gate."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_episode_local_gate as cfg


def test_four_cells():
  configs = cfg.build_configs()
  assert len(configs) == 4
  assert [c['variant'] for c in configs] == [
      'task4_stick_episode_local_gate', 'task7_shelf_episode_local_gate',
      'handle_episode_local_gate', 'window_episode_local_gate']
  for config in configs:
    assert config['success_bc_label_mode'] == 'episode_sparse_reward'
    assert config['success_bc_critic_gate'] is True
    assert config['success_bc_critic_gate_mode'] == 'local'
    assert config['success_bc_weight'] == 0.1
    assert config['seed'] == 6
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['truncate_on_success'] is False

  stick, shelf, handle, window = configs
  assert stick['start_task'] == 4
  assert stick['steps_per_task'] == 8_000_000
  assert Path(stick['resume_checkpoint_file']).is_file()
  assert shelf['start_task'] == 7
  assert Path(shelf['resume_checkpoint_file']).is_file()
  assert handle['single_task'] == 'sawyer_handle_press_side'
  assert window['single_task'] == 'sawyer_window_close'


def test_runner_still_copies_whole_episode():
  runner = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  flatten = runner.split('def flatten_fn', 1)[1].split(
      'def flatten_intrajectory_fn', 1)[0]
  episode = flatten.split(
      "success_bc_label_mode == 'episode_sparse_reward'")[1]
  episode = episode.split("success_bc_label_mode ==", 1)[0]
  assert 'tf.fill' in episode
  assert 'tf.reduce_max(sample.data.reward[:-1])' in episode


def test_launcher_points_at_grid():
  launcher = (
      REPO_ROOT / 'DRAFT_jubail_episode_local_gate.sh'
  ).read_text(encoding='utf-8')
  assert 'experiment_configs_episode_local_gate.py' in launcher
  assert 'tests/test_episode_local_gate.py' in launcher
  assert '#SBATCH --array=0-3' in launcher


if __name__ == '__main__':
  test_four_cells()
  test_runner_still_copies_whole_episode()
  test_launcher_points_at_grid()
  print('test_episode_local_gate: ok')
