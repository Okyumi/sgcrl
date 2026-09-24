#!/usr/bin/env python3
"""Per-step Success-BC probe on Tasks 4/7 and 5/8."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_current_sparse_47_58 as cfg


def test_four_cells():
  configs = cfg.build_configs()
  assert len(configs) == 4
  assert [c['variant'] for c in configs] == [
      'task4_stick_current_sparse', 'task7_shelf_current_sparse',
      'handle_current_sparse', 'window_current_sparse']
  for config in configs:
    assert config['success_bc_label_mode'] == 'current_sparse_reward'
    assert config['success_bc_weight'] == 0.1
    assert config['seed'] == 6
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['dyn_aux_weight'] == 0.0
    assert config['truncate_on_success'] is False

  stick, shelf, handle, window = configs
  assert stick['start_task'] == 4
  assert stick['steps_per_task'] == 8_000_000
  assert Path(stick['resume_checkpoint_file']).is_file()
  assert shelf['start_task'] == 7
  assert shelf['steps_per_task'] == 8_000_000
  assert Path(shelf['resume_checkpoint_file']).is_file()
  assert handle['single_task'] == 'sawyer_handle_press_side'
  assert handle['steps_per_task'] == 1_000_000
  assert window['single_task'] == 'sawyer_window_close'
  assert window['steps_per_task'] == 1_000_000


def test_runner_inserts_current_bit():
  runner = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  flatten = runner.split('def flatten_fn', 1)[1].split(
      'def flatten_intrajectory_fn', 1)[0]
  current = flatten.split(
      "success_bc_label_mode == 'current_sparse_reward'")[1]
  current = current.split("success_bc_label_mode ==", 1)[0]
  assert 'sample.data.reward[:-1] > 0.0' in current
  assert 'tf.fill' not in current


def test_launcher_points_at_grid():
  launcher = (
      REPO_ROOT / 'DRAFT_jubail_current_sparse_47_58.sh'
  ).read_text(encoding='utf-8')
  assert 'experiment_configs_current_sparse_47_58.py' in launcher
  assert 'tests/test_current_sparse_47_58.py' in launcher
  assert '#SBATCH --array=0-3' in launcher


if __name__ == '__main__':
  test_four_cells()
  test_runner_inserts_current_bit()
  test_launcher_points_at_grid()
  print('test_current_sparse_47_58: ok')
