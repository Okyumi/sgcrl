#!/usr/bin/env python3
"""Checks for the universal Success-BC 4/7 vs 5/8 label grid."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_universal_success_bc as cfg


def test_four_cells():
  configs = cfg.build_configs()
  assert len(configs) == 4
  assert [c['variant'] for c in configs] == [
      'task4_stick_terminal', 'task7_shelf_terminal',
      'handle_current_sparse', 'window_current_sparse']
  for config in configs:
    assert config['success_bc_weight'] == 0.1
    assert config['seed'] == 6
    assert config['steps_per_task'] == 1_000_000
    assert config['truncate_on_success'] is False

  stick, shelf, handle, window = configs
  assert stick['start_task'] == 4
  assert stick['success_bc_label_mode'] == 'terminal_episode'
  assert stick['sawyer_success_mode'] == 'native_info'
  assert Path(stick['resume_checkpoint_file']).is_file()
  assert shelf['start_task'] == 7
  assert shelf['success_bc_label_mode'] == 'terminal_episode'
  assert Path(shelf['resume_checkpoint_file']).is_file()
  assert handle['single_task'] == 'sawyer_handle_press_side'
  assert handle['success_bc_label_mode'] == 'current_sparse_reward'
  assert handle['sawyer_success_mode'] == 'corrected'
  assert window['single_task'] == 'sawyer_window_close'
  assert window['success_bc_label_mode'] == 'current_sparse_reward'


def test_launcher_points_at_grid():
  launcher = (
      REPO_ROOT / 'DRAFT_jubail_universal_success_bc.sh'
  ).read_text(encoding='utf-8')
  assert 'experiment_configs_universal_success_bc.py' in launcher
  assert 'tests/test_universal_success_bc.py' in launcher
  assert '#SBATCH --array=0-3' in launcher


if __name__ == '__main__':
  test_four_cells()
  test_launcher_points_at_grid()
  print('test_universal_success_bc: ok')
