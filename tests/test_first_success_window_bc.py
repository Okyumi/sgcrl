#!/usr/bin/env python3
"""Checks for terminal-gated first-success-window Success-BC."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.success_bc_labels import first_success_window_mask
import experiment_configs_first_success_window_bc as cfg


def test_mask_drops_undo_and_linger():
  linger = np.zeros(150, dtype=np.float32)
  linger[80:] = 1.0
  keep = first_success_window_mask(linger, 64)
  assert keep[:17].sum() == 0.0
  assert keep[17] == 1.0
  assert keep[80] == 1.0
  assert keep[81:].sum() == 0.0
  assert int(keep.sum()) == 64

  undo = np.zeros(150, dtype=np.float32)
  undo[40:60] = 1.0
  assert first_success_window_mask(undo, 64).sum() == 0.0

  early = np.zeros(150, dtype=np.float32)
  early[5:] = 1.0
  keep_early = first_success_window_mask(early, 64)
  assert int(keep_early[:6].sum()) == 6
  assert keep_early[6:].sum() == 0.0


def test_tf_mask_matches_numpy():
  try:
    import tensorflow as tf
    from contrastive.success_bc_labels import tensorflow_first_success_window
  except Exception:
    return
  linger = np.zeros(151, dtype=np.float32)
  linger[80:-1] = 1.0
  keep_np = first_success_window_mask(linger[:-1], 64)
  keep_tf = tensorflow_first_success_window(
      tf, tf.constant(linger), 64).numpy()
  np.testing.assert_array_equal(keep_tf, keep_np)


def test_four_cells():
  configs = cfg.build_configs()
  assert len(configs) == 4
  assert [c['variant'] for c in configs] == [
      'task4_stick_window', 'task7_shelf_window',
      'handle_window', 'window_close_window']
  for config in configs:
    assert config['success_bc_label_mode'] == 'first_success_window'
    assert config['success_bc_window'] == 64
    assert config['success_bc_weight'] == 0.1
    assert config['seed'] == 6
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


def test_runner_uses_window_helper():
  runner = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  learner = (
      REPO_ROOT / 'contrastive' / 'continual_learning_decomposed.py'
  ).read_text(encoding='utf-8')
  flatten = runner.split('def flatten_fn', 1)[1].split(
      'def flatten_intrajectory_fn', 1)[0]
  window = flatten.split(
      "success_bc_label_mode == 'first_success_window'")[1]
  window = window.split('if actor_goal_mode', 1)[0]
  assert 'tensorflow_first_success_window' in window
  assert 'tf.fill' not in window
  assert 'first_success_window' in learner
  launcher = (
      REPO_ROOT / 'DRAFT_jubail.sh'
  ).read_text(encoding='utf-8')
  assert '--success_bc_window=$SUCCESS_BC_WINDOW' in launcher


def test_launcher_points_at_grid():
  launcher = (
      REPO_ROOT / 'DRAFT_jubail_first_success_window_bc.sh'
  ).read_text(encoding='utf-8')
  assert 'experiment_configs_first_success_window_bc.py' in launcher
  assert 'tests/test_first_success_window_bc.py' in launcher
  assert '#SBATCH --array=0-3' in launcher


if __name__ == '__main__':
  test_mask_drops_undo_and_linger()
  test_tf_mask_matches_numpy()
  test_four_cells()
  test_runner_uses_window_helper()
  test_launcher_points_at_grid()
  print('test_first_success_window_bc: ok')
