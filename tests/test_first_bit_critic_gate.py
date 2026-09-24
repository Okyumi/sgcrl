#!/usr/bin/env python3
"""First-sparse-bit labels and critic-flatness Success-BC gate."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.success_bc_labels import (
    critic_flatness_gate, first_sparse_bit_mask)
import experiment_configs_first_bit_critic_gate as cfg


def test_first_bit_drops_linger():
  linger = np.zeros(150, dtype=np.float32)
  linger[80:] = 1.0
  keep = first_sparse_bit_mask(linger)
  assert keep[80] == 1.0
  assert keep[:80].sum() == 0.0
  assert keep[81:].sum() == 0.0
  assert int(keep.sum()) == 1

  empty = np.zeros(150, dtype=np.float32)
  assert first_sparse_bit_mask(empty).sum() == 0.0

  early = np.zeros(150, dtype=np.float32)
  early[5] = 1.0
  early[40:] = 1.0
  keep_early = first_sparse_bit_mask(early)
  assert keep_early[5] == 1.0
  assert int(keep_early.sum()) == 1


def test_tf_first_bit_matches_numpy():
  try:
    import tensorflow as tf
    from contrastive.success_bc_labels import tensorflow_first_sparse_bit
  except Exception:
    return
  linger = np.zeros(151, dtype=np.float32)
  linger[80:-1] = 1.0
  keep_np = first_sparse_bit_mask(linger[:-1])
  keep_tf = tensorflow_first_sparse_bit(tf, tf.constant(linger)).numpy()
  np.testing.assert_array_equal(keep_tf, keep_np)


def test_critic_flatness_gate_clones_when_action_flat():
  np.testing.assert_allclose(
      critic_flatness_gate(0.0, 2.0), 1.0, atol=1e-6)
  np.testing.assert_allclose(
      critic_flatness_gate(10.0, 1.0), 1.0 / 11.0, atol=1e-5)
  np.testing.assert_allclose(
      critic_flatness_gate(0.0, 0.0), 1.0, atol=1e-6)


def test_eight_cells():
  configs = cfg.build_configs()
  assert len(configs) == 8
  first_bit = configs[:4]
  critic_gate = configs[4:]
  assert [c['variant'] for c in first_bit] == [
      'task4_stick_first_bit', 'task7_shelf_first_bit',
      'handle_first_bit', 'window_first_bit']
  assert [c['variant'] for c in critic_gate] == [
      'task4_stick_critic_gate', 'task7_shelf_critic_gate',
      'handle_critic_gate', 'window_critic_gate']
  for config in first_bit:
    assert config['success_bc_label_mode'] == 'first_sparse_bit'
    assert config['success_bc_critic_gate'] is False
    assert config['success_bc_weight'] == 0.1
    assert config['seed'] == 6
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['dyn_aux_weight'] == 0.0
    assert config['truncate_on_success'] is False
  for config in critic_gate:
    assert config['success_bc_label_mode'] == 'current_sparse_reward'
    assert config['success_bc_critic_gate'] is True
    assert config['success_bc_weight'] == 0.1
    assert config['seed'] == 6

  stick, shelf, handle, window = first_bit
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


def test_runner_and_learner_have_no_geometry():
  runner = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  learner = (
      REPO_ROOT / 'contrastive' / 'continual_learning_decomposed.py'
  ).read_text(encoding='utf-8')
  flatten = runner.split('def flatten_fn', 1)[1].split(
      'def flatten_intrajectory_fn', 1)[0]
  first_bit = flatten.split(
      "success_bc_label_mode == 'first_sparse_bit'")[1]
  first_bit = first_bit.split('if actor_goal_mode', 1)[0]
  assert 'tensorflow_first_sparse_bit' in first_bit
  assert 'tf.fill' not in first_bit
  intra = runner.split('def flatten_intrajectory_fn', 1)[1]
  intra_bit = intra.split(
      "success_bc_label_mode == 'first_sparse_bit'")[1]
  intra_bit = intra_bit.split('if actor_goal_mode', 1)[0]
  assert 'tensorflow_first_sparse_bit' in intra_bit
  assert 'tf.gather(first_keep, anchor_index)' in intra_bit

  gate = learner.split('if success_bc_critic_gate:', 1)[1]
  gate = gate.split('else:', 1)[0]
  assert 'apply_paired_score' in gate
  assert 'jax.random.uniform' in gate
  assert 'sigma_a' in gate
  assert 'sigma_s' in gate
  assert 'score_stored' not in gate
  assert 'bc_action' not in gate.split('a_probe', 1)[1]
  assert '0.09' not in gate
  assert 'hover' not in gate.lower()
  assert 'handle' not in gate.lower()
  assert 'interaction_threshold' not in gate

  launcher = (REPO_ROOT / 'DRAFT_jubail.sh').read_text(encoding='utf-8')
  assert '--success_bc_critic_gate' in launcher
  assert '--nosuccess_bc_critic_gate' in launcher


def test_launcher_points_at_grid():
  launcher = (
      REPO_ROOT / 'DRAFT_jubail_first_bit_critic_gate.sh'
  ).read_text(encoding='utf-8')
  assert 'experiment_configs_first_bit_critic_gate.py' in launcher
  assert 'tests/test_first_bit_critic_gate.py' in launcher
  assert '#SBATCH --array=0-7' in launcher


if __name__ == '__main__':
  test_first_bit_drops_linger()
  test_tf_first_bit_matches_numpy()
  test_critic_flatness_gate_clones_when_action_flat()
  test_eight_cells()
  test_runner_and_learner_have_no_geometry()
  test_launcher_points_at_grid()
  print('test_first_bit_critic_gate: ok')
