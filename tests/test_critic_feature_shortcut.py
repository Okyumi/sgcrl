#!/usr/bin/env python3
"""Tests for critic feature-shortcut helpers and checkpoint matching."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import feature_shortcut as fs


def _load_helper(filename: str):
  import importlib.util
  path = REPO_ROOT / 'scripts' / filename
  spec = importlib.util.spec_from_file_location(filename, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_transplant_only_changes_block():
  dst = np.arange(11, dtype=np.float32)
  src = np.ones(11, dtype=np.float32)
  out = fs.transplant_indices(dst, src, fs.MECH_Z)
  assert out[6] == 1.0
  assert np.allclose(out[:6], dst[:6])
  assert np.allclose(out[7:], dst[7:])


def test_synthetic_progress_is_coordinate_local():
  hover = np.array(
      [0.1, 0.5, 0.2, 0.4, -0.07, 0.70, 0.12, 0, 0, 0, 0], dtype=np.float32)
  handle = fs.synthetic_progress_state(hover, 'sawyer_handle_press_side')
  assert abs(handle[6] - 0.07) < 1e-6
  assert np.allclose(handle[:6], hover[:6])
  push = fs.synthetic_progress_state(hover, 'sawyer_push')
  assert np.allclose(push[4:7], fs.PUSH_TARGET)
  assert np.allclose(push[:4], hover[:4])


def test_recovery_fraction():
  assert abs(fs.recovery_fraction(0.0, 8.0, 10.0) - 0.8) < 1e-6
  assert np.isnan(fs.recovery_fraction(1.0, 1.0, 1.0))


def test_z_match_shuffle_drops_accuracy():
  rng = np.random.default_rng(0)
  n = 32
  state = rng.normal(size=(n, 11)).astype(np.float32)
  goal = state.copy()
  goal[:, 6] = state[:, 6] + rng.normal(scale=0.01, size=n).astype(np.float32)

  def logits(s, g):
    return -np.abs(s[:, 6:7] - g[:, 6:7].T)

  base = fs.categorical_accuracy(logits(state, goal))
  shuffled = fs.shuffle_block(goal, fs.MECH_Z, rng)
  dropped = fs.categorical_accuracy(logits(state, shuffled))
  assert base > 0.8
  assert dropped < 0.4


def test_occupancy_binary_low_entropy():
  z = np.concatenate([np.full(90, 0.12), np.full(10, 0.07)])
  stats = fs.occupancy_stats(z, 0.05, 0.09)
  assert abs(stats['frac_in_band'] - 0.1) < 1e-6
  spread = np.linspace(0.0, 0.3, 100)
  spread_stats = fs.occupancy_stats(spread, 0.05, 0.09)
  assert spread_stats['bernoulli_entropy'] > stats['bernoulli_entropy']


def test_her_pairs_are_futures():
  states = np.stack([np.full(11, i, dtype=np.float32) for i in range(6)])
  actions = np.zeros((6, 4), dtype=np.float32)
  s, a, g = fs.make_her_pairs(
      [{'states': states, 'actions': actions}],
      np.random.default_rng(0), max_pairs=20, discount=0.99)
  assert s.shape[0] == 5
  assert np.all(g[:, 0] > s[:, 0])


def test_classify_handle_and_push():
  hover = np.array([-0.07, 0.70, 0.10, 0.3, -0.07, 0.70, 0.12], np.float32)
  succ = np.array([-0.07, 0.70, 0.08, 0.3, -0.07, 0.70, 0.07], np.float32)
  assert fs.classify_state(hover, 'sawyer_handle_press_side') == 'hover_contact'
  assert fs.classify_state(succ, 'sawyer_handle_press_side') == 'success'
  cube = np.array([0.02, 0.89, 0.08, 0.4, 0.02, 0.80, 0.02], np.float32)
  assert fs.classify_state(cube, 'sawyer_push') == 'object_progress'


def test_exact_env_dir_match_skips_inject(tmp_path: Path):
  env = 'sawyer_handle_press_side'
  measure = (
      tmp_path /
      f'actor_reset_critic_decomposed_tid_False_heads_False_success_corrected_dyn1.000_pt256x4_env_{env}'
      / 'seed_6')
  inject = (
      tmp_path /
      f'actor_reset_critic_decomposed_tid_False_heads_False_success_corrected_dyn1.000_pt256x4_env_{env}_ret_abc'
      / 'seed_6')
  measure.mkdir(parents=True)
  inject.mkdir(parents=True)
  (measure / 'task_0_step_100200.pkl').write_bytes(b'x')
  (inject / 'task_0_step_951900.pkl').write_bytes(b'y')
  runner = _load_helper('run_feature_shortcut_from_checkpoints.py')
  advice = _load_helper('run_action_advice_from_checkpoints.py')
  ckpts = runner.list_step_ckpts(tmp_path, env, 6)
  assert [s for s, _ in ckpts] == [100200]
  advice_ckpts = advice.list_step_ckpts(tmp_path, env, 6)
  assert [s for s, _ in advice_ckpts] == [100200]


def main():
  test_transplant_only_changes_block()
  test_synthetic_progress_is_coordinate_local()
  test_recovery_fraction()
  test_z_match_shuffle_drops_accuracy()
  test_occupancy_binary_low_entropy()
  test_her_pairs_are_futures()
  test_classify_handle_and_push()
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    test_exact_env_dir_match_skips_inject(Path(tmp))
  print('critic feature-shortcut tests passed')


if __name__ == '__main__':
  main()
