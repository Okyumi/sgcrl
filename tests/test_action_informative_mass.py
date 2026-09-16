#!/usr/bin/env python3
"""Tests for action-informative-mass helpers and the figure pipeline."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.action_informative_mass import (
    TASK_SPECS,
    biased_action,
    bootstrap_survival,
    clip_action,
    contact_flag,
    default_eps_grid,
    family_range,
    make_candidate_actions,
    relative_delta,
    score_range,
    state_score_scale,
    survival_curve,
    task_coordinate,
    time_bin_mean,
)


def test_task_indices_match_continual_sequence():
  from contrastive.continual_config import CONTINUAL_TASK_SEQUENCE
  assert CONTINUAL_TASK_SEQUENCE[5] == 'sawyer_handle_press_side'
  assert CONTINUAL_TASK_SEQUENCE[6] == 'sawyer_push'
  assert CONTINUAL_TASK_SEQUENCE[8] == 'sawyer_window_close'
  assert TASK_SPECS['sawyer_push']['index'] == 6


def test_candidates_are_clipped_and_labelled():
  rng = np.random.default_rng(0)
  pi = np.array([0.9, -0.9, 0.2, 0.0], dtype=np.float32)
  replay = rng.uniform(-1, 1, size=(16, 4)).astype(np.float32)
  acts, fam = make_candidate_actions(pi, replay, rng, n_gauss=2, n_shuffle=2, n_uniform=2)
  assert np.all(acts >= -1.0) and np.all(acts <= 1.0)
  assert 'pi' in fam.tolist() and 'zero' in fam.tolist()
  assert any(str(x).startswith('gauss_0.10') for x in fam)
  assert clip_action(np.array([2.0, -3.0]))[0] == 1.0


def test_survival_is_monotone_and_shared_grid():
  rng = np.random.default_rng(1)
  values = np.abs(rng.normal(size=200))
  grid = default_eps_grid(float(values.max()), n=21)
  surv = survival_curve(values, grid)
  assert np.all(np.diff(surv) <= 1e-12)
  assert surv[0] >= surv[-1]
  mean, lo, hi = bootstrap_survival(values, grid, rng, n_boot=20)
  assert mean.shape == grid.shape
  assert np.all(lo <= hi + 1e-8)


def test_family_range_uses_pi_and_prefix():
  scores = np.array([1.0, 0.0, 3.0, 2.0])
  fam = np.array(['pi', 'zero', 'gauss_0.10', 'gauss_0.10'])
  assert abs(family_range(scores, fam, 'gauss_0.10') - 2.0) < 1e-6
  assert abs(score_range(scores) - 3.0) < 1e-6


def test_contact_and_coordinates():
  hover = np.array([-0.07, 0.70, 0.10, 0.3, -0.07, 0.70, 0.12], np.float32)
  assert contact_flag(hover, 'sawyer_handle_press_side')
  assert abs(task_coordinate(hover, 'sawyer_handle_press_side') - 0.12) < 1e-6
  cube = np.array([0.02, 0.89, 0.08, 0.4, 0.02, 0.80, 0.02], np.float32)
  assert task_coordinate(cube, 'sawyer_push') > 0.05
  a = biased_action(np.zeros(4, np.float32), 'sawyer_handle_press_side')
  assert a[2] == -1.0


def test_relative_delta_is_scale_free():
  f_pi = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float64)
  delta = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
  rel = relative_delta(delta, f_pi)
  assert np.allclose(rel, 1.0 / np.std(f_pi))
  assert state_score_scale(f_pi) == float(np.std(f_pi))


def test_time_bins():
  t = np.linspace(0, 1, 150)
  y = t.copy()
  centers, means, ses, counts = time_bin_mean(t, y, n_bins=10)
  assert centers.shape == (10,)
  assert counts.sum() == 150
  assert means[0] < means[-1]


def test_window_config_matches_handle_push():
  import experiment_configs_jubail_window_measure as win
  import experiment_configs_jubail_task5_action_advice as base
  window = win.build_configs()[0]
  handle = base.build_configs()[0]
  for key in (
      'actor_mode', 'critic_mode', 'seed', 'steps_per_task',
      'network_width', 'sawyer_success_mode', 'adapt_heads_only',
      'dyn_aux_weight', 'success_bc_weight', 'mid_task_checkpoint_every'):
    assert window[key] == handle[key], key
  assert window['single_task'] == 'sawyer_window_close'


def test_plot_synthetic(tmp_path: Path):
  import importlib.util
  path = REPO_ROOT / 'scripts' / 'plot_action_informative_mass.py'
  spec = importlib.util.spec_from_file_location('plot_aim', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  records = []
  rng = np.random.default_rng(0)
  for env, loc, scale in (
      ('sawyer_handle_press_side', 0.2, 0.05),
      ('sawyer_window_close', 0.25, 0.06),
      ('sawyer_push', 1.5, 0.4),
  ):
    n = 300
    t = np.repeat(np.linspace(0, 1, 50), 6)
    spike = np.exp(-((t - 0.6) ** 2) / (2 * 0.01))
    if env == 'sawyer_push':
      delta = loc + scale * (0.3 + t)
    else:
      delta = loc + scale * spike
    contact = (np.abs(t - 0.6) < 0.08).astype(np.float64)
    if env == 'sawyer_push':
      contact = (t > 0.25).astype(np.float64) * 0.8
    npz = {
        'delta_f_task': delta,
        'delta_f_gauss_0.10': 0.4 * delta,
        'delta_f_gauss_0.25': 0.7 * delta,
        't_frac': t,
        'contact': contact,
        'env_steps': np.array([100000.0 if 'handle' in env else 250000.0]),
        'categorical_accuracy': np.array([0.9]),
        'action_shuffle_accuracy': np.array([0.85]),
        'grad_norm': rng.random(n),
        'f_pi': rng.normal(size=n),
    }
    records.append({
        'env_name': env,
        'meta': {'env_steps': int(npz['env_steps'][0]), 'env_name': env},
        'data': npz,
        'steps': int(npz['env_steps'][0]),
        'path': tmp_path / f'{env}.npz',
    })
  pdf = tmp_path / 'fig.pdf'
  png = tmp_path / 'fig.png'
  csv_path = tmp_path / 'fig.csv'
  module.plot(records, pdf, png, csv_path)
  assert pdf.exists() and png.exists() and csv_path.exists()


def main():
  test_task_indices_match_continual_sequence()
  test_candidates_are_clipped_and_labelled()
  test_survival_is_monotone_and_shared_grid()
  test_family_range_uses_pi_and_prefix()
  test_contact_and_coordinates()
  test_relative_delta_is_scale_free()
  test_time_bins()
  test_window_config_matches_handle_push()
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    test_plot_synthetic(Path(tmp))
  print('action-informative-mass tests passed')


if __name__ == '__main__':
  main()
