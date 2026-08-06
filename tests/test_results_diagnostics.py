"""Tests for mixed-cadence success and learner-diagnostic metrics."""
from pathlib import Path
import importlib.util

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'compute_metrics_for_test',
    ROOT / 'results' / 'scripts' / 'compute_metrics.py')
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def _row(env_steps, success=None, shortcut=None):
  return {
      'group': 'diagnostic',
      'actor_mode': 'reset',
      'critic_mode': 'dcc_sac',
      'seed': 5,
      'task_idx': 5,
      'env': 'sawyer_handle_press_side',
      'run_id': 'run-1',
      'env_steps': env_steps,
      'success_rate': success,
      'learner/shortcut/action_shuffle_retention': shortcut,
  }


def test_success_and_diagnostics_survive_disjoint_logging_cadences():
  histories = pd.DataFrame([
      _row(0, success=0.0),
      _row(10, shortcut=0.9),
      _row(20, success=0.4),
      _row(30, shortcut=0.5),
  ])

  success = METRICS.compute_per_seed_per_task(histories)
  assert len(success) == 1
  assert success.iloc[0]['n_evals'] == 2
  assert success.iloc[0]['best_success'] == 0.4

  per_run, summary = METRICS.compute_diagnostic_metrics(histories)
  assert len(per_run) == 1
  row = per_run.iloc[0]
  assert row['n_points'] == 2
  assert row['first'] == 0.9
  assert row['final'] == 0.5
  assert len(summary) == 1


def test_missing_diagnostic_columns_produce_empty_tables():
  histories = pd.DataFrame([
      {
          key: value
          for key, value in _row(0, success=0.1).items()
          if not key.startswith('learner/')
      },
  ])
  per_run, summary = METRICS.compute_diagnostic_metrics(histories)
  assert per_run.empty
  assert summary.empty
