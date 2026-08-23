#!/usr/bin/env python3
"""Evaluate the pre-registered 1M promotion gates from W&B histories."""
from __future__ import annotations

import argparse
import math
import statistics


STAGE_NAMES = {
    1: 'effect-only-psi',
    2: 'raw-H25-outcome',
    3: 'raw-H25-retention',
}
TASK_LABELS = ('task5', 'task8')
BASELINE_GROUPS = {
    'task5': 'DCC-intrajectory-negatives-task5',
    'task8': 'DCC-intrajectory-negatives-task8',
}
SAME_STATE_SPEARMAN_KEY = (
    'learner/action_landscape/'
    'score_vs_rollout_mechanism_progress_spearman')
FIXED_STATE_ACTION_STD_KEY = 'learner/outcome/fixed_state_action_std'
ACTION_SHUFFLE_RETENTION_KEY = (
    'learner/outcome/action_shuffle_retention')


def _finite(values):
  return [float(value) for value in values
          if value is not None and math.isfinite(float(value))]


def _metric_history(run, key):
  return _finite(row.get(key) for row in run.scan_history(keys=[key]))


def _success_summary(run, max_points=None):
  values = _metric_history(run, 'evaluator/success_rate')
  if max_points is not None:
    values = values[:max_points]
  if not values:
    return {'auc': float('nan'), 'peak': float('nan'),
            'late': float('nan'), 'retention': float('nan')}
  late_count = max(1, len(values) // 5)
  peak = max(values)
  late = statistics.fmean(values[-late_count:])
  return {
      'auc': statistics.fmean(values),
      'peak': peak,
      'late': late,
      'retention': late / max(peak, 1e-8),
  }


def _mean_latest(runs, key):
  latest = []
  for run in runs:
    values = _metric_history(run, key)
    if values:
      latest.append(values[-1])
  return statistics.fmean(latest) if latest else float('nan')


def _mean_success(runs, field, max_points=None):
  values = _finite(
      _success_summary(run, max_points=max_points)[field] for run in runs)
  return statistics.fmean(values) if values else float('nan')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--stage', type=int, choices=(1, 2, 3), required=True)
  parser.add_argument('--entity', default='nyuad_mmvc')
  parser.add_argument('--project', default='continual_gcrl_paper')
  args = parser.parse_args()
  try:
    import wandb  # pylint: disable=import-outside-toplevel
  except ImportError as error:
    raise SystemExit(
        'wandb is unavailable; activate the contrastive_rl environment.') \
        from error

  api = wandb.Api()
  all_pass = True
  for task_label in TASK_LABELS:
    group = f'OCSDCC-S{args.stage}-{STAGE_NAMES[args.stage]}-{task_label}'
    runs = list(api.runs(
        f'{args.entity}/{args.project}', filters={'group': group}))
    runs = [run for run in runs if int(run.config.get('seed', -1)) in (5, 6)]
    baselines = list(api.runs(
        f'{args.entity}/{args.project}',
        filters={'group': BASELINE_GROUPS[task_label]}))
    baselines = [run for run in baselines
                 if int(run.config.get('seed', -1)) in (5, 6)]
    if len(runs) < 2 or len(baselines) < 2:
      print(f'{task_label}: BLOCKED (need two stage and two baseline seeds)')
      all_pass = False
      continue

    auc = _mean_success(runs, 'auc', max_points=20)
    baseline_auc = _mean_success(baselines, 'auc', max_points=20)
    retention = _mean_success(runs, 'retention', max_points=20)
    spearman = _mean_latest(runs, SAME_STATE_SPEARMAN_KEY)
    fixed_std = _mean_latest(runs, FIXED_STATE_ACTION_STD_KEY)
    shuffle_retention = _mean_latest(runs, ACTION_SHUFFLE_RETENTION_KEY)

    gates = {
        'success_auc_gain>=0.05': auc - baseline_auc >= 0.05,
        'late_retention>=0.80': retention >= 0.80,
        'same_state_spearman>=0.30': spearman >= 0.30,
        'fixed_state_action_std>=0.001': fixed_std >= 0.001,
        'action_shuffle_retention<=0.50': shuffle_retention <= 0.50,
    }
    task_pass = all(gates.values())
    all_pass &= task_pass
    print(
        f'{task_label}: {"PASS" if task_pass else "FAIL"}; '
        f'AUC={auc:.3f}, CRTR={baseline_auc:.3f}, '
        f'late/peak={retention:.3f}, rho={spearman:.3f}, '
        f'fixed_std={fixed_std:.4g}, shuffle_ret={shuffle_retention:.3f}')
    for name, passed in gates.items():
      print(f'  [{"x" if passed else " "}] {name}')

  raise SystemExit(0 if all_pass else 1)


if __name__ == '__main__':
  main()
