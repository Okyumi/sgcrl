#!/usr/bin/env python3
"""Apply the pre-registered W&B gate to the 1M goal-contract ablation."""
from __future__ import annotations

import argparse
import math
import statistics


TASKS = ('task5', 'task8')
MODES = ('full-state', 'success-mechanism')


def _group(task: str, mode: str) -> str:
  return f'GOAL-VALIDITY-V1-{mode}-{task}'


def _series(run, keys):
  rows = []
  for row in run.scan_history(keys=list(keys)):
    value = row.get('evaluator/success_rate')
    if value is None or not math.isfinite(float(value)):
      continue
    step = row.get('evaluator/env_steps')
    rows.append((float(step) if step is not None else float(len(rows)),
                 float(value)))
  return rows


def _auc(rows):
  if not rows:
    return float('nan')
  if len(rows) == 1:
    return rows[0][1]
  area = 0.0
  for (x0, y0), (x1, y1) in zip(rows[:-1], rows[1:]):
    area += max(0.0, x1 - x0) * (y0 + y1) / 2.0
  horizon = rows[-1][0] - rows[0][0]
  return area / horizon if horizon > 0 else statistics.fmean(
      value for _, value in rows)


def _late_retention(rows):
  values = [value for _, value in rows]
  if not values:
    return float('nan')
  count = max(1, len(values) // 5)
  return statistics.fmean(values[-count:]) / max(max(values), 1e-8)


def _latest(run, key):
  values = [float(row[key]) for row in run.scan_history(keys=[key])
            if row.get(key) is not None
            and math.isfinite(float(row[key]))]
  return values[-1] if values else float('nan')


def main():
  parser = argparse.ArgumentParser()
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
  for task in TASKS:
    by_mode = {}
    for mode in MODES:
      runs = list(api.runs(
          f'{args.entity}/{args.project}',
          filters={'group': _group(task, mode)}))
      runs = [run for run in runs
              if run.state == 'finished'
              and int(run.config.get('seed', -1)) in (5, 6)]
      by_mode[mode] = {int(run.config['seed']): run for run in runs}

    print(task)
    if any(set(by_mode[mode]) != {5, 6} for mode in MODES):
      print('  BLOCKED: need finished seeds 5 and 6 for both goal modes')
      all_pass = False
      continue

    full_auc = {}
    mechanism_auc = {}
    mechanism_retention = {}
    diagnostic_fractions = []
    for seed in (5, 6):
      full_rows = _series(by_mode['full-state'][seed], (
          'evaluator/success_rate', 'evaluator/env_steps'))
      mechanism_rows = _series(by_mode['success-mechanism'][seed], (
          'evaluator/success_rate', 'evaluator/env_steps'))
      full_auc[seed] = _auc(full_rows)
      mechanism_auc[seed] = _auc(mechanism_rows)
      mechanism_retention[seed] = _late_retention(mechanism_rows)
      diagnostic_fractions.append(_latest(
          by_mode['success-mechanism'][seed],
          'runtime/diagnostic_fraction'))

    gains = [mechanism_auc[seed] - full_auc[seed] for seed in (5, 6)]
    mean_gain = statistics.fmean(gains)
    worst_gain = min(gains)
    mean_retention = statistics.fmean(mechanism_retention.values())
    finite_diag = [value for value in diagnostic_fractions
                   if math.isfinite(value)]
    max_diag = max(finite_diag) if finite_diag else float('nan')
    gates = {
        'mean AUC gain >= 0.05': mean_gain >= 0.05,
        'worst paired-seed gain >= -0.05': worst_gain >= -0.05,
        'late/peak retention >= 0.75': mean_retention >= 0.75,
        'serial diagnostic fraction <= 0.01': (
            math.isfinite(max_diag) and max_diag <= 0.01),
    }
    print(f'  full AUC:      {full_auc}')
    print(f'  mechanism AUC: {mechanism_auc}')
    print(f'  paired gains:  {dict(zip((5, 6), gains))}')
    print(f'  mean retention={mean_retention:.4f}, max diagnostic fraction={max_diag:.4f}')
    for name, passed in gates.items():
      print(f'  [{"x" if passed else " "}] {name}')
    task_pass = all(gates.values())
    all_pass &= task_pass
    print(f'  RESULT: {"PASS" if task_pass else "FAIL"}')

  if all_pass:
    print('\nPROMOTE: GOAL_VALIDITY_PROMOTED=true '
          'sbatch DRAFT_goal_semantics_promotion.sh')
  else:
    print('\nDO NOT PROMOTE: the custom full-state goal is not supported as '
          'the primary failure cause under the pre-registered gate.')
  raise SystemExit(0 if all_pass else 1)


if __name__ == '__main__':
  main()
