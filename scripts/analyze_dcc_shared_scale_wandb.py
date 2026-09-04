#!/usr/bin/env python3
"""Download and summarize the fixed-alpha DCC Task-5 branch experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import wandb


GROUP = 'DCC-SHARED-SCALE-TASK5-BRANCH-1M'
EVAL_KEYS = ('evaluator/env_steps', 'evaluator/success_rate')
LEARNER_KEYS = (
    'learner/env_steps',
    'learner/decomp/shared_score_fraction',
    'learner/decomp/scaled_shared_to_task_norm',
    'learner/decomp/shared_task_cosine',
    'learner/decomp/shared_goal_score_abs',
    'learner/decomp/task_goal_score_abs',
    'learner/decomp/shared_norm',
    'learner/decomp/task_norm',
    'learner/categorical_accuracy',
    'learner/binary_accuracy',
    'learner/decomp/L_dyn',
    'learner/actor_loss',
    'learner/entropy_mean',
)


def _history(run, keys):
  frame = run.history(keys=list(keys), samples=10_000, pandas=True)
  if frame.empty:
    raise RuntimeError(f'{run.id} has no history for {keys}')
  return frame.sort_values(keys[0]).drop_duplicates(keys[0], keep='last')


def _at_step(frame, step, value_key):
  index = int(np.argmin(np.abs(frame[EVAL_KEYS[0]].to_numpy() - step)))
  return float(frame.iloc[index][value_key])


def _first_crossing(frame, threshold):
  passed = frame[frame[EVAL_KEYS[1]] >= threshold]
  if passed.empty:
    return np.nan
  return float(passed.iloc[0][EVAL_KEYS[0]])


def _tail_mean(frame, key, fraction=0.1):
  count = max(1, int(np.ceil(len(frame) * fraction)))
  return float(frame[key].tail(count).mean())


def summarize_run(run):
  config = run.config
  evaluation = _history(run, EVAL_KEYS)
  learner = _history(run, LEARNER_KEYS)
  steps = evaluation[EVAL_KEYS[0]].to_numpy(dtype=float)
  success = evaluation[EVAL_KEYS[1]].to_numpy(dtype=float)
  if steps[0] > 0:
    steps = np.concatenate([[0.0], steps])
    success = np.concatenate([[0.0], success])
  max_step = float(steps[-1])
  auc = float(np.trapezoid(success, steps) / max(max_step, 1.0))
  row = {
      'run_id': run.id,
      'seed': int(config['seed']),
      'alpha': float(config['shared_repr_scale']),
      'state': run.state,
      'eval_points': len(evaluation),
      'max_eval_step': max_step,
      'success_250k': _at_step(evaluation, 250_000, EVAL_KEYS[1]),
      'success_500k': _at_step(evaluation, 500_000, EVAL_KEYS[1]),
      'success_750k': _at_step(evaluation, 750_000, EVAL_KEYS[1]),
      'success_1m': _at_step(evaluation, 1_000_000, EVAL_KEYS[1]),
      'success_best': float(evaluation[EVAL_KEYS[1]].max()),
      'success_final3': float(evaluation[EVAL_KEYS[1]].tail(3).mean()),
      'success_auc': auc,
      'steps_to_20': _first_crossing(evaluation, 0.2),
      'steps_to_40': _first_crossing(evaluation, 0.4),
  }
  for key in LEARNER_KEYS[1:]:
    row[key.removeprefix('learner/').replace('/', '_') + '_tail'] = (
        _tail_mean(learner, key))
  return row, evaluation, learner


def _corr(x, y):
  mask = np.isfinite(x) & np.isfinite(y)
  x = np.asarray(x)[mask]
  y = np.asarray(y)[mask]
  if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
    return {'n': len(x), 'pearson_r': None, 'pearson_p': None,
            'spearman_r': None, 'spearman_p': None}
  pearson = stats.pearsonr(x, y)
  spearman = stats.spearmanr(x, y)
  return {
      'n': len(x),
      'pearson_r': float(pearson.statistic),
      'pearson_p': float(pearson.pvalue),
      'spearman_r': float(spearman.statistic),
      'spearman_p': float(spearman.pvalue),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', default='nyuad_mmvc/continual-contrastive-rl')
  parser.add_argument('--group', default=GROUP)
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()

  api = wandb.Api(timeout=90)
  runs = list(api.runs(args.path, filters={'group': args.group}))
  runs = [run for run in runs if run.config.get('task_id') == 5]
  expected = {(alpha, seed) for alpha in (0, .25, .5, .75, 1, 1.5)
              for seed in (5, 6, 7)}
  found = {(float(run.config['shared_repr_scale']), int(run.config['seed']))
           for run in runs}
  if found != expected:
    raise RuntimeError(
        f'Expected 18 alpha/seed cells; missing={sorted(expected-found)}, '
        f'extra={sorted(found-expected)}')
  if any(run.state != 'finished' for run in runs):
    raise RuntimeError('Not all matching runs are finished.')

  args.output_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  eval_rows = []
  learner_rows = []
  with ThreadPoolExecutor(max_workers=6) as executor:
    summaries = list(executor.map(summarize_run, runs))
  for run, (row, evaluation, learner) in zip(runs, summaries):
    rows.append(row)
    evaluation = evaluation.assign(
        run_id=run.id, seed=row['seed'], alpha=row['alpha'])
    learner = learner.assign(
        run_id=run.id, seed=row['seed'], alpha=row['alpha'])
    eval_rows.append(evaluation)
    learner_rows.append(learner)

  per_run = pd.DataFrame(rows).sort_values(['alpha', 'seed'])
  numeric = [column for column in per_run.columns
             if column not in ('run_id', 'seed', 'alpha', 'state')]
  aggregate = per_run.groupby('alpha')[numeric].agg(['mean', 'std'])
  aggregate.columns = [f'{name}_{stat}' for name, stat in aggregate.columns]
  aggregate = aggregate.reset_index()

  outcomes = ('success_best', 'success_final3', 'success_auc',
              'success_250k', 'success_500k', 'success_750k', 'success_1m')
  mechanisms = (
      'decomp_shared_score_fraction_tail',
      'decomp_scaled_shared_to_task_norm_tail',
      'decomp_shared_task_cosine_tail',
  )
  correlations = {}
  for metric in mechanisms + outcomes:
    correlations[f'alpha_vs_{metric}'] = _corr(
        per_run['alpha'].to_numpy(), per_run[metric].to_numpy())
  for mechanism in mechanisms:
    for outcome in outcomes[:3]:
      correlations[f'{mechanism}_vs_{outcome}'] = _corr(
          per_run[mechanism].to_numpy(), per_run[outcome].to_numpy())

  # Remove seed-level offsets before testing whether alpha and outcome move
  # together within the matched checkpoint branches.
  centered = per_run.copy()
  for metric in mechanisms + outcomes:
    centered[metric] -= centered.groupby('seed')[metric].transform('mean')
  for metric in mechanisms + outcomes:
    correlations[f'within_seed_alpha_vs_{metric}'] = _corr(
        centered['alpha'].to_numpy(), centered[metric].to_numpy())

  # Alpha=0 removes the shared branch entirely. Test it separately from the
  # question of whether different non-zero scales form a useful continuum.
  nonzero = per_run[per_run['alpha'] > 0]
  comparisons = {'alpha_1_minus_alpha_0': {}, 'nonzero_alpha_tests': {}}
  for metric in outcomes:
    pivot = per_run.pivot(index='seed', columns='alpha', values=metric)
    delta = pivot[1.0] - pivot[0.0]
    comparisons['alpha_1_minus_alpha_0'][metric] = {
        'by_seed': {str(seed): float(value)
                    for seed, value in delta.items()},
        'mean': float(delta.mean()),
        'all_seeds_positive': bool((delta > 0).all()),
    }
    samples = [pivot[alpha] for alpha in (.25, .5, .75, 1.0, 1.5)]
    friedman = stats.friedmanchisquare(*samples)
    comparisons['nonzero_alpha_tests'][metric] = {
        'alpha_correlation': _corr(
            nonzero['alpha'].to_numpy(), nonzero[metric].to_numpy()),
        'friedman_statistic': float(friedman.statistic),
        'friedman_p': float(friedman.pvalue),
    }
  for metric in mechanisms + (
      'decomp_shared_norm_tail', 'decomp_task_norm_tail'):
    comparisons['nonzero_alpha_tests'][metric] = {
        'alpha_correlation': _corr(
            nonzero['alpha'].to_numpy(), nonzero[metric].to_numpy())
    }

  per_run.to_csv(args.output_dir / 'per_run.csv', index=False)
  aggregate.to_csv(args.output_dir / 'by_alpha.csv', index=False)
  pd.concat(eval_rows, ignore_index=True).to_csv(
      args.output_dir / 'evaluation_history.csv', index=False)
  pd.concat(learner_rows, ignore_index=True).to_csv(
      args.output_dir / 'learner_history.csv', index=False)
  (args.output_dir / 'correlations.json').write_text(
      json.dumps(correlations, indent=2, sort_keys=True))
  (args.output_dir / 'comparisons.json').write_text(
      json.dumps(comparisons, indent=2, sort_keys=True))
  print(aggregate.to_string(index=False))
  print(json.dumps(correlations, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
