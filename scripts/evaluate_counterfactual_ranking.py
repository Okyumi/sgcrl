#!/usr/bin/env python3
"""Evaluate pre-registered promotion gates for counterfactual-ranking DCC."""
from __future__ import annotations

import argparse
import math
import statistics


TASKS = {
    'task5': {
        'group': 'CFRDCC-taskgoal-H100-chunk5-task5',
        'baseline': 'DCC-intrajectory-negatives-task5',
    },
    'task8': {
        'group': 'CFRDCC-taskgoal-H100-chunk5-task8',
        'baseline': 'DCC-intrajectory-negatives-task8',
    },
}
KEYS = {
    'success': 'evaluator/success_rate',
    'causal_rho': (
        'learner/action_landscape/'
        'score_vs_rollout_mechanism_progress_spearman'),
    'rank_rho': (
        'learner/counterfactual_rank/'
        'score_vs_task_progress_spearman'),
    'pair_accuracy': 'learner/counterfactual_rank/pairwise_accuracy',
    'informative': (
        'learner/counterfactual_rank/informative_anchor_fraction'),
    'progress_std': 'learner/counterfactual_rank/candidate_progress_std',
    'score_std': 'learner/counterfactual_rank/fixed_state_score_std',
}


def _finite(values):
  return [float(value) for value in values
          if value is not None and math.isfinite(float(value))]


def _history(run, key, max_points=None):
  values = _finite(row.get(key) for row in run.scan_history(keys=[key]))
  return values if max_points is None else values[:max_points]


def _mean_latest(runs, key):
  values = [history[-1] for run in runs
            if (history := _history(run, key))]
  return statistics.fmean(values) if values else float('nan')


def _success(run, max_points=20):
  values = _history(run, KEYS['success'], max_points=max_points)
  if not values:
    return float('nan'), float('nan')
  late_count = max(1, len(values) // 5)
  return statistics.fmean(values), (
      statistics.fmean(values[-late_count:]) / max(max(values), 1e-8))


def _mean_success(runs, index):
  values = _finite(_success(run)[index] for run in runs)
  return statistics.fmean(values) if values else float('nan')


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
  for task_label, groups in TASKS.items():
    runs = list(api.runs(
        f'{args.entity}/{args.project}', filters={'group': groups['group']}))
    baselines = list(api.runs(
        f'{args.entity}/{args.project}', filters={'group': groups['baseline']}))
    runs = [run for run in runs
            if run.state == 'finished'
            and int(run.config.get('seed', -1)) in (5, 6)]
    baselines = [run for run in baselines
                 if run.state == 'finished'
                 and int(run.config.get('seed', -1)) in (5, 6)]
    if len(runs) < 2 or len(baselines) < 2:
      print(f'{task_label}: BLOCKED (need two finished test/baseline seeds)')
      all_pass = False
      continue

    auc = _mean_success(runs, 0)
    baseline_auc = _mean_success(baselines, 0)
    retention = _mean_success(runs, 1)
    causal_rho = _mean_latest(runs, KEYS['causal_rho'])
    rank_rho = _mean_latest(runs, KEYS['rank_rho'])
    pair_accuracy = _mean_latest(runs, KEYS['pair_accuracy'])
    informative = _mean_latest(runs, KEYS['informative'])
    progress_std = _mean_latest(runs, KEYS['progress_std'])
    score_std = _mean_latest(runs, KEYS['score_std'])
    gates = {
        'success_auc_gain>=0.05': auc - baseline_auc >= 0.05,
        'late_retention>=0.80': retention >= 0.80,
        'causal_rollout_rho>=0.30': causal_rho >= 0.30,
        'rank_batch_rho>=0.30': rank_rho >= 0.30,
        'pairwise_accuracy>=0.65': pair_accuracy >= 0.65,
        'informative_anchors>=0.25': informative >= 0.25,
        'candidate_progress_std>=0.002': progress_std >= 0.002,
        'fixed_state_score_std>=0.001': score_std >= 0.001,
    }
    task_pass = all(gates.values())
    all_pass &= task_pass
    print(
        f'{task_label}: {"PASS" if task_pass else "FAIL"}; '
        f'AUC={auc:.3f}, CRTR={baseline_auc:.3f}, '
        f'late/peak={retention:.3f}, causal_rho={causal_rho:.3f}, '
        f'rank_rho={rank_rho:.3f}, pair_acc={pair_accuracy:.3f}, '
        f'informative={informative:.3f}')
    for name, passed in gates.items():
      print(f'  [{"x" if passed else " "}] {name}')
  raise SystemExit(0 if all_pass else 1)


if __name__ == '__main__':
  main()
