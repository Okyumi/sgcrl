#!/usr/bin/env python3
"""Evaluate the pre-registered gates for staged counterfactual experiments."""
from __future__ import annotations

import argparse
import math
import statistics


TASKS = {
    'task5': {
        'baseline': 'DCC-intrajectory-negatives-task5',
    },
    'task8': {
        'baseline': 'DCC-intrajectory-negatives-task8',
    },
}


def _group(stage, task, reach_mode):
  if stage == 'A':
    return f'CFR-STAGE-A2-positive-reward-{task}'
  if stage == 'B':
    return f'CFR-STAGE-B-oracle-decomposition-{task}'
  if stage == 'C':
    return f'CFR-STAGE-C-heldout-chunk-ranker-{task}'
  return f'PGC-DCC-{reach_mode.replace("_", "-")}-{task}'


def _finite(values):
  return [float(value) for value in values
          if value is not None and math.isfinite(float(value))]


def _history(run, key, max_points=None):
  values = _finite(row.get(key) for row in run.scan_history(keys=[key]))
  return values if max_points is None else values[:max_points]


def _latest_mean(runs, key):
  values = [history[-1] for run in runs if (history := _history(run, key))]
  return statistics.fmean(values) if values else float('nan')


def _latest_values(runs, key):
  return [history[-1]
          for run in runs if (history := _history(run, key))]


def _latest_min(runs, key):
  values = _latest_values(runs, key)
  return min(values) if values else float('nan')


def _latest_max(runs, key):
  values = _latest_values(runs, key)
  return max(values) if values else float('nan')


def _success_stats(runs, max_points=20):
  aucs = []
  retentions = []
  for run in runs:
    values = _history(run, 'evaluator/success_rate', max_points=max_points)
    if not values:
      continue
    aucs.append(statistics.fmean(values))
    late_count = max(1, len(values) // 5)
    retentions.append(
        statistics.fmean(values[-late_count:]) / max(max(values), 1e-8))
  return (
      statistics.fmean(aucs) if aucs else float('nan'),
      statistics.fmean(retentions) if retentions else float('nan'))


def _check(name, value, predicate):
  passed = math.isfinite(value) and predicate(value)
  print(f'  [{"x" if passed else " "}] {name}: {value:.4f}')
  return passed


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--stage', choices=('A', 'B', 'C', 'D'), required=True)
  parser.add_argument('--reach-mode', choices=('policy', 'scripted_contact'),
                      default='scripted_contact')
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
  for task, metadata in TASKS.items():
    group = _group(args.stage, task, args.reach_mode)
    runs = list(api.runs(
        f'{args.entity}/{args.project}', filters={'group': group}))
    runs = [run for run in runs
            if run.state == 'finished'
            and int(run.config.get('seed', -1)) in (5, 6)]
    print(f'{task} / {group}')
    if len(runs) < 2:
      print('  BLOCKED: need two finished seeds (5 and 6)')
      all_pass = False
      continue

    gates = []
    if args.stage == 'A':
      rank_prefix = 'learner/counterfactual_rank/'
      landscape_prefix = 'learner/action_landscape/'
      gates.extend([
          _check('benchmark success available (worst seed)', _latest_min(
              runs, rank_prefix + 'benchmark_success_available_fraction'),
                 lambda value: value >= 0.99),
          _check('benchmark/proxy agreement (worst seed)', _latest_min(
              runs, rank_prefix + 'success_predicate_agreement'),
                 lambda value: value >= 0.99),
          _check('proxy false-positive rate (worst seed)', _latest_max(
              runs, rank_prefix + 'proxy_false_positive_fraction'),
                 lambda value: value <= 0.01),
          _check('proxy false-negative rate (worst seed)', _latest_max(
              runs, rank_prefix + 'proxy_false_negative_fraction'),
                 lambda value: value <= 0.01),
          _check('held-out metrics recorded for both seeds', float(len(
              _latest_values(
                  runs, rank_prefix + 'heldout_post/fixed_state_score_std'))),
                 lambda value: value == len(runs)),
          _check('aligned repeat is five (minimum)', _latest_min(
              runs, landscape_prefix + 'action_repeat'),
                 lambda value: abs(value - 5.0) < 1e-6),
          _check('aligned repeat is five (maximum)', _latest_max(
              runs, landscape_prefix + 'action_repeat'),
                 lambda value: abs(value - 5.0) < 1e-6),
          _check('aligned causal metric recorded for both seeds', float(len(
              _latest_values(
                  runs, landscape_prefix
                        + 'aligned_score_vs_progress_spearman'))),
                 lambda value: value == len(runs)),
      ])
    elif args.stage == 'B':
      prefix = 'learner/oracle/scripted_contact/repeat5/'
      gates.extend([
          _check('scripted-contact oracle success', _latest_mean(
              runs, prefix + 'best_success_fraction'),
                 lambda value: value >= 0.50),
          _check('oracle gain over random candidates', _latest_mean(
              runs, prefix + 'success_gain'),
                 lambda value: value >= 0.20),
      ])
      print('  policy-repeat5 reach:', _latest_mean(
          runs, 'learner/oracle/policy/repeat5/near_interaction_fraction'))
    elif args.stage == 'C':
      prefix = 'learner/counterfactual_rank/heldout_post/'
      gates.extend([
          _check('held-out outcome Spearman', _latest_mean(
              runs, prefix + 'score_vs_outcome_spearman'),
                 lambda value: value >= 0.30),
          _check('held-out pairwise accuracy', _latest_mean(
              runs, prefix + 'pairwise_accuracy'),
                 lambda value: value >= 0.65),
          _check('held-out action-permutation drop', _latest_mean(
              runs, prefix + 'action_permutation_spearman_drop'),
                 lambda value: value >= 0.20),
          _check('independent repeat-aligned Spearman', _latest_mean(
              runs, 'learner/action_landscape/'
                    'aligned_score_vs_progress_spearman'),
                 lambda value: value >= 0.30),
      ])
    else:
      auc, retention = _success_stats(runs)
      baselines = list(api.runs(
          f'{args.entity}/{args.project}',
          filters={'group': metadata['baseline']}))
      baselines = [run for run in baselines
                   if run.state == 'finished'
                   and int(run.config.get('seed', -1)) in (5, 6)]
      baseline_auc, _ = _success_stats(baselines)
      gates.extend([
          _check('success AUC gain', auc - baseline_auc,
                 lambda value: value >= 0.05),
          _check('late/peak retention', retention,
                 lambda value: value >= 0.80),
          _check('live contact episode reach rate', _latest_mean(
              runs, 'evaluator/phase_control/contact_episode_reach_rate'),
                 lambda value: value >= 0.50),
          _check('held-out chunk Spearman', _latest_mean(
              runs, 'learner/counterfactual_rank/heldout_post/'
                    'score_vs_outcome_spearman'),
                 lambda value: value >= 0.30),
          _check('independent aligned Spearman', _latest_mean(
              runs, 'learner/action_landscape/'
                    'aligned_score_vs_progress_spearman'),
                 lambda value: value >= 0.30),
      ])
      print(f'  AUC={auc:.4f}, matched CRTR={baseline_auc:.4f}')
    task_pass = all(gates)
    all_pass &= task_pass
    print(f'  RESULT: {"PASS" if task_pass else "FAIL"}')

  raise SystemExit(0 if all_pass else 1)


if __name__ == '__main__':
  main()
