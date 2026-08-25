#!/usr/bin/env python3
"""Aggregate the six Task-5/Task-8 checkpoint-only reevaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED = {
    ('sawyer_handle_press_side', seed) for seed in (5, 6, 7)
} | {
    ('sawyer_window_close', seed) for seed in (5, 6, 7)
}


def summarize(payloads):
  indexed = {}
  for payload in payloads:
    if payload.get('status') != 'finished':
      raise ValueError('Every reevaluation input must have status=finished.')
    key = (payload['env_name'], int(payload['seed']))
    if key in indexed:
      raise ValueError(f'Duplicate reevaluation result for {key}.')
    indexed[key] = payload
  missing = EXPECTED - set(indexed)
  extra = set(indexed) - EXPECTED
  if missing or extra:
    raise ValueError(
        f'Expected Task 5/8 seeds 5/6/7; missing={sorted(missing)}, '
        f'extra={sorted(extra)}.')

  tasks = {}
  for env_name in ('sawyer_handle_press_side', 'sawyer_window_close'):
    task_payloads = [indexed[(env_name, seed)] for seed in (5, 6, 7)]
    legacy = [p['summary']['legacy_success_rate'] for p in task_payloads]
    axis = [p['summary']['task_axis_success_rate'] for p in task_payloads]
    rescued = [p['summary']['axis_rescued_success_rate'] for p in task_payloads]
    tasks[env_name] = {
        'legacy_success_rate_mean': float(np.mean(legacy)),
        'legacy_success_rate_per_seed': legacy,
        'task_axis_success_rate_mean': float(np.mean(axis)),
        'task_axis_success_rate_per_seed': axis,
        'axis_rescued_success_rate_mean': float(np.mean(rescued)),
        'success_rate_gain_mean': float(np.mean(np.asarray(axis) - legacy)),
    }
  any_gain = any(
      task['success_rate_gain_mean'] > 0.0 for task in tasks.values())
  return {
      'status': 'finished',
      'num_checkpoints': len(indexed),
      'tasks': tasks,
      'conclusion': (
          'axis_metric_changes_reported_performance'
          if any_gain else 'axis_metric_does_not_rescue_existing_policies'),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputs', nargs='+')
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  payloads = [json.loads(Path(path).read_text(encoding='utf-8'))
              for path in args.inputs]
  result = summarize(payloads)
  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(
      json.dumps(result, indent=2, sort_keys=True) + '\n',
      encoding='utf-8')
  for env_name, task in result['tasks'].items():
    print(
        f'{env_name}: legacy={task["legacy_success_rate_mean"]:.3f}, '
        f'axis={task["task_axis_success_rate_mean"]:.3f}, '
        f'gain={task["success_rate_gain_mean"]:+.3f}')
  print(f'Conclusion: {result["conclusion"]}')
  print(f'Results written to {output_path}')


if __name__ == '__main__':
  main()
