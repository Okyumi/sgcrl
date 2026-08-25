#!/usr/bin/env python3
"""Aggregate paired wrapper positive-control JSON files across seeds."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any, Sequence


TASK_LABELS = {
    'sawyer_handle_press_side': 'Task 5',
    'sawyer_window_close': 'Task 8',
}
REQUIRED_SEEDS = (5, 6, 7)


def _load(paths: Sequence[str]) -> list[dict[str, Any]]:
  payloads = []
  for path_string in paths:
    path = Path(path_string)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if int(payload.get('audit_version', -1)) != 5:
      raise ValueError(
          f'{path} is not a version-5 native-success positive-control audit; '
          'older proxy/adapter results cannot be promoted.')
    payload['_source'] = str(path)
    payloads.append(payload)
  return payloads


def _result_by_task(payload: dict[str, Any]) -> dict[str, Any]:
  return {result['env_name']: result for result in payload['results']}


def aggregate(
    payloads: Sequence[dict[str, Any]],
    required_seeds: Sequence[int] = REQUIRED_SEEDS,
) -> dict[str, Any]:
  by_seed = {int(payload['seed']): payload for payload in payloads}
  missing = sorted(set(required_seeds) - set(by_seed))
  unexpected = sorted(set(by_seed) - set(required_seeds))
  if missing:
    raise ValueError(f'Missing required audit seeds: {missing}.')
  if unexpected:
    raise ValueError(f'Unexpected audit seeds: {unexpected}.')
  if len(by_seed) != len(payloads):
    raise ValueError('Duplicate seed payloads were supplied.')

  tasks = {}
  for env_name, label in TASK_LABELS.items():
    seed_rows = []
    for seed in required_seeds:
      task_results = _result_by_task(by_seed[seed])
      if env_name not in task_results:
        raise ValueError(f'Seed {seed} has no result for {env_name}.')
      result = task_results[env_name]
      summary = result['summary']
      classification = summary['classification']
      seed_rows.append({
          'seed': seed,
          'decision': classification['decision'],
          'native_success': summary['conditions']['native_official'][
              'native_info_success_mean'],
          'wrapper_native_success': summary['conditions'][
              'wrapper_native_target_policy'][
                  'positive_reward_success_mean'],
          'wrapper_native_info_success': summary['conditions'][
              'wrapper_native_target_policy']['native_info_success_mean'],
          'wrapper_native_axis_success': summary['conditions'][
              'wrapper_native_target_policy']['native_axis_success_mean'],
          'wrapper_native_replay_reward_success': summary['conditions'][
              'wrapper_native_target_replay'][
                  'positive_reward_success_mean'],
          'wrapper_native_replay_info_success': summary['conditions'][
              'wrapper_native_target_replay']['native_info_success_mean'],
          'wrapper_native_replay_axis_success': summary['conditions'][
              'wrapper_native_target_replay']['native_axis_success_mean'],
          'wrapper_fixed_success': summary['conditions'][
              'wrapper_fixed_target_policy'][
                  'positive_reward_success_mean'],
          'wrapper_fixed_info_success': summary['conditions'][
              'wrapper_fixed_target_policy']['native_info_success_mean'],
          'wrapper_fixed_axis_success': summary['conditions'][
              'wrapper_fixed_target_policy']['fixed_axis_success_mean'],
          'fixed_replay_native_endpoint_success': summary['conditions'][
              'wrapper_fixed_target_replay']['native_axis_success_mean'],
          'fixed_replay_fixed_endpoint_success': summary['conditions'][
              'wrapper_fixed_target_replay']['fixed_axis_success_mean'],
          'fixed_native_target_distance': summary['pairing'][
              'fixed_to_native_target_distance_mean'],
          'fixed_native_success_axis_distance': summary['pairing'][
              'fixed_to_native_success_axis_distance_mean'],
          'native_target_pair_error': summary['pairing'][
              'native_target_pair_linf_error_max'],
          'rand_vec_pair_error': summary['pairing'][
              'rand_vec_pair_linf_error_max'],
          'initial_mechanism_pair_error': summary['pairing'][
              'initial_mechanism_pair_linf_error_max'],
          'native_replay_trajectory_error': summary['conditions'][
              'wrapper_native_target_replay'][
                  'trajectory_linf_error_vs_native_max'],
          'native_policy_trajectory_error': summary['conditions'][
              'wrapper_native_target_policy'][
                  'trajectory_linf_error_vs_native_max'],
          'native_policy_input_error': summary['conditions'][
              'wrapper_native_target_policy'][
                  'policy_input_linf_error_vs_native_max'],
          'native_policy_action_error': summary['conditions'][
              'wrapper_native_target_policy'][
                  'action_linf_error_vs_native_max'],
          'fixed_replay_trajectory_error': summary['conditions'][
              'wrapper_fixed_target_replay'][
                  'trajectory_linf_error_vs_native_max'],
          'native_info_axis_mismatch': max(
              summary['conditions'][condition][
                  'native_info_axis_mismatch_fraction_mean']
              for condition in summary['conditions']),
          'wrapper_reward_info_mismatch': max(
              summary['conditions'][condition][
                  'reward_native_info_mismatch_fraction_mean']
              for condition in (
                  'wrapper_native_target_policy',
                  'wrapper_native_target_replay',
                  'wrapper_fixed_target_policy',
                  'wrapper_fixed_target_replay')),
          'wrapper_evaluate_state_fallback_fraction': max(
              summary['conditions'][condition][
                  'evaluate_state_fallback_fraction_mean']
              for condition in (
                  'wrapper_native_target_policy',
                  'wrapper_native_target_replay',
                  'wrapper_fixed_target_policy',
                  'wrapper_fixed_target_replay')),
      })
    decisions = Counter(row['decision'] for row in seed_rows)
    fixed_misalignment_confirmed = (
        decisions.get('fixed_global_target_misaligned', 0)
        == len(required_seeds))
    fixed_target_valid = (
        decisions.get('fixed_target_valid', 0) == len(required_seeds))
    tasks[env_name] = {
        'label': label,
        'seed_results': seed_rows,
        'decision_counts': dict(decisions),
        'fixed_misalignment_confirmed': fixed_misalignment_confirmed,
        'fixed_target_valid': fixed_target_valid,
        'means': {
            key: statistics.fmean(row[key] for row in seed_rows)
            for key in (
                'native_success',
                'wrapper_native_success',
                'wrapper_native_info_success',
                'wrapper_native_axis_success',
                'wrapper_native_replay_reward_success',
                'wrapper_native_replay_info_success',
                'wrapper_native_replay_axis_success',
                'wrapper_fixed_success',
                'wrapper_fixed_info_success',
                'wrapper_fixed_axis_success',
                'fixed_replay_native_endpoint_success',
                'fixed_replay_fixed_endpoint_success',
                'fixed_native_target_distance',
                'fixed_native_success_axis_distance',
            )
        },
        'maxima': {
            key: max(row[key] for row in seed_rows)
            for key in (
                'native_target_pair_error',
                'rand_vec_pair_error',
                'initial_mechanism_pair_error',
                'native_replay_trajectory_error',
                'native_policy_trajectory_error',
                'native_policy_input_error',
                'native_policy_action_error',
                'fixed_replay_trajectory_error',
                'native_info_axis_mismatch',
                'wrapper_reward_info_mismatch',
                'wrapper_evaluate_state_fallback_fraction',
            )
        },
    }

  all_current_wrappers_valid = all(
      task['fixed_target_valid'] for task in tasks.values())
  all_fixed_misalignment_confirmed = all(
      task['fixed_misalignment_confirmed'] for task in tasks.values())
  if all_current_wrappers_valid:
    conclusion = 'current_fixed_targets_valid'
  elif all_fixed_misalignment_confirmed:
    conclusion = 'fixed_global_targets_invalid'
  elif any(
      'audit_metric_inconsistent' in task['decision_counts']
      for task in tasks.values()):
    conclusion = 'audit_metric_inconsistent'
  elif any(
      'native_positive_control_failed' in task['decision_counts']
      for task in tasks.values()):
    conclusion = 'native_or_expert_positive_control_invalid'
  elif any(
      'custom_wrapper_invalid' in task['decision_counts']
      for task in tasks.values()):
    conclusion = 'custom_wrapper_invalid_beyond_fixed_target'
  else:
    conclusion = 'inconclusive'

  return {
      'required_seeds': list(required_seeds),
      'tasks': tasks,
      'all_current_wrappers_valid': all_current_wrappers_valid,
      'all_fixed_misalignment_confirmed':
          all_fixed_misalignment_confirmed,
      'promotion_allowed': all_current_wrappers_valid,
      'conclusion': conclusion,
  }


def _markdown(report: dict[str, Any]) -> str:
  lines = [
      '# Sawyer goal-wrapper positive-control results',
      '',
      f'Conclusion: **{report["conclusion"]}**',
      '',
      '| Task | Native expert | Native wrapper policy R/I/A | '
      'Native wrapper replay R/I/A | Fixed wrapper policy R/I/A | '
      'Fixed replay reaches native/fixed axis | Max info/axis mismatch | '
      'Max policy input/action error | Max reward/info mismatch | Decision |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
  ]
  for task in report['tasks'].values():
    means = task['means']
    decisions = ', '.join(
        f'{name} × {count}'
        for name, count in sorted(task['decision_counts'].items()))
    lines.append(
        f'| {task["label"]} | {means["native_success"]:.3f} | '
        f'{means["wrapper_native_success"]:.3f}/'
        f'{means["wrapper_native_info_success"]:.3f}/'
        f'{means["wrapper_native_axis_success"]:.3f} | '
        f'{means["wrapper_native_replay_reward_success"]:.3f}/'
        f'{means["wrapper_native_replay_info_success"]:.3f}/'
        f'{means["wrapper_native_replay_axis_success"]:.3f} | '
        f'{means["wrapper_fixed_success"]:.3f}/'
        f'{means["wrapper_fixed_info_success"]:.3f}/'
        f'{means["wrapper_fixed_axis_success"]:.3f} | '
        f'{means["fixed_replay_native_endpoint_success"]:.3f}/'
        f'{means["fixed_replay_fixed_endpoint_success"]:.3f} | '
        f'{task["maxima"]["native_info_axis_mismatch"]:.3f} | '
        f'{task["maxima"]["native_policy_input_error"]:.3g}/'
        f'{task["maxima"]["native_policy_action_error"]:.3g} | '
        f'{task["maxima"]["wrapper_reward_info_mismatch"]:.3f} | '
        f'{decisions} |')
  lines.extend([
      '',
      '`R/I/A` denotes wrapper positive-reward success, MetaWorld '
      '`info["success"]`, and the exact task-axis success proxy.',
      '',
      f'Promotion allowed: **{report["promotion_allowed"]}**',
      '',
      'Promotion is allowed only when the historical fixed-target wrapper '
      'passes all three seeds on both tasks. A confirmed fixed-target '
      'misalignment instead requires repairing and revalidating the wrapper '
      'before any algorithm promotion.',
      '',
  ])
  return '\n'.join(lines)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('inputs', nargs='+')
  parser.add_argument(
      '--output',
      default='logs/goal_validity/positive_controls_v5_summary.json')
  parser.add_argument(
      '--markdown-output',
      default='logs/goal_validity/positive_controls_v5_summary.md')
  parser.add_argument('--strict-promotion', action='store_true')
  args = parser.parse_args()

  report = aggregate(_load(args.inputs))
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
  markdown_output = Path(args.markdown_output)
  markdown_output.parent.mkdir(parents=True, exist_ok=True)
  markdown_output.write_text(_markdown(report), encoding='utf-8')
  print(json.dumps({
      'conclusion': report['conclusion'],
      'promotion_allowed': report['promotion_allowed'],
      'output': str(output),
      'markdown_output': str(markdown_output),
  }, indent=2))
  if args.strict_promotion and not report['promotion_allowed']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
