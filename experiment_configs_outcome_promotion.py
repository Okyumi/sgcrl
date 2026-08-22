#!/usr/bin/env python3
"""Full-horizon promotion cells for one winning 1M falsification stage."""
import argparse
import os
import shlex
import sys

from experiment_configs_outcome_falsification import build_configs


def _winner_stage():
  value = int(os.environ.get('OUTCOME_WINNER_STAGE', '0'))
  if value not in (1, 2, 3):
    raise ValueError(
        'Set OUTCOME_WINNER_STAGE=1, 2, or 3 after applying the 1M gates.')
  return value


def promotion_configs():
  stage = _winner_stage()
  stage_templates = [
      config for config in build_configs()
      if config['falsification_stage'] == stage and config['seed'] == 5
  ]
  configs = []
  for template in stage_templates:
    label = 'task5' if template['single_task'] == \
        'sawyer_handle_press_side' else 'task8'
    for seed in (5, 6, 7):
      config = dict(template)
      config.pop('falsification_stage', None)
      config.update({
          'seed': seed,
          'steps_per_task': 8_000_000,
          'base_steps': 8_000_000,
          'action_landscape_diagnostic_interval_steps': 500_000,
          'wandb_group': f'OCSDCC-PROMOTE-S{stage}-{label}',
      })
      configs.append(config)
  return configs


def _emit(config):
  for key, value in config.items():
    if isinstance(value, bool):
      value = 'true' if value else 'false'
    elif isinstance(value, str):
      value = shlex.quote(value)
    print(f'{key.upper()}={value}')


def main():
  parser = argparse.ArgumentParser()
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--setting', type=int)
  group.add_argument('--total', action='store_true')
  group.add_argument('--list', action='store_true')
  args = parser.parse_args()
  try:
    configs = promotion_configs()
  except ValueError as error:
    print(f'ERROR: {error}', file=sys.stderr)
    raise SystemExit(2)
  if args.total:
    print(len(configs))
    return
  if args.list:
    for index, cfg in enumerate(configs):
      print(index, cfg['wandb_group'], cfg['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    print(f'ERROR: setting {args.setting} out of range', file=sys.stderr)
    raise SystemExit(1)
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
