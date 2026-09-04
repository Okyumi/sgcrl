#!/usr/bin/env python3
"""Task-5 sweep with unit-normalized DCC shared/task branch mixing."""
from __future__ import annotations

import argparse
import shlex

from experiment_configs_dcc_shared_scale_task5 import (
    PREFIX_CHECKPOINT_DIR,
    SEEDS,
    SHARED_SCALES,
    build_configs as build_unnormalized_configs,
)


WANDB_GROUP = 'DCC-NORMALIZED-SHARED-SCALE-TASK5-BRANCH-1M'


def build_configs():
  """Reuse the matched Task-0-to-4 prefixes and change only score mixing."""
  configs = []
  for config in build_unnormalized_configs():
    configs.append(config | {
        'shared_repr_normalization': 'unit_mix',
        'wandb_group': WANDB_GROUP,
    })
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
  configs = build_configs()
  if args.total:
    print(len(configs))
    return
  if args.list:
    for index, config in enumerate(configs):
      print(index, config['shared_repr_scale'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
