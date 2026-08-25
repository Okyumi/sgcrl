#!/usr/bin/env python3
"""Enumerate the three paired Sawyer wrapper-audit seeds."""
from __future__ import annotations

import argparse
import shlex


SEEDS = (5, 6, 7)


def build_configs():
  return [{
      'seed': seed,
      'episodes': 50,
      'max_steps': 150,
      'expert_success_min': 0.80,
      'fixed_success_max': 0.20,
      'trajectory_tolerance': 1e-5,
      'target_tolerance': 1e-6,
      'wandb_group': 'GOAL-WRAPPER-POSITIVE-CONTROLS-V3',
      'output': (
          f'logs/goal_validity/positive_controls_v3_seed{seed}.json'),
  } for seed in SEEDS]


def _emit(config):
  for key, value in config.items():
    print(f'{key.upper()}={shlex.quote(str(value))}')


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
      print(index, config['seed'], config['wandb_group'], config['output'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
