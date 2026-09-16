#!/usr/bin/env python3
"""Matched 1M-step DCC run for Task 8 (window-close), seed 6.

Same algorithm/wrapper as the Jubail handle/push action-advice cells so the
action-informative-mass figure can compare Tasks 5, 8, and 6 without mixing
configs. Videos off.
"""
from __future__ import annotations

import argparse
import shlex

import experiment_configs_jubail_task5_action_advice as base

SEED = 6
WANDB_GROUP = 'TASK58-JUBAIL-ACTION-MASS-WINDOW'


def build_configs():
  config = base._shared(
      SEED, 'sawyer_window_close', inject=False, target_frac=0.0)
  config['variant'] = 'window_measure'
  config['eval_record_video'] = False
  config['wandb_group'] = WANDB_GROUP
  config['run_action_advice_probe'] = False
  config['action_advice_targets'] = ''
  return [config]


def _emit(config):
  skip = {'variant'}
  for key, value in config.items():
    if key in skip:
      continue
    if isinstance(value, bool):
      value = 'true' if value else 'false'
    elif isinstance(value, str):
      value = shlex.quote(value)
    print(f'{key.upper()}={value}')
  print(f"VARIANT={shlex.quote(config['variant'])}")


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
    for i, c in enumerate(configs):
      print(i, c['variant'], c['single_task'], c['seed'])
    return
  _emit(configs[args.setting])


if __name__ == '__main__':
  main()
