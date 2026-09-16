#!/usr/bin/env python3
"""Remaining 10-seed fill behind the breadth-first paper first-seed wave.

Same native_info checkpoints as the 9-baseline rerun. Excludes the 22
first-seed runs so those jobs are not doubled. Packs put leftover R/P
task-7 seeds ahead of leftover CKA seeds.
"""
from __future__ import annotations

import argparse

import experiment_configs_paper_9baseline_10seed as baseline
import experiment_configs_paper_first_seeds as first_seeds

SEEDS = baseline.SEEDS
checkpoint_config_key = baseline.checkpoint_config_key
checkpoint_path = baseline.checkpoint_path

# 68 leftover runs: 28 non-CKA then 40 CKA, grouped by current progress.
REMAINING_PACKS = (
    (
        ('reset', 'reset', 8),
        ('reset', 'reset', 9),
        ('reset', 'reset', 10),
        ('reset', 'reset', 11),
    ),
    (
        ('reset', 'reset', 12),
        ('reset', 'persistent', 10),
        ('reset', 'persistent', 11),
        ('reset', 'persistent', 12),
    ),
    (
        ('reset', 'persistent', 13),
        ('reset', 'persistent', 14),
        ('persistent', 'reset', 10),
        ('persistent', 'reset', 11),
    ),
    (
        ('persistent', 'reset', 12),
        ('persistent', 'reset', 13),
        ('persistent', 'reset', 14),
        ('persistent', 'persistent', 8),
    ),
    (
        ('persistent', 'persistent', 9),
        ('persistent', 'persistent', 10),
        ('persistent', 'persistent', 11),
        ('persistent', 'persistent', 12),
    ),
    (
        ('reset', 'reset', 13),
        ('reset', 'reset', 14),
        ('reset', 'persistent', 5),
        ('reset', 'persistent', 6),
    ),
    (
        ('persistent', 'reset', 5),
        ('persistent', 'reset', 6),
        ('persistent', 'persistent', 13),
        ('persistent', 'persistent', 14),
    ),
    (
        ('cka', 'reset', 7),
        ('cka', 'reset', 8),
        ('cka', 'reset', 9),
        ('cka', 'reset', 10),
    ),
    (
        ('cka', 'reset', 11),
        ('cka', 'reset', 12),
        ('cka', 'reset', 13),
        ('cka', 'reset', 14),
    ),
    (
        ('reset', 'cka', 7),
        ('reset', 'cka', 8),
        ('reset', 'cka', 9),
        ('reset', 'cka', 10),
    ),
    (
        ('reset', 'cka', 11),
        ('reset', 'cka', 12),
        ('reset', 'cka', 13),
        ('reset', 'cka', 14),
    ),
    (
        ('persistent', 'cka', 5),
        ('persistent', 'cka', 6),
        ('persistent', 'cka', 9),
        ('persistent', 'cka', 10),
    ),
    (
        ('persistent', 'cka', 11),
        ('persistent', 'cka', 12),
        ('persistent', 'cka', 13),
        ('persistent', 'cka', 14),
    ),
    (
        ('cka', 'persistent', 7),
        ('cka', 'persistent', 8),
        ('cka', 'persistent', 9),
        ('cka', 'persistent', 10),
    ),
    (
        ('cka', 'persistent', 11),
        ('cka', 'persistent', 12),
        ('cka', 'persistent', 13),
        ('cka', 'persistent', 14),
    ),
    (
        ('cka', 'cka', 5),
        ('cka', 'cka', 6),
        ('cka', 'cka', 7),
        ('cka', 'cka', 8),
    ),
    (
        ('cka', 'cka', 9),
        ('cka', 'cka', 10),
        ('cka', 'cka', 11),
        ('cka', 'cka', 12),
    ),
)


def build_configs():
  by_key = {
      (config['actor_mode'], config['critic_mode'], config['seed']): config
      for config in baseline.build_configs()
  }
  first = first_seeds.first_seed_keys()
  configs = []
  seen = set()
  for pack in REMAINING_PACKS:
    for actor_mode, critic_mode, seed in pack:
      key = (actor_mode, critic_mode, seed)
      if key in first:
        raise ValueError(f'remaining pack overlaps first seeds: {key}')
      if key in seen:
        raise ValueError(f'duplicate remaining pack entry {key}')
      seen.add(key)
      config = dict(by_key[key])
      config['num_actors'] = 2
      configs.append(config)
  expected = {
      (config['actor_mode'], config['critic_mode'], config['seed'])
      for config in baseline.build_configs()
  } - first
  if seen != expected:
    raise ValueError(
        f'remaining packs must cover the leftover 68 runs, '
        f'missing {sorted(expected - seen)} extra {sorted(seen - expected)}')
  return configs


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
      print(index, config['wandb_group'], config['seed'])
    return
  if args.setting < 0 or args.setting >= len(configs):
    raise SystemExit(f'ERROR: setting {args.setting} out of range')
  baseline._emit(configs[args.setting])


if __name__ == '__main__':
  main()
