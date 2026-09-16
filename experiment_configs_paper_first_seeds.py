#!/usr/bin/env python3
"""Breadth-first paper seeds: a few furthest-along runs of every 9-cell variant.

Same hyperparameters and checkpoint identity as the 9-baseline × 10-seed
rerun. The first wave finishes 3 seeds of each R/P cell (already at
task 5–7) and 2 seeds of each CKA cell, then a later 10-seed sweep can
fill the rest.
"""
from __future__ import annotations

import argparse
from collections import Counter

import experiment_configs_paper_9baseline_10seed as baseline

SEEDS = baseline.SEEDS
ALL_CELLS = tuple(
    (actor, critic)
    for actor in ('reset', 'persistent', 'cka')
    for critic in ('reset', 'persistent', 'cka')
)

# Arrays 0–5 (2-pack: 0–10) keep furthest-along seeds together so
# task-7 R/P runs are not stuck behind task-0 CKA roommates.
FIRST_SEED_PACKS = (
    (
        ('reset', 'reset', 5),
        ('reset', 'reset', 6),
        ('reset', 'reset', 7),
        ('reset', 'persistent', 7),
    ),
    (
        ('reset', 'persistent', 8),
        ('reset', 'persistent', 9),
        ('persistent', 'reset', 7),
        ('persistent', 'reset', 8),
    ),
    (
        ('persistent', 'reset', 9),
        ('persistent', 'persistent', 5),
        ('persistent', 'persistent', 6),
        ('persistent', 'persistent', 7),
    ),
    (
        ('cka', 'reset', 5),
        ('cka', 'reset', 6),
        ('reset', 'cka', 5),
        ('reset', 'cka', 6),
    ),
    (
        ('persistent', 'cka', 7),
        ('persistent', 'cka', 8),
        ('cka', 'persistent', 5),
        ('cka', 'persistent', 6),
    ),
    (
        ('cka', 'cka', 13),
        ('cka', 'cka', 14),
    ),
)

checkpoint_config_key = baseline.checkpoint_config_key
checkpoint_path = baseline.checkpoint_path


def first_seed_keys():
  return {
      (actor_mode, critic_mode, seed)
      for pack in FIRST_SEED_PACKS
      for actor_mode, critic_mode, seed in pack
  }


def build_configs():
  by_key = {
      (config['actor_mode'], config['critic_mode'], config['seed']): config
      for config in baseline.build_configs()
  }
  configs = []
  seen = set()
  for pack in FIRST_SEED_PACKS:
    for actor_mode, critic_mode, seed in pack:
      key = (actor_mode, critic_mode, seed)
      if key in seen:
        raise ValueError(f'duplicate pack entry {key}')
      seen.add(key)
      config = dict(by_key[key])
      # Two collectors per learner; Torch launchers pack two learners
      # per L40S so the GPU still sees four MuJoCo sims, matching Jubail.
      config['num_actors'] = 2
      configs.append(config)
  counts = Counter((actor, critic) for actor, critic, _ in seen)
  missing = [cell for cell in ALL_CELLS if counts[cell] < 2]
  if missing:
    raise ValueError(f'every variant needs >=2 seeds, missing {missing}')
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
