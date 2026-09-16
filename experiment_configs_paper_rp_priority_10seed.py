#!/usr/bin/env python3
"""Priority finish of the four non-CKA paper cells (R/R, R/P, P/R, P/P).

Same hyperparameters and checkpoint identity as
``experiment_configs_paper_9baseline_10seed.py``. The 40 runs are packed
by remaining curriculum progress so almost-done seeds share GPUs and are
not blocked by CKA jobs.
"""
from __future__ import annotations

import argparse

import experiment_configs_paper_9baseline_10seed as baseline

SEEDS = baseline.SEEDS
RP_CELLS = (
    ('reset', 'reset'),
    ('reset', 'persistent'),
    ('persistent', 'reset'),
    ('persistent', 'persistent'),
)

# Four-wide packs ordered by remaining work as of 2026-09-16:
# arrays 0-7 finished task 7; array 8 finished task 6; array 9 finished
# task 5. Auto-resume still keys off actor/critic/seed, not array index.
PRIORITY_PACKS = (
    tuple(('reset', 'reset', seed) for seed in range(5, 9)),
    tuple(('reset', 'reset', seed) for seed in range(9, 13)),
    tuple(('reset', 'persistent', seed) for seed in range(7, 11)),
    tuple(('reset', 'persistent', seed) for seed in range(11, 15)),
    tuple(('persistent', 'reset', seed) for seed in range(7, 11)),
    tuple(('persistent', 'reset', seed) for seed in range(11, 15)),
    tuple(('persistent', 'persistent', seed) for seed in range(5, 9)),
    tuple(('persistent', 'persistent', seed) for seed in range(9, 13)),
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
)

checkpoint_config_key = baseline.checkpoint_config_key
checkpoint_path = baseline.checkpoint_path


def build_configs():
  by_key = {
      (config['actor_mode'], config['critic_mode'], config['seed']): config
      for config in baseline.build_configs()
      if (config['actor_mode'], config['critic_mode']) in RP_CELLS
  }
  configs = []
  seen = set()
  for pack in PRIORITY_PACKS:
    for actor_mode, critic_mode, seed in pack:
      key = (actor_mode, critic_mode, seed)
      if key in seen:
        raise ValueError(f'duplicate pack entry {key}')
      seen.add(key)
      configs.append(by_key[key])
  expected = {(actor, critic, seed)
              for actor, critic in RP_CELLS
              for seed in SEEDS}
  if seen != expected:
    raise ValueError(f'RP packs must cover all 40 cells, missing '
                     f'{sorted(expected - seen)} extra {sorted(seen - expected)}')
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
