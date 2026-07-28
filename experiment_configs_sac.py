#!/usr/bin/env python3
"""Enumerate experiment configurations for the continual SAC baseline grid.

SAC counterpart of ``experiment_configs.py``, consumed by ``draft_sac.sh`` in
exactly the same way: the launcher asks for ``--total`` to size the SLURM
array, then ``eval``s the ``KEY=VALUE`` lines from ``--setting N``.

The default grid is the two diagonal cells the collaborator ran, over three
seeds -- R/R (fully independent per task) and P/P (both actor and critic carry
over) -- i.e. 6 runs::

    (actor_mode, critic_mode) in {('reset','reset'), ('persistent','persistent')}
    seed in {1, 2, 3}

Add rows to ``CELLS`` for anything outside the Cartesian product (a different
reward shape, a wider network, a single task, ...). Cell dicts win over
``EXTRA_OVERRIDES``, which wins over the shell defaults in ``draft_sac.sh``.

Usage:
  python experiment_configs_sac.py --total          # count (sizes the array)
  python experiment_configs_sac.py --setting 0      # KEY=VALUE for one run
  python experiment_configs_sac.py --list           # human-readable table

The collaborator's original CLI is also accepted so their notes keep working:
``--num-variants`` / ``--num-seeds`` print the grid dimensions and
``--config V S`` prints the cell at ``(variant, seed)``.
"""
import argparse
import itertools
import sys

# =====================================================================
# Cartesian grid.
# =====================================================================

# (actor_mode, critic_mode) pairs. R/R isolates per-task learning; P/P is the
# transfer condition. The 3rd axis of the CRL table ('cka') is deliberately
# absent: the SAC baseline exists to measure the critic, not the pool.
ACTOR_CRITIC_PAIRS = [
    ('reset', 'reset'),
    ('persistent', 'persistent'),
]

SEEDS = [1, 2, 3]

# Applied to every Cartesian cell. Example: {'steps_per_task': 4_000_000}.
EXTRA_OVERRIDES: dict = {}

# =====================================================================
# Explicit cells -- one dict per run, appended after the Cartesian grid.
# =====================================================================

CELLS: list = [
    # ---- reward-shape ablation (+1/0 instead of the 0/-1 step penalty) ----
    # Note this changes the checkpoint path key, so it cannot resume from or
    # collide with a step-penalty run at the same seed.
    # {'actor_mode': 'reset', 'critic_mode': 'reset', 'seed': 1,
    #  'step_penalty_reward': False},

    # ---- goal-threshold ablation for the looser tasks --------------------
    # {'actor_mode': 'reset', 'critic_mode': 'reset', 'seed': 1,
    #  'her_reward_threshold': 0.1},
]

# Every emitted config must carry these keys; draft_sac.sh assumes them.
_REQUIRED_KEYS = ('actor_mode', 'critic_mode', 'seed')


def build_configs():
  """Return a list of dicts, one per run: Cartesian grid then ``CELLS``."""
  configs: list = []

  for (actor_mode, critic_mode), seed in itertools.product(
      ACTOR_CRITIC_PAIRS, SEEDS):
    cfg = {'actor_mode': actor_mode, 'critic_mode': critic_mode, 'seed': seed}
    cfg.update(EXTRA_OVERRIDES)
    configs.append(cfg)

  for i, cell in enumerate(CELLS):
    missing = [k for k in _REQUIRED_KEYS if k not in cell]
    if missing:
      raise ValueError(
          f'CELLS[{i}] is missing required keys {missing}: {cell!r}')
    cfg = dict(EXTRA_OVERRIDES)
    cfg.update(cell)
    configs.append(cfg)

  return configs


def _emit(cfg):
  """Print one config as shell-sourceable ``KEY=VALUE`` lines."""
  for key, value in cfg.items():
    if isinstance(value, bool):
      # 'true' / 'false' so the launcher's `if [ "$VAR" = "true" ]` works.
      value = 'true' if value else 'false'
    print(f'{key.upper()}={value}')


def main():
  parser = argparse.ArgumentParser(
      description='Enumerate continual SAC experiment configurations.')
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--setting', type=int,
                     help='Print the config at this index (0-based).')
  group.add_argument('--total', action='store_true',
                     help='Print the total number of configs.')
  group.add_argument('--list', action='store_true',
                     help='Print all configs as a table.')
  # Collaborator's original CLI, kept so their notes still work.
  group.add_argument('--num-variants', action='store_true',
                     help='Print the number of (actor, critic) pairs.')
  group.add_argument('--num-seeds', action='store_true',
                     help='Print the number of seeds.')
  group.add_argument('--config', nargs=2, type=int, metavar=('VARIANT', 'SEED'),
                     help='Print the config at (variant index, seed index).')
  args = parser.parse_args()

  configs = build_configs()

  if args.total:
    print(len(configs))
    return

  if args.num_variants:
    print(len(ACTOR_CRITIC_PAIRS))
    return

  if args.num_seeds:
    print(len(SEEDS))
    return

  if args.config is not None:
    variant, seed_idx = args.config
    if not 0 <= variant < len(ACTOR_CRITIC_PAIRS):
      print(f'ERROR: variant {variant} out of range '
            f'[0, {len(ACTOR_CRITIC_PAIRS) - 1}]', file=sys.stderr)
      sys.exit(1)
    if not 0 <= seed_idx < len(SEEDS):
      print(f'ERROR: seed index {seed_idx} out of range '
            f'[0, {len(SEEDS) - 1}]', file=sys.stderr)
      sys.exit(1)
    _emit(configs[variant * len(SEEDS) + seed_idx])
    return

  if args.list:
    print(f'Total: {len(configs)} configurations')
    print(f'  Cartesian: {len(ACTOR_CRITIC_PAIRS) * len(SEEDS)} '
          f'(pairs={ACTOR_CRITIC_PAIRS}, seeds={SEEDS})')
    print(f'  Explicit cells: {len(CELLS)}')
    if EXTRA_OVERRIDES:
      print(f'  EXTRA_OVERRIDES applied to all: {EXTRA_OVERRIDES}')
    print('')
    header = '  '.join([f'{"idx":>4}', f'{"actor":<11}', f'{"critic":<11}',
                        f'{"seed":>4}', f'{"steppen":>8}', f'{"tau":>6}'])
    print(header)
    print('-' * len(header))
    for i, c in enumerate(configs):
      print('  '.join([
          f'{i:>4}',
          f'{c.get("actor_mode", "-"):<11}',
          f'{c.get("critic_mode", "-"):<11}',
          f'{c.get("seed", "-"):>4}',
          f'{c.get("step_penalty_reward", "-")!s:>8}',
          f'{c.get("her_reward_threshold", "-")!s:>6}',
      ]))
    return

  idx = args.setting
  if not 0 <= idx < len(configs):
    print(f'ERROR: setting {idx} out of range [0, {len(configs) - 1}]',
          file=sys.stderr)
    sys.exit(1)
  _emit(configs[idx])


if __name__ == '__main__':
  main()
