#!/usr/bin/env python3
"""Enumerate all experiment configurations for the continual CRL grid.

The 9-configuration ablation grid:
  actor_mode  ∈ {reset, persistent, cka}
  critic_mode ∈ {reset, persistent, cka}

Each configuration is run across multiple seeds.

Usage:
  # Print a specific configuration (for the shell launcher):
  python experiment_configs.py --setting 0

  # Print total number of configurations:
  python experiment_configs.py --total

  # Print all configurations:
  python experiment_configs.py --list
"""
import argparse
import itertools
import json
import sys


# =====================================================================
# Grid definition – edit these to change the experiment sweep
# =====================================================================

ACTOR_MODES  = ['cka']
CRITIC_MODES = ['reset', 'persistent', 'cka']
SEEDS        = [99, 98, 97]

# All other parameters use the defaults from draft_3.sh / draft_4.sh.
# Override here only if a parameter DIFFERS from the shared defaults.
# Each entry is a dict of {flag_name: value} merged on top of the
# shell-script defaults.
EXTRA_OVERRIDES = {}  # e.g., {'steps_per_task': 4000000} to shorten runs


# =====================================================================
# Grid enumeration
# =====================================================================

def build_configs():
    """Return a list of dicts, one per experiment run."""
    configs = []
    for actor_mode, critic_mode in itertools.product(ACTOR_MODES, CRITIC_MODES):
        for seed in SEEDS:
            cfg = {
                'actor_mode':  actor_mode,
                'critic_mode': critic_mode,
                'seed':        seed,
            }
            cfg.update(EXTRA_OVERRIDES)
            configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser(
        description='Enumerate experiment configurations.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--setting', type=int,
                       help='Print the config for this index (0-based).')
    group.add_argument('--total', action='store_true',
                       help='Print total number of configs.')
    group.add_argument('--list', action='store_true',
                       help='Print all configs as a table.')
    args = parser.parse_args()

    configs = build_configs()

    if args.total:
        print(len(configs))
        return

    if args.list:
        print(f'Total: {len(configs)} configurations')
        print(f'{"idx":>4}  {"actor_mode":<12} {"critic_mode":<12} {"seed":>4}')
        print('-' * 40)
        for i, c in enumerate(configs):
            print(f'{i:4d}  {c["actor_mode"]:<12} {c["critic_mode"]:<12} {c["seed"]:4d}')
        return

    # --setting N: output key=value pairs (shell-sourceable)
    idx = args.setting
    if idx < 0 or idx >= len(configs):
        print(f'ERROR: setting {idx} out of range [0, {len(configs) - 1}]',
              file=sys.stderr)
        sys.exit(1)

    cfg = configs[idx]
    for key, value in cfg.items():
        # Uppercase keys for shell variables
        print(f'{key.upper()}={value}')


if __name__ == '__main__':
    main()
