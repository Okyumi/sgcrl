#!/usr/bin/env python3
"""Dependency-light checks for the guarded 110-run paper matrix."""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ['NATIVE_SUCCESS_11VARIANT_PROMOTED'] = 'true'

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_native_success_11variant_10seed as configs


def test_exact_variant_and_seed_matrix():
  runs = configs.build_configs()
  assert len(configs.VARIANTS) == 11
  assert len(configs.SEEDS) == 10
  assert len(runs) == 110
  assert set(configs.SEEDS) == set(range(5, 15))


def test_nine_classic_cells_plus_two_dcc_cells():
  classic = configs.VARIANTS[:9]
  assert {(a, c) for _, a, c, _ in classic} == {
      (a, c) for a in configs.BASE_MODES for c in configs.BASE_MODES
  }
  assert configs.VARIANTS[9:] == (
      ('dcc-dyn-on', 'reset', 'decomposed', 1.0),
      ('dcc-dyn-off', 'reset', 'decomposed', 0.0),
  )


def test_full_runs_use_corrected_native_wrapper():
  for run in configs.build_configs():
    assert run['sawyer_success_mode'] == 'native_info'
    assert run['goal_conditioning_mode'] == 'full_state'
    assert run['num_tasks'] == 10
    assert run['steps_per_task'] == 8_000_000
    assert run['actor_auto_reset'] is False


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'native-success 11-variant config tests passed ({len(tests)})')
