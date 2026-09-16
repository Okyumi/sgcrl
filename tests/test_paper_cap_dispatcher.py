#!/usr/bin/env python3
"""Checks for GPU-cap-safe leftover dispatch."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def _load():
  spec = importlib.util.spec_from_file_location(
      'paper_cap_dispatcher',
      REPO_ROOT / 'scripts' / 'paper_cap_dispatcher.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


disp = _load()


def test_parse_array_ranges():
  assert disp.parse_array_ids('17898476_[0-5]') == set(range(6))
  assert disp.parse_array_ids('17897986_[0,2,5]') == {0, 2, 5}
  assert disp.parse_array_ids('17898476_3') == {3}
  assert disp.parse_array_ids('17898476_[10-16]') == set(range(10, 17))


def test_gpu_task_count_sums_array_sizes():
  jobs = [
      {'n': 6, 'name': 'paper_fs'},
      {'n': 3, 'name': 'paper_rest'},
  ]
  assert disp.gpu_task_count(jobs) == 9


def test_launcher_is_cpu_only():
  launcher = (REPO_ROOT / 'DRAFT_paper_cap_dispatcher.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --partition=cs' in launcher
  assert 'gres=gpu' not in launcher
  assert 'paper_cap_dispatcher.py --loop' in launcher


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'paper cap dispatcher tests passed ({len(tests)})')
