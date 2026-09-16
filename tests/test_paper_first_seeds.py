#!/usr/bin/env python3
"""Checks for the breadth-first paper first-seed wave."""
from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_paper_9baseline_10seed as baseline
import experiment_configs_paper_first_seeds as configs


def _load_status():
  spec = importlib.util.spec_from_file_location(
      'paper_first_seeds_status',
      REPO_ROOT / 'scripts' / 'paper_first_seeds_status.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


status = _load_status()


def test_covers_every_variant_with_at_least_two_seeds():
  runs = configs.build_configs()
  assert len(runs) == 22
  counts = Counter((run['actor_mode'], run['critic_mode']) for run in runs)
  assert set(counts) == set(configs.ALL_CELLS)
  for cell, n_seeds in counts.items():
    if cell[0] != 'cka' and cell[1] != 'cka':
      assert n_seeds == 3, cell
    else:
      assert n_seeds == 2, cell


def test_checkpoint_paths_match_nine_baseline_rerun():
  for run in configs.build_configs():
    assert configs.checkpoint_path(
        '/tmp/ckpts', run['actor_mode'], run['critic_mode'], run['seed'], 7
    ) == baseline.checkpoint_path(
        '/tmp/ckpts', run['actor_mode'], run['critic_mode'], run['seed'], 7)
    assert run['sawyer_success_mode'] == 'native_info'
    assert run['num_actors'] == 2
    assert run['wandb_group'].startswith('PAPER-9BASELINE-10SEED-')


def test_furthest_seeds_are_selected():
  keys = {(run['actor_mode'], run['critic_mode'], run['seed'])
          for run in configs.build_configs()}
  assert ('reset', 'reset', 5) in keys
  assert ('cka', 'reset', 5) in keys
  assert ('cka', 'cka', 13) in keys
  assert ('cka', 'cka', 14) in keys
  assert ('reset', 'reset', 13) not in keys
  assert ('cka', 'cka', 5) not in keys


def test_status_incomplete_ids_cover_six_arrays():
  with tempfile.TemporaryDirectory() as tmp:
    ids = status.incomplete_array_ids(tasks_per_gpu=2, checkpoint_dir=tmp)
    assert ids == list(range(11))


def test_launcher_fits_six_gpus_and_reuses_baseline_checkpoints():
  launcher = (REPO_ROOT / 'DRAFT_paper_first_seeds.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-10' in launcher
  assert 'job-name=paper_fs' in launcher
  assert 'CONFIG_LIMIT=22' in launcher
  assert 'TASKS_PER_GPU="${TASKS_PER_GPU:-2}"' in launcher
  assert 'XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"' in launcher
  assert 'paper_9baseline_checkpoints/10seed' in launcher
  assert 'experiment_configs_paper_first_seeds.py' in launcher
  assert 'PAPER_FIRST_SEEDS_MAX_CHAIN:-0' in launcher


def test_config_emits_num_actors():
  emitted = subprocess.run(
      [sys.executable, 'experiment_configs_paper_first_seeds.py',
       '--setting', '0'],
      cwd=REPO_ROOT, capture_output=True, text=True, check=True)
  assert 'NUM_ACTORS=2' in emitted.stdout
  assert 'SAWYER_SUCCESS_MODE=native_info' in emitted.stdout


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'paper first-seeds tests passed ({len(tests)})')
