#!/usr/bin/env python3
"""Checks for the leftover 10-seed paper fill."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_paper_9baseline_10seed as baseline
import experiment_configs_paper_first_seeds as first_seeds
import experiment_configs_paper_remaining_seeds as remaining


def _load_status():
  spec = importlib.util.spec_from_file_location(
      'paper_remaining_seeds_status',
      REPO_ROOT / 'scripts' / 'paper_remaining_seeds_status.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


status = _load_status()


def test_first_plus_remaining_cover_the_ninety_run_matrix():
  first = first_seeds.first_seed_keys()
  rest = {
      (run['actor_mode'], run['critic_mode'], run['seed'])
      for run in remaining.build_configs()
  }
  all_keys = {
      (run['actor_mode'], run['critic_mode'], run['seed'])
      for run in baseline.build_configs()
  }
  assert len(first_seeds.build_configs()) == 22
  assert len(remaining.build_configs()) == 68
  assert not (first & rest)
  assert first | rest == all_keys


def test_remaining_packs_put_non_cka_ahead_of_cka():
  runs = remaining.build_configs()
  assert len(remaining.REMAINING_PACKS) == 17
  assert all(run['actor_mode'] != 'cka' and run['critic_mode'] != 'cka'
             for run in runs[:28])
  assert all(run['actor_mode'] == 'cka' or run['critic_mode'] == 'cka'
             for run in runs[28:])


def test_checkpoint_paths_match_nine_baseline():
  run = remaining.build_configs()[0]
  assert remaining.checkpoint_path(
      '/tmp/ckpts', run['actor_mode'], run['critic_mode'], run['seed'], 7
  ) == baseline.checkpoint_path(
      '/tmp/ckpts', run['actor_mode'], run['critic_mode'], run['seed'], 7)
  assert all(run['num_actors'] == 2 for run in remaining.build_configs())
  emitted = subprocess.run(
      [sys.executable, 'experiment_configs_paper_remaining_seeds.py',
       '--setting', '0'],
      cwd=REPO_ROOT, capture_output=True, text=True, check=True)
  assert 'NUM_ACTORS=2' in emitted.stdout


def test_status_incomplete_ids_cover_seventeen_arrays():
  with tempfile.TemporaryDirectory() as tmp:
    ids = status.incomplete_array_ids(tasks_per_gpu=2, checkpoint_dir=tmp)
    assert ids == list(range(34))


def test_launcher_uses_lower_slurm_priority():
  launcher = (REPO_ROOT / 'DRAFT_paper_remaining_seeds.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-33' in launcher
  assert '#SBATCH --nice=100' in launcher
  assert 'job-name=paper_rest' in launcher
  assert 'CONFIG_LIMIT=68' in launcher
  assert 'TASKS_PER_GPU="${TASKS_PER_GPU:-2}"' in launcher


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'paper remaining-seeds tests passed ({len(tests)})')
