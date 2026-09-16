#!/usr/bin/env python3
"""Checks for the four-cell RP priority finish."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_paper_9baseline_10seed as baseline
import experiment_configs_paper_rp_priority_10seed as configs


def _load_status():
  spec = importlib.util.spec_from_file_location(
      'paper_rp_priority_status',
      REPO_ROOT / 'scripts' / 'paper_rp_priority_status.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


status = _load_status()


def test_covers_exactly_the_four_non_cka_cells():
  runs = configs.build_configs()
  assert len(runs) == 40
  assert len(configs.PRIORITY_PACKS) == 10
  assert all(len(pack) == 4 for pack in configs.PRIORITY_PACKS)
  keys = {(run['actor_mode'], run['critic_mode'], run['seed']) for run in runs}
  expected = {(actor, critic, seed)
              for actor, critic in configs.RP_CELLS
              for seed in configs.SEEDS}
  assert keys == expected
  assert all(run['actor_mode'] != 'cka' and run['critic_mode'] != 'cka'
             for run in runs)


def test_checkpoint_paths_match_nine_baseline_rerun():
  for run in configs.build_configs():
    assert configs.checkpoint_path(
        '/tmp/ckpts', run['actor_mode'], run['critic_mode'], run['seed'], 7
    ) == baseline.checkpoint_path(
        '/tmp/ckpts', run['actor_mode'], run['critic_mode'], run['seed'], 7)
    assert run['sawyer_success_mode'] == 'native_info'
    assert run['steps_per_task'] == 8_000_000
    assert run['network_width'] == 1024
    assert run['wandb_group'].startswith('PAPER-9BASELINE-10SEED-')


def test_first_eight_packs_are_the_task7_seeds():
  runs = configs.build_configs()
  task7 = {(run['actor_mode'], run['critic_mode'], run['seed'])
           for run in runs[:32]}
  assert ('reset', 'reset', 5) in task7
  assert ('reset', 'reset', 13) not in task7
  assert ('persistent', 'reset', 5) not in task7
  assert ('persistent', 'persistent', 14) not in task7
  lagging = [(run['actor_mode'], run['critic_mode'], run['seed'])
             for run in runs[32:]]
  assert lagging == [
      ('reset', 'reset', 13),
      ('reset', 'reset', 14),
      ('reset', 'persistent', 5),
      ('reset', 'persistent', 6),
      ('persistent', 'reset', 5),
      ('persistent', 'reset', 6),
      ('persistent', 'persistent', 13),
      ('persistent', 'persistent', 14),
  ]


def test_status_incomplete_ids_cover_ten_arrays():
  with tempfile.TemporaryDirectory() as tmp:
    ids = status.incomplete_array_ids(tasks_per_gpu=4, checkpoint_dir=tmp)
    assert ids == list(range(10))


def test_launcher_packs_four_and_reuses_baseline_checkpoints():
  launcher = (REPO_ROOT / 'DRAFT_paper_rp_priority_10seed.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-9' in launcher
  assert 'job-name=paper_rp' in launcher
  assert 'TASKS_PER_GPU="${TASKS_PER_GPU:-4}"' in launcher
  assert 'paper_9baseline_checkpoints/10seed' in launcher
  assert 'experiment_configs_paper_rp_priority_10seed.py' in launcher
  assert '--dependency="afterany:${SLURM_JOB_ID}"' in launcher


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'paper RP-priority 10-seed tests passed ({len(tests)})')
