#!/usr/bin/env python3
"""From-scratch 5-seed gated DCC curriculum."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_full_critic_gate_5seed as cfg


def _load_status():
  import importlib.util
  spec = importlib.util.spec_from_file_location(
      'full_critic_gate_5seed_status',
      REPO_ROOT / 'scripts' / 'full_critic_gate_5seed_status.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


status = _load_status()


def test_five_from_scratch_cells():
  configs = cfg.build_configs()
  assert cfg.SEEDS == tuple(range(5, 10))
  assert len(configs) == 5
  assert [c['seed'] for c in configs] == list(range(5, 10))
  for config in configs:
    assert config['variant'] == 'dcc_gated_success_matching'
    assert config['actor_mode'] == 'reset'
    assert config['critic_mode'] == 'decomposed'
    assert config['start_task'] == 0
    assert config['num_tasks'] == 10
    assert config['steps_per_task'] == 8_000_000
    assert config['eval_every'] == 100_000
    assert config['success_bc_weight'] == 0.1
    assert config['success_bc_label_mode'] == 'episode_sparse_reward'
    assert config['success_bc_critic_gate'] is True
    assert config['success_bc_critic_gate_mode'] == 'local'
    assert config['success_bc_critic_state_noise'] == 0.1
    assert config['dyn_aux_weight'] == 0.0
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['network_width'] == 1024
    assert config['num_actors'] == 2
    assert 'resume_checkpoint_file' not in config
    assert config['wandb_group'] == 'PAPER-DCC-FULL-CRITICGATE-5SEED'


def test_status_requires_all_ten_task_pickles():
  import tempfile
  run = cfg.build_configs()[0]
  with tempfile.TemporaryDirectory() as tmp:
    prefix = cfg.checkpoint_config_prefix() + '_bridge_abc'
    seed_dir = Path(tmp) / prefix / f'seed_{run["seed"]}'
    seed_dir.mkdir(parents=True)
    for task_id in range(9):
      (seed_dir / f'task_{task_id}.pkl').write_bytes(b'x')
    assert not status.is_complete(run, tmp)
    (seed_dir / 'task_9.pkl').write_bytes(b'x')
    assert status.is_complete(run, tmp)
    incomplete = status.incomplete_array_ids(1, tmp)
    assert 0 not in incomplete
    assert incomplete == [1, 2, 3, 4]


def test_launcher_points_at_grid():
  launcher = (
      REPO_ROOT / 'DRAFT_jubail_full_critic_gate_5seed.sh'
  ).read_text(encoding='utf-8')
  assert 'experiment_configs_full_critic_gate_5seed.py' in launcher
  assert 'tests/test_full_critic_gate_5seed.py' in launcher
  assert '#SBATCH --array=0-4' in launcher
  assert '#SBATCH --gres=gpu:a100:1' in launcher
  assert 'START_TASK=0' in launcher
  assert '--dependency="afterany:${SLURM_JOB_ID}"' in launcher
  assert 'PAPER-DCC-FULL-CRITICGATE-5SEED' in launcher


def test_config_emission_forwards_gate_flags():
  emitted = subprocess.run(
      [sys.executable, 'experiment_configs_full_critic_gate_5seed.py',
       '--setting', '0'],
      cwd=REPO_ROOT, capture_output=True, text=True, check=True)
  assert 'SUCCESS_BC_CRITIC_GATE=true' in emitted.stdout
  assert 'SUCCESS_BC_CRITIC_GATE_MODE=local' in emitted.stdout
  assert 'SUCCESS_BC_WEIGHT=0.1' in emitted.stdout
  assert 'SUCCESS_BC_LABEL_MODE=episode_sparse_reward' in emitted.stdout
  assert 'START_TASK=0' in emitted.stdout
  assert 'NUM_TASKS=10' in emitted.stdout
  assert 'SEED=5' in emitted.stdout


if __name__ == '__main__':
  test_five_from_scratch_cells()
  test_status_requires_all_ten_task_pickles()
  test_launcher_points_at_grid()
  test_config_emission_forwards_gate_flags()
  print('test_full_critic_gate_5seed: ok')
