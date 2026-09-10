#!/usr/bin/env python3
"""Dependency-light checks for the sparse SAC 10-seed paper rerun."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_paper_sparse_sac_10seed as configs


def _load_status():
  spec = importlib.util.spec_from_file_location(
      'paper_sparse_sac_status',
      REPO_ROOT / 'scripts' / 'paper_sparse_sac_status.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


status = _load_status()


def test_exact_two_by_ten_matrix():
  runs = configs.build_configs()
  assert len(configs.VARIANTS) == 2
  assert len(configs.SEEDS) == 10
  assert len(runs) == 20
  assert set(configs.SEEDS) == set(range(5, 15))
  assert {(run['actor_mode'], run['critic_mode']) for run in runs} == {
      ('reset', 'reset'),
      ('persistent', 'persistent'),
  }


def test_paper_sparse_sac_uses_step_penalty_native_wrapper():
  for run in configs.build_configs():
    assert run['step_penalty_reward'] is True
    assert run['her_reward_threshold'] == 0.05
    assert run['sawyer_success_mode'] == 'native_info'
    assert run['goal_conditioning_mode'] == 'full_state'
    assert run['network_width'] == 1024
    assert run['eval_every'] == 100_000
    assert run['log_rl_metrics'] is True
    assert run['rl_metrics_occasional_multiplier'] == 2
    assert run['steps_per_task'] == 8_000_000
    assert run['post_task_eval_scope'] == 'current'


def test_checkpoint_key_matches_sac_identity():
  path = configs.checkpoint_path(
      '/tmp/ckpts', 'reset', 'reset', 6, 9)
  assert path.endswith(
      'actor_reset_critic_reset_tid_False_heads_True'
      '_rew_steppen_tau_0p05_success_native_info/seed_6/task_9.pkl')


def test_status_treats_missing_checkpoints_as_incomplete():
  with tempfile.TemporaryDirectory() as tmp:
    ids = status.incomplete_array_ids(tasks_per_gpu=4, checkpoint_dir=tmp)
    assert ids == list(range(5))


def test_launcher_packs_four_jobs_and_chains_resubmits():
  launcher = (REPO_ROOT / 'DRAFT_paper_sparse_sac_10seed.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-4' in launcher
  assert '#SBATCH --partition=l40s_public' in launcher
  assert 'TASKS_PER_GPU="${TASKS_PER_GPU:-4}"' in launcher
  assert 'run_continual_sac.py' in launcher
  assert '--dependency="afterany:${SLURM_JOB_ID}"' in launcher
  assert 'step_penalty_reward' in launcher
  assert 'native_info' in launcher


def test_sac_driver_forwards_native_success_and_metrics_multiplier():
  flags_src = (REPO_ROOT / 'sac' / 'flags.py').read_text(encoding='utf-8')
  training = (REPO_ROOT / 'sac' / 'training.py').read_text(encoding='utf-8')
  ckpt = (REPO_ROOT / 'sac' / 'checkpointing.py').read_text(encoding='utf-8')
  assert "'sawyer_success_mode', 'corrected'" in flags_src
  assert 'rl_metrics_occasional_multiplier' in flags_src
  assert 'sawyer_success_mode=_sawyer_success_mode(f)' in training
  assert 'occasional_mult * metrics_every' in training
  assert 'next_metrics_occasional = env_steps_done + 5 * metrics_every' not in training
  assert "if sawyer_success_mode != 'corrected':" in ckpt


def test_config_emission_forwards_paper_settings():
  result = subprocess.run(
      [sys.executable, 'experiment_configs_paper_sparse_sac_10seed.py',
       '--setting', '0'],
      cwd=REPO_ROOT, capture_output=True, text=True, check=True)
  assert 'SAWYER_SUCCESS_MODE=native_info' in result.stdout
  assert 'STEP_PENALTY_REWARD=true' in result.stdout
  assert 'HER_REWARD_THRESHOLD=0.05' in result.stdout
  assert 'NETWORK_WIDTH=1024' in result.stdout
  assert 'ACTOR_MODE=reset' in result.stdout
  assert 'CRITIC_MODE=reset' in result.stdout


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'paper sparse SAC 10-seed tests passed ({len(tests)})')
