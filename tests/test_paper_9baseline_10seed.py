#!/usr/bin/env python3
"""Dependency-light checks for the 9-baseline × 10-seed paper rerun."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_paper_9baseline_10seed as configs


def _load_status():
  spec = importlib.util.spec_from_file_location(
      'paper_9baseline_status',
      REPO_ROOT / 'scripts' / 'paper_9baseline_status.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


status = _load_status()


def test_exact_nine_by_ten_matrix():
  runs = configs.build_configs()
  assert len(configs.VARIANTS) == 9
  assert len(configs.SEEDS) == 10
  assert len(runs) == 90
  assert set(configs.SEEDS) == set(range(5, 15))
  assert {(actor, critic) for _, actor, critic in configs.VARIANTS} == {
      (actor, critic)
      for actor in configs.BASE_MODES
      for critic in configs.BASE_MODES
  }


def test_paper_runtime_and_metrics_settings():
  for run in configs.build_configs():
    assert run['sawyer_success_mode'] == 'native_info'
    assert run['goal_conditioning_mode'] == 'full_state'
    assert run['num_tasks'] == 10
    assert run['steps_per_task'] == 8_000_000
    assert run['network_width'] == 1024
    assert run['k_max'] == 5
    assert run['eval_every'] == 100_000
    assert run['log_rl_metrics'] is True
    assert run['rl_metrics_occasional_multiplier'] == 2
    assert run['profile_runtime'] is True
    assert run['post_task_eval_scope'] == 'current'
    assert run['counterfactual_rank_interval_steps'] == 0
    uses_cka = run['actor_mode'] == 'cka' or run['critic_mode'] == 'cka'
    assert run['log_mixture_norm'] is uses_cka
    assert run['log_pool_cosine'] is uses_cka


def test_checkpoint_key_matches_runner_identity():
  path = configs.checkpoint_path(
      '/tmp/ckpts', 'reset', 'persistent', 6, 9)
  assert path.endswith(
      'actor_reset_critic_persistent_tid_False_heads_True'
      '_success_native_info/seed_6/task_9.pkl')


def test_status_treats_missing_checkpoints_as_incomplete():
  with tempfile.TemporaryDirectory() as tmp:
    ids = status.incomplete_array_ids(
        tasks_per_gpu=4, checkpoint_dir=tmp)
    assert ids == list(range(23))
    assert status.latest_completed_task(
        configs.build_configs()[0], tmp) == -1


def test_status_complete_requires_all_ten_task_pickles():
  run = configs.build_configs()[0]
  with tempfile.TemporaryDirectory() as tmp:
    for task_id in range(9):
      path = Path(configs.checkpoint_path(
          tmp, run['actor_mode'], run['critic_mode'], run['seed'],
          task_id))
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_bytes(b'x')
    assert not status.is_complete(run, tmp)
    Path(configs.checkpoint_path(
        tmp, run['actor_mode'], run['critic_mode'], run['seed'], 9)
        ).write_bytes(b'x')
    assert status.is_complete(run, tmp)


def test_runner_exposes_occasional_multiplier_flag():
  source = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  tree = ast.parse(source)
  flag_names = []
  for node in ast.walk(tree):
    if (isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith('DEFINE_')
        and node.args
        and isinstance(node.args[0], ast.Constant)):
      flag_names.append(node.args[0].value)
  assert 'rl_metrics_occasional_multiplier' in flag_names
  assert 'occasional_mult * metrics_every' in source
  assert 'next_metrics_occasional = env_steps_done + 5 * metrics_every' not in source


def test_launcher_packs_four_jobs_and_chains_resubmits():
  launcher = (REPO_ROOT / 'DRAFT_paper_9baseline_10seed.sh').read_text(
      encoding='utf-8')
  draft = (REPO_ROOT / 'DRAFT.sh').read_text(encoding='utf-8')
  assert '#SBATCH --array=0-22' in launcher
  assert '#SBATCH --partition=l40s_public' in launcher
  assert 'TASKS_PER_GPU="${TASKS_PER_GPU:-4}"' in launcher
  assert 'XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.22}"' in launcher
  assert 'native_info' in launcher or 'CONFIG_SCRIPT="experiment_configs_paper_9baseline_10seed.py"' in launcher
  assert '--dependency="afterany:${SLURM_JOB_ID}"' in launcher
  assert 'HOME}/.wandb_api_key' in launcher
  assert '--rl_metrics_occasional_multiplier=$RL_METRICS_OCCASIONAL_MULTIPLIER' in draft


def test_config_emission_forwards_paper_settings():
  result = subprocess.run(
      [sys.executable, 'experiment_configs_paper_9baseline_10seed.py',
       '--setting', '0'],
      cwd=REPO_ROOT, capture_output=True, text=True, check=True)
  assert 'SAWYER_SUCCESS_MODE=native_info' in result.stdout
  assert 'NETWORK_WIDTH=1024' in result.stdout
  assert 'LOG_RL_METRICS=true' in result.stdout
  assert 'K_MAX=5' in result.stdout
  assert 'ACTOR_MODE=reset' in result.stdout
  assert 'CRITIC_MODE=reset' in result.stdout


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'paper 9-baseline 10-seed tests passed ({len(tests)})')
