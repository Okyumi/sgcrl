#!/usr/bin/env python3
"""Dependency-light validation for efficient ten-task wrapper experiments."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from experiment_configs_native_success_wrapper import build_configs


def test_smoke_matrix_is_paired_and_cheap():
  configs = build_configs()
  assert len(configs) == 9
  assert {config['seed'] for config in configs} == {5, 6, 7}
  assert {(config['actor_mode'], config['critic_mode'])
          for config in configs} == {
              ('reset', 'decomposed'),
              ('reset', 'reset'),
              ('persistent', 'persistent'),
          }
  for config in configs:
    assert config['num_tasks'] == 10
    assert config['steps_per_task'] == 100_000
    assert config['base_steps'] == 100_000
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['goal_conditioning_mode'] == 'full_state'
    assert config['eval_every'] == 100_000
    assert config['post_task_eval_scope'] == 'current'
    assert config['counterfactual_rank_interval_steps'] == 0
    assert config['counterfactual_oracle_interval_steps'] == 0
    assert config['action_landscape_diagnostic_interval_steps'] == 0
    assert config['shortcut_diagnostic_interval'] == 0
    assert config['log_rl_metrics'] is False
    assert config['log_pool_cosine'] is False


def test_promotion_is_guarded_and_uses_separate_outputs():
  env = dict(os.environ)
  env.pop('NATIVE_SUCCESS_WRAPPER_PROMOTED', None)
  result = subprocess.run(
      [sys.executable,
       'experiment_configs_native_success_wrapper_promotion.py', '--total'],
      cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
  assert result.returncode != 0
  assert 'NATIVE_SUCCESS_WRAPPER_PROMOTED=true' in result.stderr
  smoke = (REPO_ROOT / 'DRAFT_native_success_wrapper_smoke.sh').read_text(
      encoding='utf-8')
  promotion = (
      REPO_ROOT / 'DRAFT_native_success_wrapper_promotion.sh').read_text(
          encoding='utf-8')
  assert 'TASKS_PER_GPU=3' in smoke
  assert 'TASKS_PER_GPU=3' in promotion
  assert 'native_success_checkpoints/v1_smoke' in smoke
  assert 'native_success_checkpoints/v2_promotion' in promotion
  assert '#SBATCH --cpus-per-task=16' in smoke


def test_config_emission_forwards_native_mode():
  result = subprocess.run(
      [sys.executable, 'experiment_configs_native_success_wrapper.py',
       '--setting', '0'], cwd=REPO_ROOT, capture_output=True, text=True,
      check=True)
  assert 'SAWYER_SUCCESS_MODE=native_info' in result.stdout
  assert 'NUM_TASKS=10' in result.stdout
  assert 'STEPS_PER_TASK=100000' in result.stdout


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'native-success config tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
