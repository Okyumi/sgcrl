#!/usr/bin/env python3
"""Checks for per-step Success-BC labels and the Task 4/7 resume probe."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_current_sparse_bc_task47 as cfg


def test_two_resume_cells():
  configs = cfg.build_configs()
  assert len(configs) == 2
  assert configs[0]['start_task'] == 4
  assert configs[0]['num_tasks'] == 5
  assert configs[1]['start_task'] == 7
  assert configs[1]['num_tasks'] == 8
  for config in configs:
    assert config['success_bc_label_mode'] == 'current_sparse_reward'
    assert config['success_bc_weight'] == 0.1
    assert config['seed'] == 6
    assert config['steps_per_task'] == 8_000_000
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['dyn_aux_weight'] == 0.0
    assert Path(config['resume_checkpoint_file']).is_file()


def test_runner_uses_per_step_sparse_bit():
  runner = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  assert 'current_sparse_reward' in runner
  learner = (
      REPO_ROOT / 'contrastive' / 'continual_learning_decomposed.py'
  ).read_text(encoding='utf-8')
  assert 'current_sparse_reward' in learner
  flatten = runner.split('def flatten_fn', 1)[1].split(
      'def flatten_intrajectory_fn', 1)[0]
  current = flatten.split("success_bc_label_mode == 'current_sparse_reward'")[1]
  current = current.split("if actor_goal_mode", 1)[0]
  assert 'sample.data.reward[:-1] > 0.0' in current
  assert 'tf.fill' not in current
  assert 'tf.reduce_max' not in current


if __name__ == '__main__':
  test_two_resume_cells()
  test_runner_uses_per_step_sparse_bit()
  print('test_current_sparse_bc_task47: ok')
