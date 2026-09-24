#!/usr/bin/env python3
"""Checks for terminal-episode Success-BC on Task 4 / Task 7."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_terminal_bc_task47 as cfg


def test_ten_seed_resume_cells():
  configs = cfg.build_configs()
  assert cfg.SEEDS == tuple(range(5, 15))
  assert len(configs) == 20
  stick = configs[:10]
  shelf = configs[10:]
  assert [c['start_task'] for c in stick] == [4] * 10
  assert [c['num_tasks'] for c in stick] == [5] * 10
  assert [c['seed'] for c in stick] == list(range(5, 15))
  assert [c['start_task'] for c in shelf] == [7] * 10
  assert [c['num_tasks'] for c in shelf] == [8] * 10
  assert [c['seed'] for c in shelf] == list(range(5, 15))
  for config in configs:
    assert config['success_bc_label_mode'] == 'terminal_episode'
    assert config['success_bc_weight'] == 0.1
    assert config['steps_per_task'] == 8_000_000
    assert config['eval_every'] == 100_000
    assert config['eval_episodes'] == 10
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['dyn_aux_weight'] == 0.0
    assert Path(config['resume_checkpoint_file']).is_file()
    assert f"seed_{config['seed']}" in config['resume_checkpoint_file']


def test_runner_uses_last_transition_bit():
  runner = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  flatten = runner.split('def flatten_fn', 1)[1].split(
      'def flatten_intrajectory_fn', 1)[0]
  terminal = flatten.split("success_bc_label_mode == 'terminal_episode'")[1]
  terminal = terminal.split("success_bc_label_mode == 'episode_sparse_reward'", 1)[0]
  assert 'sample.data.reward[seq_len - 2]' in terminal
  assert 'tf.fill' in terminal
  assert 'tf.fill(\n              [seq_len - 1], terminal_success)' in terminal
  assert "success_bc_label_mode == 'episode_sparse_reward'" in flatten


def test_launcher_uses_terminal_cells():
  launcher = (REPO_ROOT / 'DRAFT_jubail_terminal_bc_task47.sh').read_text(
      encoding='utf-8')
  assert 'experiment_configs_terminal_bc_task47.py' in launcher
  assert 'tests/test_terminal_bc_task47.py' in launcher
  assert '#SBATCH --array=0-19' in launcher


if __name__ == '__main__':
  test_ten_seed_resume_cells()
  test_runner_uses_last_transition_bit()
  test_launcher_uses_terminal_cells()
  print('test_terminal_bc_task47: ok')
