#!/usr/bin/env python3
"""10-seed gated episode-BC on Tasks 4, 7, and 8."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_episode_local_gate_478_10seed as cfg


def test_thirty_resume_cells():
  configs = cfg.build_configs()
  assert cfg.SEEDS == tuple(range(5, 15))
  assert len(configs) == 30
  stick, shelf, window = configs[:10], configs[10:20], configs[20:]
  assert [c['start_task'] for c in stick] == [4] * 10
  assert [c['num_tasks'] for c in stick] == [5] * 10
  assert [c['seed'] for c in stick] == list(range(5, 15))
  assert [c['start_task'] for c in shelf] == [7] * 10
  assert [c['num_tasks'] for c in shelf] == [8] * 10
  assert [c['seed'] for c in shelf] == list(range(5, 15))
  assert [c['start_task'] for c in window] == [8] * 10
  assert [c['num_tasks'] for c in window] == [9] * 10
  assert [c['seed'] for c in window] == list(range(5, 15))
  for config in configs:
    assert config['success_bc_label_mode'] == 'episode_sparse_reward'
    assert config['success_bc_critic_gate'] is True
    assert config['success_bc_critic_gate_mode'] == 'local'
    assert config['success_bc_weight'] == 0.1
    assert config['steps_per_task'] == 8_000_000
    assert config['eval_every'] == 100_000
    assert config['sawyer_success_mode'] == 'native_info'
    assert Path(config['resume_checkpoint_file']).is_file()
    assert f"seed_{config['seed']}" in config['resume_checkpoint_file']
  assert all('task_3.pkl' in c['resume_checkpoint_file'] for c in stick)
  assert all('task_6.pkl' in c['resume_checkpoint_file'] for c in shelf)
  assert all('task_7.pkl' in c['resume_checkpoint_file'] for c in window)


def test_launcher_points_at_grid():
  launcher = (
      REPO_ROOT / 'DRAFT_jubail_episode_local_gate_478_10seed.sh'
  ).read_text(encoding='utf-8')
  assert 'experiment_configs_episode_local_gate_478_10seed.py' in launcher
  assert 'tests/test_episode_local_gate_478_10seed.py' in launcher
  assert '#SBATCH --array=0-29' in launcher
  assert 'PAPER-DCC-EPISODE-LOCALGATE-478-10SEED' in launcher


if __name__ == '__main__':
  test_thirty_resume_cells()
  test_launcher_points_at_grid()
  print('test_episode_local_gate_478_10seed: ok')
