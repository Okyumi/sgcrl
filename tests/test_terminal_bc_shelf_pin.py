#!/usr/bin/env python3
"""Task 7 terminal Success-BC probe after the shelf-mesh pin."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_terminal_bc_shelf_pin as cfg


def test_ten_task7_resume_cells():
  configs = cfg.build_configs()
  assert cfg.SEEDS == tuple(range(5, 15))
  assert len(configs) == 10
  assert [c['seed'] for c in configs] == list(range(5, 15))
  for config in configs:
    assert config['variant'] == 'task7_shelf_terminal_pin'
    assert config['start_task'] == 7
    assert config['num_tasks'] == 8
    assert config['success_bc_label_mode'] == 'terminal_episode'
    assert config['success_bc_weight'] == 0.1
    assert 'success_bc_warmup_episodes' not in config
    assert config['steps_per_task'] == 8_000_000
    assert config['eval_every'] == 100_000
    assert config['eval_episodes'] == 10
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['dyn_aux_weight'] == 0.0
    assert config['resume_checkpoint_file'].endswith(
        f"/seed_{config['seed']}/task_6.pkl")
    assert Path(config['resume_checkpoint_file']).is_file()


def test_launcher_and_wrapper_pin():
  launcher = (REPO_ROOT / 'DRAFT_jubail_terminal_bc_shelf_pin.sh').read_text(
      encoding='utf-8')
  assert 'experiment_configs_terminal_bc_shelf_pin.py' in launcher
  assert 'tests/test_terminal_bc_shelf_pin.py' in launcher
  assert 'tests/test_shelf_place_fixed_goal_pin.py' in launcher
  assert '#SBATCH --array=0-9' in launcher
  wrapper = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  reset = wrapper.split('class SawyerShelfPlace(', 1)[1].split(
      'def reset(self):', 1)[1].split('def step(', 1)[0]
  assert "body_name2id('shelf')" in reset
  assert 'self._goal - np.array([0.0, 0.0, 0.3]' in reset


if __name__ == '__main__':
  test_ten_task7_resume_cells()
  test_launcher_and_wrapper_pin()
  print('test_terminal_bc_shelf_pin: ok')
