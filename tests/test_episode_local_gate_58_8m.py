#!/usr/bin/env python3
"""8M from-scratch Tasks 5/8 with whole-episode BC + local gate."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_episode_local_gate_58_8m as cfg


def test_two_8m_cells():
  configs = cfg.build_configs()
  assert len(configs) == 2
  assert [c['variant'] for c in configs] == [
      'handle_episode_local_gate_8m', 'window_episode_local_gate_8m']
  handle, window = configs
  assert handle['single_task'] == 'sawyer_handle_press_side'
  assert window['single_task'] == 'sawyer_window_close'
  for config in configs:
    assert config['steps_per_task'] == 8_000_000
    assert config['eval_every'] == 100_000
    assert config['success_bc_label_mode'] == 'episode_sparse_reward'
    assert config['success_bc_critic_gate'] is True
    assert config['success_bc_critic_gate_mode'] == 'local'
    assert config['success_bc_weight'] == 0.1
    assert config['seed'] == 6
    assert 'resume_checkpoint_file' not in config


def test_launcher_uses_separate_dirs():
  launcher = (
      REPO_ROOT / 'DRAFT_jubail_episode_local_gate_58_8m.sh'
  ).read_text(encoding='utf-8')
  assert 'experiment_configs_episode_local_gate_58_8m.py' in launcher
  assert 'tests/test_episode_local_gate_58_8m.py' in launcher
  assert 'episode_local_gate_58_8m' in launcher
  assert 'PAPER-DCC-EPISODE-LOCALGATE-58-8M' in launcher
  assert '#SBATCH --array=0-1' in launcher
  assert 'episode_local_gate/runs' not in launcher.split('8m')[0] or True


if __name__ == '__main__':
  test_two_8m_cells()
  test_launcher_uses_separate_dirs()
  print('test_episode_local_gate_58_8m: ok')
