#!/usr/bin/env python3
"""Config checks for Task-5/8 success-truncation cells."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_jubail_success_truncate as cfg


def test_two_diagnosis_cells():
  configs = cfg.build_configs()
  assert len(configs) == 2
  assert [c['variant'] for c in configs] == [
      'handle_truncate', 'window_truncate']
  assert configs[0]['single_task'] == 'sawyer_handle_press_side'
  assert configs[1]['single_task'] == 'sawyer_window_close'
  for config in configs:
    assert config['truncate_on_success'] is True
    assert config['success_bc_weight'] == 0.0
    assert config['critic_mode'] == 'decomposed'
    assert config['actor_mode'] == 'reset'
    assert config['actor_auto_reset'] is False
    assert config['network_width'] == 1024
    assert config['seed'] == 6
    assert config['steps_per_task'] == 1_000_000
    assert config['sawyer_success_mode'] == 'corrected'
    assert config['eval_video_first_success'] is True
    assert config['mid_task_checkpoint_every'] == 50_000


if __name__ == '__main__':
  test_two_diagnosis_cells()
  print('test_jubail_success_truncate: ok')
