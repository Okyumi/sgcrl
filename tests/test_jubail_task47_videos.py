#!/usr/bin/env python3
"""Config checks for paper Task-4 / Task-7 GIF export cells."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_jubail_task47_videos as cfg


def _load_recorder():
  path = REPO_ROOT / 'scripts' / 'record_checkpoint_rollout_gifs.py'
  spec = importlib.util.spec_from_file_location(
      'record_checkpoint_rollout_gifs', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_four_final_policy_cells():
  configs = cfg.build_configs()
  assert len(configs) == 4
  assert [c['variant'] for c in configs] == [
      'stick_plain', 'stick_bc', 'shelf_plain', 'shelf_bc']
  assert configs[0]['single_task'] == 'sawyer_stick_pull'
  assert configs[2]['single_task'] == 'sawyer_shelf_place'
  assert configs[0]['success_bc_weight'] == 0.0
  assert configs[1]['success_bc_weight'] == 0.1
  assert all(c['seed'] == 6 for c in configs)
  assert all(Path(c['checkpoint_file']).is_file() for c in configs)
  assert all(c['sawyer_success_mode'] == 'native_info' for c in configs)


def test_recorder_knows_stick_and_shelf_goals():
  recorder = _load_recorder()
  assert 'sawyer_stick_pull' in recorder.FIXED_GOALS
  assert 'sawyer_shelf_place' in recorder.FIXED_GOALS


if __name__ == '__main__':
  test_four_final_policy_cells()
  test_recorder_knows_stick_and_shelf_goals()
  print('test_jubail_task47_videos: ok')
