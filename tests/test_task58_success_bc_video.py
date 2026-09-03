#!/usr/bin/env python3
"""Dependency-light checks for the Task-5 success-BC video retry sweep."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from experiment_configs_task58_success_bc_video import build_configs


def test_six_cell_layout():
  configs = build_configs()
  assert len(configs) == 6
  baselines = [c for c in configs if c['variant'] == 'dcc_baseline']
  retentions = [c for c in configs if c['variant'] == 'success_bc_combined']
  assert len(baselines) == 3
  assert len(retentions) == 3
  assert {c['seed'] for c in baselines} == {5, 6, 7}
  assert {c['seed'] for c in retentions} == {5, 6, 7}


def test_baseline_is_plain_dcc():
  baseline = build_configs()[0]
  assert baseline['critic_mode'] == 'decomposed'
  assert baseline['action_effect_enabled'] is False
  assert baseline['success_bc_weight'] == 0.0
  assert baseline['sawyer_success_mode'] == 'corrected'
  assert baseline['adapt_heads_only'] is False


def test_retention_uses_combined_actor_and_bc():
  retention = build_configs()[3]
  assert retention['critic_mode'] == 'advantage_decomposed'
  assert retention['action_effect_enabled'] is True
  assert retention['action_effect_target_mode'] == 'raw_horizon'
  assert retention['action_effect_actor_mode'] == 'combined'
  assert retention['success_bc_weight'] == 0.1
  assert retention['success_buffer_capacity'] == 4096


def test_videos_enabled():
  configs = build_configs()
  assert all(c['eval_record_video'] is True for c in configs)
  assert all(c['eval_video_every'] == 100_000 for c in configs)
  assert all(c['wandb_group'] == 'TASK58-SUCCESS-BC-VIDEO-4M' for c in configs)


def test_launcher_points_at_sweep():
  launcher = (REPO_ROOT / 'DRAFT_task58_success_bc_video.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-5' in launcher
  assert 'experiment_configs_task58_success_bc_video.py' in launcher
  assert 'task58_success_bc_video_v1' in launcher
  assert 'tests/test_task58_success_bc_video.py' in launcher


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Task-5 success-BC video sweep tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
