#!/usr/bin/env python3
"""Checks for Task-7 shelf_place policy GIF export."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_jubail_task7_policy_gifs as cfg


def _load_recorder():
  path = REPO_ROOT / 'scripts' / 'record_checkpoint_rollout_gifs.py'
  spec = importlib.util.spec_from_file_location(
      'record_checkpoint_rollout_gifs', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_three_shelf_cells():
  configs = cfg.build_configs()
  assert len(configs) == 3
  assert [c['variant'] for c in configs] == [
      'shelf_plain', 'shelf_bc', 'shelf_terminal']
  assert [c['job_mode'] for c in configs] == [
      'checkpoint_file', 'checkpoint_file', 'mid_ckpts']
  assert all(c['single_task'] == 'sawyer_shelf_place' for c in configs)
  assert all(c['seed'] == 6 for c in configs)
  assert Path(configs[0]['checkpoint_file']).is_file()
  assert Path(configs[1]['checkpoint_file']).is_file()
  assert Path(configs[2]['checkpoint_dir']).is_dir()
  assert configs[2]['success_bc_label_mode'] == 'terminal_episode'
  assert configs[2]['first_success_step'] == 1_500_750
  assert configs[2]['horizon'] == 8_000_000
  assert configs[1]['success_bc_label_mode'] == 'episode_sparse_reward'


def test_recorder_lists_task7_mid_ckpts():
  recorder = _load_recorder()
  tokens = recorder.even_target_tokens(10, horizon=8_000_000)
  assert tokens == [
      '800000', '1600000', '2400000', '3200000', '4000000',
      '4800000', '5600000', '6400000', '7200000', '8000000']
  ckpts = recorder.list_task_step_ckpts(
      Path(cfg.TERMINAL_DIR), cfg.SEED, 7)
  assert ckpts
  assert ckpts[0][0] == 100050
  picked = recorder.pick_targets(ckpts, tokens)
  assert len(picked) == 10
  first = recorder.pick_targets(ckpts, ['1500750'])
  assert first and abs(first[0][0] - 1_500_750) < 1000


def test_launcher():
  launcher = (REPO_ROOT / 'DRAFT_jubail_task7_policy_gifs.sh').read_text(
      encoding='utf-8')
  assert 'experiment_configs_jubail_task7_policy_gifs.py' in launcher
  assert 'tests/test_jubail_task7_policy_gifs.py' in launcher
  assert '#SBATCH --array=0-2' in launcher
  assert '--task-id' in launcher
  assert '--horizon' in launcher


if __name__ == '__main__':
  test_three_shelf_cells()
  test_recorder_lists_task7_mid_ckpts()
  test_launcher()
  print('test_jubail_task7_policy_gifs: ok')
