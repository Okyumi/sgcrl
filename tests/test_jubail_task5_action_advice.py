#!/usr/bin/env python3
"""Sanity checks for the Jubail Task-5 action-advice diagnostic configs."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_jubail_task5_action_advice as cfg


def test_counts_and_variants():
  configs = cfg.build_configs()
  assert len(configs) == 3
  names = [c['variant'] for c in configs]
  assert names == [
      'handle_measure', 'push_measure', 'handle_inject_20pct']
  assert all(c['seed'] == 6 for c in configs)
  assert configs[0]['single_task'] == 'sawyer_handle_press_side'
  assert configs[1]['single_task'] == 'sawyer_push'
  assert configs[2]['success_inject_enabled'] is True
  assert configs[2]['success_inject_target_frac'] == 0.20
  assert configs[2]['success_inject_clone'] is True
  assert configs[0]['success_inject_enabled'] is False
  sample = configs[0]
  assert sample['adapt_heads_only'] is False
  assert sample['success_bc_weight'] == 0.0
  assert sample['stage_dwell_log_enabled'] is True
  assert sample['press_vs_pi_probe_enabled'] is True
  assert sample['critic_phase_probe_enabled'] is True
  assert sample['run_action_advice_probe'] is True
  assert sample['action_advice_targets'] == '100000,latest'
  assert configs[1]['action_advice_targets'] == '250000,latest'
  assert sample['wandb_group'] == cfg.WANDB_GROUP
  assert sample['mid_task_checkpoint_every'] == 50_000
  assert sample['network_width'] == 1024


def _load_helper():
  import importlib.util
  path = REPO_ROOT / 'scripts' / 'run_action_advice_from_checkpoints.py'
  spec = importlib.util.spec_from_file_location(
      'run_action_advice_from_checkpoints', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_helper_picks_closest_and_latest(tmp_path: Path):
  helper = _load_helper()

  env = 'sawyer_handle_press_side'
  root = tmp_path / 'ckpts'
  seed_dir = root / f'actor_reset_critic_decomposed_tid_False_heads_False_success_corrected_dyn1.000_pt256x4_env_{env}' / 'seed_6'
  seed_dir.mkdir(parents=True)
  for step in (50100, 100200, 951900):
    (seed_dir / f'task_0_step_{step}.pkl').write_bytes(b'x')
  ckpts = helper.list_step_ckpts(root, env, 6)
  assert [s for s, _ in ckpts] == [50100, 100200, 951900]
  picked = helper.pick_targets(ckpts, ['100000', 'latest'])
  assert [s for s, _ in picked] == [100200, 951900]


def main():
  test_counts_and_variants()
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    test_helper_picks_closest_and_latest(Path(tmp))
  print('jubail task5 action-advice config tests passed')


if __name__ == '__main__':
  main()
