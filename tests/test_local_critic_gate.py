#!/usr/bin/env python3
"""Local-σ_s critic-gated Success-BC on Tasks 4/7 and 5/8."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_local_critic_gate as cfg


def test_four_cells():
  configs = cfg.build_configs()
  assert len(configs) == 4
  assert [c['variant'] for c in configs] == [
      'task4_stick_local_gate', 'task7_shelf_local_gate',
      'handle_local_gate', 'window_local_gate']
  for config in configs:
    assert config['success_bc_label_mode'] == 'current_sparse_reward'
    assert config['success_bc_critic_gate'] is True
    assert config['success_bc_critic_gate_mode'] == 'local'
    assert config['success_bc_critic_state_noise'] == 0.1
    assert config['success_bc_weight'] == 0.1
    assert config['seed'] == 6
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['dyn_aux_weight'] == 0.0
    assert config['truncate_on_success'] is False

  stick, shelf, handle, window = configs
  assert stick['start_task'] == 4
  assert stick['steps_per_task'] == 8_000_000
  assert Path(stick['resume_checkpoint_file']).is_file()
  assert shelf['start_task'] == 7
  assert Path(shelf['resume_checkpoint_file']).is_file()
  assert handle['single_task'] == 'sawyer_handle_press_side'
  assert handle['steps_per_task'] == 1_000_000
  assert window['single_task'] == 'sawyer_window_close'
  assert window['steps_per_task'] == 1_000_000


def test_learner_jitters_this_state_not_other_batch_states():
  learner = (
      REPO_ROOT / 'contrastive' / 'continual_learning_decomposed.py'
  ).read_text(encoding='utf-8')
  local = learner.split("success_bc_critic_gate_mode == 'local':", 1)[1]
  local = local.split('else:', 1)[0]
  assert 'state_pert' in local
  assert 'bc_state' in local
  assert 'obs_dim = decomp_nets.obs_dim' not in local
  assert 'sample_fn(bc_dist_params' in local
  assert 'bc_action' not in local
  assert '0.09' not in local
  assert 'hover' not in local.lower()
  assert 'handle' not in local.lower()
  batch = learner.split("success_bc_critic_gate_mode == 'local':", 1)[1]
  batch = batch.split('else:', 1)[1]
  batch = batch.split('denom = sigma_s + sigma_a', 1)[0]
  assert 'jnp.mean(probe_scores, axis=0)' in batch


def test_launcher_points_at_grid():
  launcher = (REPO_ROOT / 'DRAFT_jubail.sh').read_text(encoding='utf-8')
  assert '--success_bc_critic_gate_mode=$SUCCESS_BC_CRITIC_GATE_MODE' in launcher
  array = (
      REPO_ROOT / 'DRAFT_jubail_local_critic_gate.sh'
  ).read_text(encoding='utf-8')
  assert 'experiment_configs_local_critic_gate.py' in array
  assert 'tests/test_local_critic_gate.py' in array
  assert '#SBATCH --array=0-3' in array


if __name__ == '__main__':
  test_four_cells()
  test_learner_jitters_this_state_not_other_batch_states()
  test_launcher_points_at_grid()
  print('test_local_critic_gate: ok')
