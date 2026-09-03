#!/usr/bin/env python3
"""Dependency-light checks for the corrected Task-5/8 algorithm pilot."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_configs_task58_algorithm_ablation import build_configs  # noqa: E402


def main():
  configs = build_configs()
  assert len(configs) == 12
  assert {c['seed'] for c in configs} == {5, 6}
  assert {c['single_task'] for c in configs} == {
      'sawyer_handle_press_side', 'sawyer_window_close'}
  assert all(c['sawyer_success_mode'] == 'corrected' for c in configs)
  assert all(c['steps_per_task'] == 1_000_000 for c in configs)

  by_name = {}
  for config in configs:
    by_name.setdefault(config['name'], []).append(config)
  assert set(by_name) == {
      'advantage_1step', 'advantage_h25', 'terminal_success_bc'}
  assert all(len(rows) == 4 for rows in by_name.values())

  one_step = by_name['advantage_1step']
  assert all(c['action_effect_enabled'] for c in one_step)
  assert all(c['action_effect_target_mode'] == 'psi_one_step'
             for c in one_step)
  assert all(c['success_bc_weight'] == 0 for c in one_step)

  h25 = by_name['advantage_h25']
  assert all(c['action_effect_enabled'] for c in h25)
  assert all(c['action_effect_target_mode'] == 'raw_horizon' for c in h25)
  assert all(c['success_bc_weight'] == 0 for c in h25)

  bc = by_name['terminal_success_bc']
  assert all(not c['action_effect_enabled'] for c in bc)
  assert all(c['critic_mode'] == 'decomposed' for c in bc)
  assert all(c['success_bc_weight'] == 0.1 for c in bc)
  assert all(c['success_bc_label_mode'] == 'terminal_episode' for c in bc)

  runner = (ROOT / 'run_continual_contrastive.py').read_text()
  learner = (ROOT / 'contrastive/continual_learning_decomposed.py').read_text()
  launcher = (ROOT / 'DRAFT_task58_algorithm_ablation.sh').read_text()
  assert "sample.data.reward[seq_len - 2]" in runner
  assert "success_bc_label_mode == 'terminal_episode'" in runner
  assert "if success_bc_weight > 0:" in learner
  assert 'exec bash "$REPO_DIR/DRAFT.sh"' not in launcher
  assert 'TASK58-CORRECTED-ADVANTAGE-1STEP-1M' in str(configs)
  assert 'TASK58-CORRECTED-ADVANTAGE-H25-1M' in str(configs)
  assert 'TASK58-CORRECTED-TERMINAL-SUCCESS-BC-1M' in str(configs)
  print('Task-5/Task-8 algorithm ablation checks passed (12 configs).')


if __name__ == '__main__':
  main()
