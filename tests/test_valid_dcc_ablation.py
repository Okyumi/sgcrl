#!/usr/bin/env python3
"""Dependency-light protocol and launcher checks for the valid DCC batch."""
from __future__ import annotations

from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_configs_valid_dcc_ablation import build_configs  # noqa: E402


def main():
  configs = build_configs()
  assert len(configs) == 14
  assert {c['seed'] for c in configs} == {5, 6}
  assert all(c['sawyer_success_mode'] == 'corrected' for c in configs)
  assert all(c['goal_conditioning_mode'] == 'full_state' for c in configs)
  assert all(c['success_bc_label_mode'] == 'episode_sparse_reward'
             for c in configs)

  task58 = [c for c in configs if c['suite'] == 'task58_4m']
  continual_steps = int(os.environ.get(
      'CONTINUAL_STEPS_PER_TASK', '4000000'))
  continual = [c for c in configs if c['num_tasks'] == 10]
  assert len(task58) == 8
  assert len(continual) == 6
  assert all(c['num_tasks'] == 1 and c['steps_per_task'] == 4_000_000
             for c in task58)
  assert all(c['num_tasks'] == 10
             and c['steps_per_task'] == continual_steps
             for c in continual)
  assert {c['variant'] for c in task58} == {
      'plain_dcc', 'terminal_success_bc_l0p1'}
  assert {c['variant'] for c in continual} == {
      'plain_dcc', 'advantage_1step', 'terminal_success_bc_l0p1'}

  bc = [c for c in configs if c['variant'] == 'terminal_success_bc_l0p1']
  assert all(c['success_bc_weight'] == 0.1 for c in bc)
  assert all(c['success_bc_label_mode'] == 'episode_sparse_reward' for c in bc)
  assert all(not c['action_effect_enabled'] for c in bc)

  advantage = [c for c in configs if c['variant'] == 'advantage_1step']
  assert all(c['critic_mode'] == 'advantage_decomposed' for c in advantage)
  assert all(c['action_effect_target_mode'] == 'psi_one_step'
             for c in advantage)
  assert all(c['success_bc_weight'] == 0 for c in advantage)

  # The launch configurations must not contain any privileged task geometry.
  forbidden = {
      'outcome_horizon', 'outcome_success_threshold',
      'counterfactual_rank_success_threshold',
      'action_landscape_success_threshold',
  }
  assert all(not forbidden.intersection(c) for c in configs)

  runner = (ROOT / 'run_continual_contrastive.py').read_text()
  learner = (ROOT / 'contrastive/continual_learning_decomposed.py').read_text()
  marker = "success_bc_label_mode == 'episode_sparse_reward'"
  assert runner.count(marker) >= 3
  assert 'tf.reduce_max(sample.data.reward[:-1]) > 0.0' in runner
  assert "'success_bc_label_mode': FLAGS.success_bc_label_mode" in runner
  assert 'or FLAGS.success_bc_weight > 0' in runner
  assert 'or success_bc_enabled' in runner
  assert "'retention/bc_weight': success_bc_weight" in learner
  assert "'retention/bc_to_dcc_loss_ratio':" in learner
  assert "jnp.concatenate([new_obs, bc_observation], axis=0)" in learner
  success_update = learner.split('def update_success_buffer', 1)[1].split(
      'def update_step', 1)[0]
  assert 'jnp.cumsum(keep)' in success_update
  assert 'jax.lax.scan' not in success_update

  launcher = (ROOT / 'DRAFT_valid_dcc_ablation.sh').read_text()
  assert '#SBATCH --array=0-6%7' in launcher
  assert 'RUNS_PER_GPU=2' in launcher
  assert '--action_landscape_diagnostic_interval_steps=0' in launcher
  assert '--shortcut_diagnostic_interval=0' in launcher
  assert '--nolog_rl_metrics' in launcher
  assert 'exec bash "$REPO_DIR/DRAFT.sh"' not in launcher

  print('Valid DCC ablation checks passed (14 configs).')


if __name__ == '__main__':
  main()
