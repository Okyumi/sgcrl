#!/usr/bin/env python3
"""Dependency-light checks for the Task-5 actor retention v3 sweep."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from experiment_configs_task58_actor_retention_v3 import (
    WANDB_GROUP, build_configs)
from contrastive import rbc_checkpointing


def test_fifteen_cell_layout():
  configs = build_configs()
  assert len(configs) == 15
  variants = [c['variant'] for c in configs]
  assert variants.count('dcc_control') == 3
  assert variants.count('actor_goal_task') == 3
  assert variants.count('actor_goal_mix') == 3
  assert variants.count('actor_success_score') == 3
  assert variants.count('success_bc_terminal') == 3


def test_variant_settings():
  by_variant = {c['variant']: c for c in build_configs()}
  assert by_variant['dcc_control']['actor_goal_mode'] == 'her'
  assert by_variant['dcc_control']['actor_success_score_weight'] == 0.0
  assert by_variant['dcc_control']['success_bc_weight'] == 0.0
  assert by_variant['actor_goal_task']['actor_goal_mode'] == 'task'
  assert by_variant['actor_goal_mix']['actor_goal_mode'] == 'mix'
  assert by_variant['actor_success_score']['actor_success_score_weight'] == 0.1
  assert by_variant['actor_success_score']['success_bc_weight'] == 0.0
  assert by_variant['success_bc_terminal']['success_bc_weight'] == 0.1
  assert by_variant['success_bc_terminal']['success_bc_label_mode'] == (
      'terminal_episode')
  assert by_variant['actor_success_score']['success_bc_label_mode'] == (
      'terminal_episode')


def test_retention_identity_fingerprints_differ():
  payloads = []
  for config in build_configs():
    if config['seed'] != 5:
      continue
    payloads.append(rbc_checkpointing.fingerprint_payload({
        'her_future_sampling_mode': config['her_future_sampling_mode'],
        'her_future_discount': float(config['her_future_discount']),
        'her_success_oversample_boost': 9.0,
        'her_success_distance_threshold': float(
            config['her_success_distance_threshold']),
        'freeze_critic_after_success_rate': float(
            config['freeze_critic_after_success_rate']),
        'freeze_critic_min_env_steps': 50_000,
        'use_action_entropy': bool(config['use_action_entropy']),
        'actor_goal_mode': config['actor_goal_mode'],
        'actor_success_score_weight': float(
            config['actor_success_score_weight']),
        'success_bc_weight': float(config['success_bc_weight']),
        'success_bc_label_mode': config['success_bc_label_mode'],
    }))
  assert len(payloads) == 5
  assert len(set(payloads)) == 5


def test_probe_enabled():
  configs = build_configs()
  assert all(c['critic_phase_probe_enabled'] is True for c in configs)
  assert all(c['mid_task_checkpoint_every'] == 50_000 for c in configs)
  assert all(c['wandb_group'] == WANDB_GROUP for c in configs)


def test_launcher_and_wiring():
  launcher = (
      REPO_ROOT / 'DRAFT_task58_actor_retention_v3.sh').read_text(
          encoding='utf-8')
  assert '#SBATCH --array=0-14' in launcher
  assert 'experiment_configs_task58_actor_retention_v3.py' in launcher
  assert 'task58_actor_retention_v3' in launcher
  draft = (REPO_ROOT / 'DRAFT.sh').read_text(encoding='utf-8')
  assert 'ACTOR_GOAL_MODE' in draft
  assert 'ACTOR_SUCCESS_SCORE_WEIGHT' in draft
  assert 'SUCCESS_BC_LABEL_MODE' in draft
  source = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  assert 'actor_goal_mode' in source
  assert 'actor_success_score_weight' in source
  assert 'encode_goal_embedding' in (
      REPO_ROOT / 'contrastive' / 'continual_learning_decomposed.py'
  ).read_text(encoding='utf-8')
  assert 'psi_cosine_task_vs_hover' in (
      REPO_ROOT / 'contrastive' / 'critic_phase_probe.py'
  ).read_text(encoding='utf-8')


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Task-5 actor retention v3 tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
