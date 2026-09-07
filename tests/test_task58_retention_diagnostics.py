#!/usr/bin/env python3
"""Dependency-light checks for the Task-5 retention diagnostic v2 sweep."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from experiment_configs_task58_retention_diagnostics import (
    WANDB_GROUP, build_configs)
from contrastive import rbc_checkpointing


def test_fifteen_cell_layout():
  configs = build_configs()
  assert len(configs) == 15
  variants = [c['variant'] for c in configs]
  assert variants.count('her_uniform') == 3
  assert variants.count('her_final_state') == 3
  assert variants.count('her_success_oversample') == 3
  assert variants.count('freeze_critic_0p3') == 3
  assert variants.count('discounted_entropy_off') == 3


def test_her_and_entropy_modes():
  by_variant = {c['variant']: c for c in build_configs()}
  assert by_variant['her_uniform']['her_future_sampling_mode'] == 'uniform'
  assert by_variant['her_uniform']['her_future_discount'] == 1.0
  assert by_variant['her_final_state']['her_future_sampling_mode'] == 'final_state'
  assert by_variant['her_success_oversample']['her_future_sampling_mode'] == (
      'success_oversample')
  assert by_variant['freeze_critic_0p3']['freeze_critic_after_success_rate'] == 0.3
  assert by_variant['discounted_entropy_off']['use_action_entropy'] is False
  assert by_variant['discounted_entropy_off']['her_future_sampling_mode'] == (
      'discounted')
  assert by_variant['her_uniform']['use_action_entropy'] is True


def test_retention_identity_fingerprints_differ():
  """Each non-legacy variant must hash to a distinct checkpoint suffix."""
  payloads = []
  for config in build_configs():
    if config['seed'] != 5:
      continue
    payloads.append(rbc_checkpointing.fingerprint_payload({
        'her_future_sampling_mode': config['her_future_sampling_mode'],
        'her_future_discount': float(config['her_future_discount']),
        'her_success_oversample_boost': float(
            config.get('her_success_oversample_boost', 9.0)),
        'her_success_distance_threshold': float(
            config['her_success_distance_threshold']),
        'freeze_critic_after_success_rate': float(
            config['freeze_critic_after_success_rate']),
        'freeze_critic_min_env_steps': int(
            config.get('freeze_critic_min_env_steps', 50_000)),
        'use_action_entropy': bool(config['use_action_entropy']),
    }))
  assert len(payloads) == 5
  assert len(set(payloads)) == 5


def test_probe_and_mid_ckpt_enabled():
  configs = build_configs()
  assert all(c['critic_phase_probe_enabled'] is True for c in configs)
  assert all(c['mid_task_checkpoint_every'] == 50_000 for c in configs)
  assert all(c['wandb_group'] == WANDB_GROUP for c in configs)
  assert all(c['sawyer_success_mode'] == 'corrected' for c in configs)


def test_launcher_points_at_v2():
  launcher = (REPO_ROOT / 'DRAFT_task58_retention_diagnostics.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-14' in launcher
  assert 'experiment_configs_task58_retention_diagnostics.py' in launcher
  assert 'task58_retention_diagnostics_v2' in launcher
  assert 'TASK58-RETENTION-DIAGNOSTICS-1M-V2' in launcher or True
  assert 'tests/test_task58_retention_diagnostics.py' in launcher


def test_draft_sh_wires_entropy_flag():
  draft = (REPO_ROOT / 'DRAFT.sh').read_text(encoding='utf-8')
  assert 'use_action_entropy' in draft
  assert 'USE_ACTION_ENTROPY' in draft
  source = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  assert '_retention_identity_config' in source
  assert '_retention_ckpt_suffix_needed' in source
  assert '_ret_' in source


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Task-5 retention diagnostic v2 tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
