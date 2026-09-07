#!/usr/bin/env python3
"""Checks for the HER-phase geometry comparison sweep."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from experiment_configs_task58_her_phase_geometry import (
    WANDB_GROUP, build_configs)


def test_twelve_cell_layout():
  configs = build_configs()
  assert len(configs) == 12
  variants = [c['variant'] for c in configs]
  assert variants.count('push_progress') == 3
  assert variants.count('handle_contact') == 3
  assert variants.count('faucet_contact') == 3
  assert variants.count('peg_contact') == 3


def test_phase_log_enabled():
  configs = build_configs()
  assert all(c['her_phase_log_enabled'] is True for c in configs)
  assert all(c['critic_phase_probe_enabled'] is True for c in configs)
  assert all(c['wandb_group'] == WANDB_GROUP for c in configs)


def test_tasks_and_wiring():
  by_variant = {c['variant']: c for c in build_configs()}
  assert by_variant['push_progress']['single_task'] == 'sawyer_push'
  assert by_variant['handle_contact']['single_task'] == (
      'sawyer_handle_press_side')
  assert by_variant['faucet_contact']['single_task'] == 'sawyer_faucet_close'
  assert by_variant['peg_contact']['single_task'] == 'sawyer_peg_unplug_side'
  launcher = (
      REPO_ROOT / 'DRAFT_task58_her_phase_geometry.sh').read_text(
          encoding='utf-8')
  assert '#SBATCH --array=0-11' in launcher
  draft = (REPO_ROOT / 'DRAFT.sh').read_text(encoding='utf-8')
  assert 'HER_PHASE_LOG_ENABLED' in draft
  runner = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  assert 'her_phase_log_enabled' in runner
  assert 'her_future_phase' in runner


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'HER-phase geometry config tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
