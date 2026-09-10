#!/usr/bin/env python3
"""Sanity checks for Task5 vs push success-propagation configs."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_task58_success_propagation as cfg


def test_counts_and_variants():
  configs = cfg.build_configs()
  assert len(configs) == 12
  by_variant = {}
  for c in configs:
    by_variant.setdefault(c['variant'], []).append(c)
  assert set(by_variant) == {
      'handle_measure', 'push_measure', 'handle_inject', 'push_inject'}
  for name, rows in by_variant.items():
    assert len(rows) == 3
    assert {r['seed'] for r in rows} == {5, 6, 7}
  assert all(not c['success_inject_enabled']
             for c in by_variant['handle_measure'])
  assert all(c['success_inject_enabled']
             for c in by_variant['handle_inject'])
  assert by_variant['push_measure'][0]['single_task'] == 'sawyer_push'
  assert by_variant['handle_measure'][0]['single_task'] == (
      'sawyer_handle_press_side')
  sample = by_variant['handle_measure'][0]
  assert sample['success_trace_log_enabled'] is True
  assert sample['actor_follow_probe_enabled'] is True
  assert sample['her_phase_log_enabled'] is True
  assert sample['critic_phase_probe_enabled'] is True
  assert sample['success_bc_weight'] == 0.0


def main():
  test_counts_and_variants()
  print('task58 success-propagation config tests passed')


if __name__ == '__main__':
  main()
