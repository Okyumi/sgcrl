#!/usr/bin/env python3
"""Unit tests for HER future-phase classification."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import her_future_phase as hfp


def test_handle_success_vs_hover():
  success = np.array(
      [-0.07, 0.70, 0.08, 0.3, -0.07, 0.70, 0.07], dtype=np.float32)
  hover = np.array(
      [-0.07, 0.70, 0.10, 0.3, -0.07, 0.70, 0.12], dtype=np.float32)
  far = np.array(
      [0.10, 0.40, 0.20, 0.3, -0.07, 0.70, 0.12], dtype=np.float32)
  assert hfp.classify_her_goal(
      success, 'sawyer_handle_press_side') == hfp.PHASE_SUCCESS
  assert hfp.classify_her_goal(
      hover, 'sawyer_handle_press_side') == hfp.PHASE_HOVER
  assert hfp.classify_her_goal(
      far, 'sawyer_handle_press_side') == hfp.PHASE_FAR


def test_push_progress_bucket():
  at_goal = np.array(
      [0.02, 0.89, 0.05, 0.3, 0.02, 0.89, 0.02], dtype=np.float32)
  mid = np.array(
      [0.02, 0.80, 0.05, 0.3, 0.02, 0.80, 0.02], dtype=np.float32)
  assert hfp.classify_her_goal(at_goal, 'sawyer_push') == hfp.PHASE_SUCCESS
  assert hfp.classify_her_goal(mid, 'sawyer_push') == hfp.PHASE_PROGRESS


def test_batch_summary_and_ema():
  goals = np.stack([
      np.array([-0.07, 0.70, 0.08, 0.3, -0.07, 0.70, 0.07], dtype=np.float32),
      np.array([-0.07, 0.70, 0.10, 0.3, -0.07, 0.70, 0.12], dtype=np.float32),
  ])
  metrics = hfp.summarize_her_goal_batch(
      goals, 'sawyer_handle_press_side')
  assert metrics['her_phase/frac_success'] == 0.5
  assert metrics['her_phase/frac_hover_unsolved'] == 0.5
  ema = hfp.ema_update({}, metrics, decay=0.5)
  assert abs(ema['her_phase/frac_success'] - 0.5) < 1e-6


def test_taxonomy_covers_continual_sequence():
  rows = hfp.continual_sequence_taxonomy()
  assert len(rows) == 10
  families = {row['env_name']: row['family'] for row in rows}
  assert families['sawyer_push'] == 'continuous_object_progress'
  assert families['sawyer_handle_press_side'] == 'short_contact_mechanism'
  assert families['sawyer_peg_unplug_side'] == 'short_contact_mechanism'
  assert families['sawyer_hammer'] == 'multi_phase_contact'
  # Not only tasks 5/8: faucet and peg are also contact/mechanism risks.
  contactish = [r for r in rows if r['short_contact_plus_jump_risk']]
  assert len(contactish) >= 5


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'HER future-phase tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
