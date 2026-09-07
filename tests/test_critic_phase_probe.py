#!/usr/bin/env python3
"""Unit tests for the Task-5/8 critic phase probe classifiers."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import critic_phase_probe as probe


def test_classify_success_hover_mid_reach():
  success_state = np.array(
      [-0.07, 0.70, 0.08, 0.3, -0.07, 0.70, 0.07], dtype=np.float32)
  hover_state = np.array(
      [-0.07, 0.70, 0.10, 0.3, -0.07, 0.70, 0.12], dtype=np.float32)
  mid_state = np.array(
      [0.10, 0.50, 0.20, 0.3, -0.07, 0.70, 0.12], dtype=np.float32)

  assert probe.classify_transition(
      success_state, episode_succeeded=True) == probe.FAMILY_SUCCESS
  assert probe.classify_transition(
      hover_state, episode_succeeded=False) == probe.FAMILY_HOVER
  assert probe.classify_transition(
      mid_state, episode_succeeded=False) == probe.FAMILY_MID_REACH


def test_aggregate_gaps():
  metrics = probe.aggregate_family_scores({
      'success': [2.0, 4.0],
      'hover': [1.0, 1.0],
      'mid_reach': [0.0],
  })
  assert metrics['probe/n_success'] == 2.0
  assert metrics['probe/score_success_vs_task_goal_mean'] == 3.0
  assert metrics['probe/gap_success_minus_hover'] == 2.0
  assert metrics['probe/gap_success_minus_mid_reach'] == 3.0


def test_goal_embedding_metrics():
  metrics = probe.aggregate_goal_embedding_metrics(
      psi_task=np.array([1.0, 0.0], dtype=np.float32),
      psi_hover=np.array([1.0, 0.0], dtype=np.float32),
      success_scores_under_hover_goal=[1.0, 3.0],
      hover_scores_under_hover_goal=[0.0, 2.0],
      success_scores_under_task_goal=[4.0, 6.0],
      hover_scores_under_task_goal=[1.0, 1.0],
  )
  assert metrics['probe/psi_l2_task_vs_hover'] == 0.0
  assert abs(metrics['probe/psi_cosine_task_vs_hover'] - 1.0) < 1e-5
  assert metrics['probe/gap_success_minus_hover_under_hover_goal'] == 1.0
  assert metrics['probe/gap_success_task_goal_minus_hover_goal'] == 3.0
  assert metrics['probe/gap_hover_task_goal_minus_hover_goal'] == 0.0


def test_task_goal_observation_padding():
  state = np.arange(7, dtype=np.float32)
  goal = np.ones(7, dtype=np.float32)
  obs = probe.build_task_goal_observation(state, goal, obs_dim=10)
  assert obs.shape == (20,)
  assert np.allclose(obs[:7], state)
  assert np.allclose(obs[10:17], goal)


def test_axis_spec():
  assert probe.task58_axis_spec('sawyer_handle_press_side')[0] == 6
  assert probe.task58_axis_spec('sawyer_window_close')[0] == 5


def test_offline_script_exists():
  path = REPO_ROOT / 'scripts' / 'probe_critic_phases.py'
  assert path.is_file()
  text = path.read_text(encoding='utf-8')
  assert 'run_critic_phase_probe' in text
  assert 'decomposed_training_state' in text


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Critic phase probe tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
