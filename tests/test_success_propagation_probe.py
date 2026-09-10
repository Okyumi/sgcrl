#!/usr/bin/env python3
"""Unit tests for success-propagation probes (D2–D4 helpers)."""
from __future__ import annotations

from pathlib import Path
import sys

import dm_env
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import success_propagation_probe as spp


def test_aggregate_success_trace_metrics():
  metrics = spp.aggregate_success_trace_metrics(
      [3.0, 5.0, 1.0, 0.0],
      [True, True, False, False],
      near_unsolved_flags=[False, False, True, False],
  )
  assert metrics['trace/n_success_ep_transitions'] == 2.0
  assert metrics['trace/score_success_ep_mean'] == 4.0
  assert metrics['trace/score_fail_ep_mean'] == 0.5
  assert metrics['trace/gap_success_minus_fail'] == 3.5
  assert metrics['trace/score_fail_hover_mean'] == 1.0
  assert metrics['trace/gap_success_minus_fail_hover'] == 3.0


def test_aggregate_actor_follow_metrics():
  metrics = spp.aggregate_actor_follow_metrics(
      [0, 2, 0], [0.0, 1.5, 0.1], [0.0, 0.4, 0.2])
  assert metrics['follow/n_anchors'] == 3.0
  assert abs(metrics['follow/pi_rank_mean'] - 2.0 / 3.0) < 1e-5
  assert abs(metrics['follow/pi_is_argmax_frac'] - 2.0 / 3.0) < 1e-5
  assert abs(metrics['follow/score_gap_argmax_minus_pi_mean'] - 1.6 / 3.0) < 1e-5


def test_resolve_task_goal_reachable_and_fallback():
  reachable = {
      'sawyer_handle_press_side': np.arange(7, dtype=np.float32),
  }
  goal = spp.resolve_task_goal(
      'sawyer_handle_press_side', 10, reachable_goals=reachable)
  assert goal.shape == (10,)
  assert np.allclose(goal[:7], np.arange(7))

  obs = np.concatenate([np.zeros(8), np.ones(8)]).astype(np.float32)
  goal2 = spp.resolve_task_goal('sawyer_push', 8, observation=obs)
  assert np.allclose(goal2, np.ones(8))


def test_is_near_object_unsolved_handle():
  hover = np.array(
      [-0.07, 0.70, 0.10, 0.3, -0.07, 0.70, 0.12], dtype=np.float32)
  successish = np.array(
      [-0.07, 0.70, 0.08, 0.3, -0.07, 0.70, 0.07], dtype=np.float32)
  assert spp.is_near_object_unsolved(hover, 'sawyer_handle_press_side')
  assert not spp.is_near_object_unsolved(
      successish, 'sawyer_handle_press_side')


def test_write_episode_to_adder():
  calls = []

  class FakeAdder:
    def reset(self):
      calls.append('reset')

    def add_first(self, ts):
      calls.append(('first', ts.step_type))

    def add(self, action, ts):
      calls.append(('add', float(action[0]), ts.step_type))

  first = dm_env.restart(np.zeros(4, dtype=np.float32))
  mid = dm_env.transition(
      reward=0.0, observation=np.ones(4, dtype=np.float32), discount=1.0)
  last = dm_env.termination(
      reward=1.0, observation=np.ones(4, dtype=np.float32) * 2)
  n = spp.write_episode_to_adder(
      FakeAdder(), first,
      [(np.array([0.1], dtype=np.float32), mid),
       (np.array([0.2], dtype=np.float32), last)])
  assert n == 2
  assert calls[0] == 'reset'
  assert calls[1][0] == 'first'
  assert calls[2][0] == 'add'
  assert calls[3][0] == 'add'


def test_sample_candidates_include_pi():
  rng = np.random.default_rng(0)
  pi = np.array([0.1, -0.2, 0.0, 0.3], dtype=np.float32)
  amin = -np.ones(4, dtype=np.float32)
  amax = np.ones(4, dtype=np.float32)
  cands = spp._sample_candidate_actions(
      pi, action_min=amin, action_max=amax, num_candidates=16,
      local_noise_std=0.05, rng=rng)
  assert cands.shape == (16, 4)
  assert np.allclose(cands[0], pi)


def test_stage_dwell_counts():
  labels = (['far'] * 80 + ['near_approach'] * 20 + ['hover_contact'] * 30
            + ['success'] * 20)
  metrics = spp.summarize_stage_dwell(labels)
  assert metrics['dwell/steps_far'] == 80
  assert metrics['dwell/steps_hover_contact'] == 30
  assert metrics['dwell/steps_success'] == 20
  assert metrics['dwell/steps_object_progress'] == 0
  assert abs(metrics['dwell/frac_success'] - 20 / 150) < 1e-6


def test_inject_clone_reaches_target():
  writes = []

  class FakeAdder:
    def reset(self):
      pass

    def add_first(self, ts):
      writes.append(('first', ts))

    def add(self, action, ts):
      writes.append(('add', action, ts))

  class FakeEnv:
    def __init__(self):
      self.i = 0

    def reset(self):
      self.i = 0
      return dm_env.restart(np.zeros(4, dtype=np.float32))

    def step(self, action):
      self.i += 1
      obs = np.ones(4, dtype=np.float32) * self.i
      if self.i >= 3:
        return dm_env.termination(reward=1.0, observation=obs)
      return dm_env.transition(
          reward=0.0, observation=obs, discount=1.0)

  class FakeActor:
    def select_action(self, obs):
      del obs
      return np.array([0.1], dtype=np.float32)

  metrics = spp.inject_successful_episodes(
      environment=FakeEnv(),
      actor=FakeActor(),
      adder=FakeAdder(),
      n_transitions=20,
      max_attempts=2,
      max_episode_steps=10,
      clone=True,
  )
  assert metrics['inject/n_transitions'] >= 20
  assert metrics['inject/n_success_episodes'] == 2
  assert metrics['inject/n_clones'] >= 1


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Success-propagation probe tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
