#!/usr/bin/env python3
"""Dependency-light tests for eval rollout video helpers."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import eval_video


class _FakeActor:
  def __init__(self):
    self._step = 0

  def observe_first(self, timestep):
    del timestep

  def select_action(self, observation):
    del observation
    return np.zeros(4, dtype=np.float32)

  def observe(self, action, next_timestep):
    del action, next_timestep


class _FakeEnv:
  def __init__(self, rewards):
    self._rewards = list(rewards)
    self._index = 0

  def reset(self):
    self._index = 0
    return self._timestep(reward=0.0, last=False)

  def step(self, action):
    del action
    self._index += 1
    if self._index >= len(self._rewards):
      return self._timestep(reward=0.0, last=True)
    return self._timestep(
        reward=self._rewards[self._index - 1], last=False)

  def _timestep(self, reward, last):
    class _Ts:
      observation = np.zeros(8, dtype=np.float32)
    ts = _Ts()
    ts.reward = reward
    ts.last = lambda: last
    return ts


def test_as_uint8_frame_scales_float01():
  frame = eval_video._as_uint8_frame(np.array([0.0, 0.5, 1.0]))
  assert frame.dtype == np.uint8
  assert tuple(frame.tolist()) == (0, 127, 255)


def test_record_episode_frames_counts_success():
  original = eval_video.render_rgb_array
  eval_video.render_rgb_array = lambda env: np.zeros((4, 4, 3), dtype=np.uint8)
  try:
    env = _FakeEnv(rewards=[0.0, 1.0, 0.0])
    frames, episode_return, success = eval_video.record_episode_frames(
        env, _FakeActor())
    assert frames.shape == (4, 4, 4, 3)
    assert episode_return == 1.0
    assert success == 1.0
  finally:
    eval_video.render_rgb_array = original


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Eval video tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
