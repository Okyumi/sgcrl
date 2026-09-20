#!/usr/bin/env python3
"""Gym-only tests for collection-time success truncation.

Avoids importing ``env_utils`` (MetaWorld / mujoco_py) so this can run on
a login node. The wrapper copy must stay in lockstep with
``env_utils.SuccessTruncateGymWrapper``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import gym


def _sparse_success_from_step(reward, info) -> bool:
  if info is not None and 'success' in info:
    return float(info['success']) >= 1.0
  return float(reward) >= 1.0


class SuccessTruncateGymWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self._frozen = False
    self._last_obs = None

  def reset(self, **kwargs):
    self._frozen = False
    self._last_obs = self.env.reset(**kwargs)
    return self._last_obs

  def step(self, action):
    if self._frozen:
      info = {
          'success': 1.0,
          'truncated_on_success': 1.0,
          'frozen_after_success': 1.0,
      }
      return self._last_obs, 0.0, False, info
    result = self.env.step(action)
    obs, reward, done, info = result
    success = _sparse_success_from_step(reward, info)
    if info is not None:
      info = dict(info)
      info['truncated_on_success'] = float(success)
      info['frozen_after_success'] = float(success)
    if success:
      self._frozen = True
    self._last_obs = obs
    return obs, reward, done, info


class _FakeEnv(gym.Env):
  def __init__(self, rewards):
    self.observation_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    self._rewards = list(rewards)
    self._index = 0

  def reset(self, **kwargs):
    del kwargs
    self._index = 0
    return np.zeros(4, dtype=np.float32)

  def step(self, action):
    del action
    reward = float(self._rewards[self._index])
    self._index += 1
    last = self._index >= len(self._rewards)
    info = {'success': float(reward >= 1.0)}
    return np.zeros(4, dtype=np.float32), reward, last, info


def test_source_contains_production_wrapper():
  text = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  assert 'class SuccessTruncateGymWrapper' in text
  assert 'truncate_on_success' in text
  assert 'frozen_after_success' in text
  train = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  assert 'truncate_on_success=truncate_on_success' in train
  assert 'eval stays live' in train


def test_sparse_success_prefers_info():
  assert _sparse_success_from_step(0.0, {'success': 1.0}) is True
  assert _sparse_success_from_step(1.0, {'success': 0.0}) is False
  assert _sparse_success_from_step(1.0, {}) is True


def test_wrapper_ends_on_first_success_and_keeps_prefix():
  env = SuccessTruncateGymWrapper(_FakeEnv([0.0, 0.0, 1.0, 0.0]))
  env.reset()
  _, reward, done, info = env.step(np.zeros(2))
  assert reward == 0.0 and done is False
  _, reward, done, info = env.step(np.zeros(2))
  assert reward == 0.0 and done is False
  _, reward, done, info = env.step(np.zeros(2))
  assert reward == 1.0
  assert done is False
  assert info['truncated_on_success'] == 1.0
  assert env.env._index == 3
  frozen_obs, frozen_reward, frozen_done, frozen_info = env.step(np.ones(2))
  assert frozen_reward == 0.0
  assert frozen_done is False
  assert frozen_info['frozen_after_success'] == 1.0
  assert env.env._index == 3
  assert np.allclose(frozen_obs, np.zeros(4))


def test_wrapper_does_not_end_without_success():
  env = SuccessTruncateGymWrapper(_FakeEnv([0.0, 0.0, 0.0]))
  env.reset()
  done = False
  for _ in range(3):
    _, reward, done, info = env.step(np.zeros(2))
    assert reward == 0.0
    assert info['truncated_on_success'] == 0.0
  assert done is True


if __name__ == '__main__':
  test_source_contains_production_wrapper()
  test_sparse_success_prefers_info()
  test_wrapper_ends_on_first_success_and_keeps_prefix()
  test_wrapper_does_not_end_without_success()
  print('test_success_truncate: ok')
