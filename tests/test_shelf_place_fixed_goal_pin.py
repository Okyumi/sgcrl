#!/usr/bin/env python3
"""The Task-7 wrapper must pin the shelf mesh to the commanded goal."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

FIXED_GOAL = np.array([0.02, 0.89, 0.30], dtype=np.float32)
SHELF_Z_OFFSET = np.array([0.0, 0.0, 0.3], dtype=np.float32)


def _wrapper_source():
  source = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  return source.split('class SawyerShelfPlace(', 1)[1].split(
      'class SawyerWindowClose(', 1)[0]


def test_reset_pins_shelf_body_to_fixed_goal():
  wrapper = _wrapper_source()
  reset = wrapper.split('def reset(self):', 1)[1].split('def step(', 1)[0]
  assert "body_name2id('shelf')" in reset
  assert 'self._fixed_start_end is not None' in reset
  assert 'self._goal - np.array([0.0, 0.0, 0.3]' in reset
  assert 'sawyer_success.synchronize_simulator_after_reset(self)' in reset
  assert '_freeze_rand_vec' not in reset
  assert "body_name2id('obj')" not in reset


def test_fixed_goal_keeps_shelf_and_randomizes_object():
  try:
    from env_utils import SawyerShelfPlace
  except Exception:
    return

  env = SawyerShelfPlace(fixed_start_end=FIXED_GOAL)
  shelf_id = env.model.body_name2id('shelf')
  shelves = []
  objects = []
  for _ in range(5):
    obs = env.reset()
    np.testing.assert_allclose(env._goal, FIXED_GOAL, atol=1e-6)
    np.testing.assert_allclose(env._target_pos, FIXED_GOAL, atol=1e-6)
    np.testing.assert_allclose(obs[15:18], FIXED_GOAL, atol=1e-6)
    shelf = np.array(env.sim.model.body_pos[shelf_id], dtype=np.float32)
    np.testing.assert_allclose(shelf, FIXED_GOAL - SHELF_Z_OFFSET, atol=1e-6)
    site = np.array(env._get_site_pos('goal'), dtype=np.float32)
    np.testing.assert_allclose(site, FIXED_GOAL, atol=1e-5)
    shelves.append(shelf)
    objects.append(np.array(env._get_pos_objects(), dtype=np.float32))

  assert max(np.linalg.norm(s - shelves[0]) for s in shelves) < 1e-6
  assert max(np.linalg.norm(o[:2] - objects[0][:2]) for o in objects) > 1e-3


if __name__ == '__main__':
  test_reset_pins_shelf_body_to_fixed_goal()
  test_fixed_goal_keeps_shelf_and_randomizes_object()
  print('ok')
