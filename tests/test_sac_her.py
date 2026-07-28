"""Tests for the sparse HER reward and terminal-discount rule (``sac.her``).

Exercised through the numpy backend, so no TensorFlow is required; the driver
uses the same function with :func:`sac.her.tensorflow_ops`.
"""
import numpy as np
import pytest

from sac import her

TAU = 0.05


def _reward(achieved, goal, env_discount=0.99, threshold=TAU,
            step_penalty_reward=True):
  return her.her_reward_and_discount(
      np.asarray(achieved, dtype=np.float32),
      np.asarray(goal, dtype=np.float32),
      np.asarray(env_discount, dtype=np.float32),
      threshold=threshold,
      step_penalty_reward=step_penalty_reward)


def test_step_penalty_shape_is_zero_on_reach_and_minus_one_otherwise():
  achieved = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
  goal = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  reward, _ = _reward(achieved, goal, env_discount=[0.99, 0.99])
  np.testing.assert_allclose(reward, [0.0, -1.0])


def test_sparse01_shape_is_one_on_reach_and_zero_otherwise():
  achieved = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
  goal = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  reward, _ = _reward(achieved, goal, env_discount=[0.99, 0.99],
                      step_penalty_reward=False)
  np.testing.assert_allclose(reward, [1.0, 0.0])


@pytest.mark.parametrize('step_penalty_reward', [True, False])
def test_discount_is_zeroed_on_goal_reach_for_both_reward_shapes(
    step_penalty_reward):
  """Terminal bootstrap: reaching the goal cuts the TD target's tail."""
  achieved = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
  goal = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  _, discount = _reward(achieved, goal, env_discount=[0.99, 0.99],
                        step_penalty_reward=step_penalty_reward)
  np.testing.assert_allclose(discount, [0.0, 0.99])


def test_discount_preserves_env_discount_when_goal_not_reached():
  """A mid-episode env discount of 0 (true terminal) must stay 0."""
  achieved = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
  goal = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  _, discount = _reward(achieved, goal, env_discount=[0.99, 0.0])
  np.testing.assert_allclose(discount, [0.99, 0.0])


def test_threshold_is_strict_inequality_on_euclidean_norm():
  # Distance exactly tau is NOT a reach; just inside tau is.
  reward_at, _ = _reward([[TAU, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
  reward_inside, _ = _reward([[TAU - 1e-3, 0.0, 0.0]], [[0.0, 0.0, 0.0]])
  np.testing.assert_allclose(reward_at, [-1.0])
  np.testing.assert_allclose(reward_inside, [0.0])


def test_distance_uses_all_goal_dimensions():
  """A per-axis offset under tau can still exceed tau in Euclidean norm."""
  offset = TAU * 0.7  # 0.035 per axis -> norm 0.0606 > tau over 3 axes
  reward, _ = _reward([[offset, offset, offset]], [[0.0, 0.0, 0.0]])
  np.testing.assert_allclose(reward, [-1.0])


def test_larger_threshold_admits_looser_reaches():
  """--her_reward_threshold=0.12 is the documented knob for stick_pull."""
  achieved = [[0.1, 0.0, 0.0]]
  goal = [[0.0, 0.0, 0.0]]
  tight, _ = _reward(achieved, goal, threshold=0.05)
  loose, _ = _reward(achieved, goal, threshold=0.12)
  np.testing.assert_allclose(tight, [-1.0])
  np.testing.assert_allclose(loose, [0.0])


def test_output_dtypes_and_shapes_match_the_batch():
  batch = 7
  rng = np.random.default_rng(0)
  achieved = rng.normal(size=(batch, 3)).astype(np.float32)
  goal = rng.normal(size=(batch, 3)).astype(np.float32)
  env_discount = np.full((batch,), 0.99, dtype=np.float32)
  reward, discount = _reward(achieved, goal, env_discount)
  assert reward.shape == (batch,)
  assert discount.shape == (batch,)
  assert reward.dtype == np.float32
  assert discount.dtype == np.float32


def test_reward_shapes_differ_by_exactly_a_constant_shift():
  """sparse01 = steppen + 1 everywhere, so only the offset changes."""
  rng = np.random.default_rng(1)
  achieved = rng.normal(scale=0.05, size=(32, 3)).astype(np.float32)
  goal = np.zeros((32, 3), dtype=np.float32)
  env_discount = np.full((32,), 0.99, dtype=np.float32)
  steppen, disc_a = _reward(achieved, goal, env_discount)
  sparse01, disc_b = _reward(achieved, goal, env_discount,
                             step_penalty_reward=False)
  np.testing.assert_allclose(sparse01, steppen + 1.0)
  np.testing.assert_allclose(disc_a, disc_b)
