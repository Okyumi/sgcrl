"""Finite-horizon raw outcome labels for Outcome-Calibrated Sequence DCC."""
from __future__ import annotations

import argparse


def tensorflow_finite_horizon_labels(
    tf, all_state, anchor_index, goal, *, horizon: int, threshold: float):
  """Return mechanism progress and H-step reachability for each anchor.

  The caller supplies TensorFlow explicitly so dependency-light config tests
  can import this module before the cluster Conda environment is activated.
  """
  if horizon <= 0:
    raise ValueError('horizon must be positive')
  seq_len = tf.shape(all_state)[0]
  offsets = tf.range(1, horizon + 1, dtype=tf.int32)
  future_index = anchor_index[:, None] + offsets[None, :]
  valid = future_index < seq_len
  clipped = tf.minimum(future_index, seq_len - 1)
  future_mechanism = tf.gather(all_state[:, 4:7], clipped)
  goal_mechanism = goal[:, None, 4:7]
  distance = tf.linalg.norm(future_mechanism - goal_mechanism, axis=2)
  large = tf.fill(tf.shape(distance), tf.constant(1e6, tf.float32))
  valid_distance = tf.where(valid, distance, large)
  current_mechanism = tf.gather(all_state[:, 4:7], anchor_index)
  current_distance = tf.linalg.norm(
      current_mechanism - goal[:, 4:7], axis=1)
  best_distance = tf.minimum(
      current_distance, tf.reduce_min(valid_distance, axis=1))
  progress = current_distance - best_distance
  success = tf.reduce_any(
      tf.logical_and(valid, distance <= threshold), axis=1)
  return progress, tf.cast(success, tf.float32)


def _self_test():
  try:
    import tensorflow as tf  # pylint: disable=import-outside-toplevel
  except ImportError as error:
    raise SystemExit(
        'TensorFlow is unavailable. Activate the contrastive_rl Conda '
        'environment before running the outcome-credit self-test.') from error
  state = tf.zeros((5, 7), dtype=tf.float32)
  state = tf.tensor_scatter_nd_update(
      state, [[i, 4] for i in range(5)],
      tf.constant([0.0, 0.1, 0.2, 0.3, 0.4], dtype=tf.float32))
  anchors = tf.constant([0, 1], dtype=tf.int32)
  goal = tf.zeros((2, 7), dtype=tf.float32)
  goal = tf.tensor_scatter_nd_update(
      goal, [[0, 4], [1, 4]], tf.constant([0.3, 0.4], tf.float32))
  progress, success = tensorflow_finite_horizon_labels(
      tf, state, anchors, goal, horizon=2, threshold=0.11)
  tf.debugging.assert_near(progress, [0.2, 0.2], atol=1e-6)
  tf.debugging.assert_equal(success, [1.0, 1.0])
  print('outcome-credit TensorFlow self-test passed')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--self-test', action='store_true', required=True)
  parser.parse_args()
  _self_test()


if __name__ == '__main__':
  main()
