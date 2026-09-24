"""Sparse-bit masks for the Success-BC ring.

``first_success_window`` clones only the actions that first caused success,
and only if the episode is still successful at the last transition. That
keeps the latch-task approach (Tasks 5/8) without copying a 150-step
prefix of a lucky multi-phase episode (Tasks 4/7).

``first_sparse_bit`` keeps only the first ``r>0`` transition. Later 1-bits
are linger / hold, not the press or place.

``critic_flatness_gate`` downweights BC when the existing contrastive
critic already ranks random actions at the observed (s, g). Probe
actions are sampled from the action box; the stored success action is
not used as a ranking label.
"""
from __future__ import annotations

import numpy as np


def first_success_window_mask(step_reward, window):
  """Keep ``[t_first - K + 1, t_first]`` iff the last sparse bit is 1.

  Args:
    step_reward: 1D array of per-transition env rewards, length ``T``.
    window: number of transitions to keep, ending at the first ``r>0``.

  Returns:
    Float32 mask of length ``T``.
  """
  step_reward = np.asarray(step_reward, dtype=np.float32)
  if step_reward.ndim != 1:
    raise ValueError('step_reward must be 1D')
  length = int(step_reward.shape[0])
  keep = np.zeros(length, dtype=np.float32)
  if length == 0 or int(window) <= 0:
    return keep
  if not (step_reward[-1] > 0.0):
    return keep
  positive = np.flatnonzero(step_reward > 0.0)
  if positive.size == 0:
    return keep
  t_first = int(positive[0])
  start = max(t_first - int(window) + 1, 0)
  keep[start:t_first + 1] = 1.0
  return keep


def first_sparse_bit_mask(step_reward):
  """Keep only the first transition whose sparse bit is 1.

  Later 1-bits (hover after a press, linger in the goal ball) are
  dropped. Episodes with no positive bit keep nothing. Uses the env
  0/1 bit only.
  """
  step_reward = np.asarray(step_reward, dtype=np.float32)
  if step_reward.ndim != 1:
    raise ValueError('step_reward must be 1D')
  keep = np.zeros(step_reward.shape[0], dtype=np.float32)
  if keep.size == 0:
    return keep
  positive = np.flatnonzero(step_reward > 0.0)
  if positive.size == 0:
    return keep
  keep[int(positive[0])] = 1.0
  return keep


def critic_flatness_gate(sigma_action, sigma_state, eps=1e-6):
  """BC weight from critic action-sensitivity, not from a stored action.

  ``sigma_action`` is the std of score(s, a, g) over random actions at a
  fixed observed (s, g). ``sigma_state`` is the std of the same scores'
  action-average across the BC batch. Gate → 1 when the critic is flat
  in a (cannot rank), → 0 when action variation dominates. Both-zero
  (hover cluster, flat critic) clones. No a_succ, hover radius, or task
  name.
  """
  sigma_action = np.asarray(sigma_action, dtype=np.float32)
  sigma_state = np.asarray(sigma_state, dtype=np.float32)
  denom = sigma_state + sigma_action
  safe = np.maximum(denom, float(eps))
  gate = np.where(
      denom > float(eps),
      sigma_state / safe,
      np.ones_like(denom, dtype=np.float32))
  return np.asarray(gate, dtype=np.float32)


def recency_logits(size, index, capacity, half_life, episode_len=150):
  """Logits that overweight recent ring slots (half-life in episodes).

  Age 0 is the newest filled slot. Empty slots get a large negative logit.
  ``half_life <= 0`` is uniform over filled slots.
  """
  size = int(size)
  index = int(index)
  capacity = int(capacity)
  logits = np.full(capacity, -1e9, dtype=np.float32)
  if size <= 0:
    return logits
  newest = (index - 1) % capacity
  positions = np.arange(capacity)
  age = (newest - positions) % capacity
  filled = age < size
  if float(half_life) <= 0.0:
    logits[filled] = 0.0
    return logits
  denom = float(half_life) * float(episode_len)
  logits[filled] = np.float32(
      -np.log(2.0) * age[filled] / max(denom, 1.0))
  return logits


def warmup_scale(buffer_size, warmup_episodes, episode_len=150):
  """Scale λ up from 0 toward 1 as D_succ fills with whole episodes.

  Does not delay BC: the first terminal-success episode already clones,
  at λ * (episode_len / (N0 * episode_len)) = λ / N0. Uniform sampling
  is unchanged. ``warmup_episodes <= 0`` disables the scale (paper λ).
  """
  if int(warmup_episodes) <= 0:
    return 1.0
  if int(episode_len) <= 0:
    raise ValueError('episode_len must be positive.')
  return float(min(
      1.0,
      float(buffer_size) / float(int(warmup_episodes) * int(episode_len))))


def tensorflow_first_success_window(tf, reward, window):
  """TF mask over ``reward[:-1]`` for the replay flatten functions."""
  step_reward = reward[:-1]
  length = tf.shape(step_reward)[0]
  positive = step_reward > 0.0
  any_positive = tf.reduce_any(positive)
  t_first = tf.argmax(tf.cast(positive, tf.int32), output_type=tf.int32)
  terminal_ok = step_reward[length - 1] > 0.0
  idx = tf.range(length, dtype=tf.int32)
  start = tf.maximum(t_first - (tf.cast(window, tf.int32) - 1), 0)
  in_window = (idx >= start) & (idx <= t_first)
  return tf.cast(terminal_ok & any_positive & in_window, tf.float32)


def tensorflow_first_sparse_bit(tf, reward):
  """TF mask over ``reward[:-1]``: 1 only at the first sparse hit."""
  step_reward = reward[:-1]
  positive = step_reward > 0.0
  any_positive = tf.reduce_any(positive)
  t_first = tf.argmax(tf.cast(positive, tf.int32), output_type=tf.int32)
  idx = tf.range(tf.shape(step_reward)[0], dtype=tf.int32)
  return tf.cast(any_positive & (idx == t_first), tf.float32)
