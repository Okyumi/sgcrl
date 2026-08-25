"""Paired legacy/axis success scoring for Task-5/Task-8 rollouts.

The functions in this module are deliberately independent of MetaWorld, Acme,
and JAX so their semantics can be tested on a login node.  The observer is
duck-typed to Acme's ``EnvLoopObserver`` interface by the evaluation script.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Task58SuccessSpec:
  axis: int
  threshold: float


TASK58_SUCCESS_SPECS: Mapping[str, Task58SuccessSpec] = {
    'sawyer_handle_press_side': Task58SuccessSpec(axis=2, threshold=0.02),
    'sawyer_window_close': Task58SuccessSpec(axis=0, threshold=0.05),
}

MECHANISM_START = 4
MECHANISM_END = 7


def success_distances(observation, obs_dim: int, env_name: str):
  """Return historical full-3D and corrected task-axis distances.

  ``observation`` must use the historical full-state contract
  ``[state, goal]``.  The mechanism occupies coordinates ``state[4:7]`` and
  the corresponding goal coordinates ``goal[4:7]`` for both tasks.
  """
  if env_name not in TASK58_SUCCESS_SPECS:
    raise ValueError(
        f'Paired Task-5/Task-8 scoring does not support {env_name!r}.')
  observation = np.asarray(observation)
  if observation.ndim != 1:
    raise ValueError(
        f'Expected one unbatched observation, got shape {observation.shape}.')
  if obs_dim < MECHANISM_END or observation.size < obs_dim + MECHANISM_END:
    raise ValueError(
        'The full-state goal contract must expose state[4:7] and goal[4:7]; '
        f'got obs_dim={obs_dim}, observation size={observation.size}.')

  mechanism = observation[MECHANISM_START:MECHANISM_END]
  goal = observation[
      obs_dim + MECHANISM_START:obs_dim + MECHANISM_END]
  spec = TASK58_SUCCESS_SPECS[env_name]
  legacy_distance = float(np.linalg.norm(mechanism - goal))
  axis_distance = abs(float(mechanism[spec.axis] - goal[spec.axis]))
  return legacy_distance, axis_distance


def success_flags(observation, obs_dim: int, env_name: str):
  """Return ``(legacy_success, task_axis_success)`` for one state.

  The strict historical comparison matches ``env_utils.py``.  The inclusive
  axis comparison matches the official Task-5/Task-8 thresholds and the new
  ``task_axis`` wrapper mode.
  """
  legacy_distance, axis_distance = success_distances(
      observation, obs_dim, env_name)
  threshold = TASK58_SUCCESS_SPECS[env_name].threshold
  return legacy_distance < threshold, axis_distance <= threshold


class PairedTask58SuccessObserver:
  """Score both success definitions on exactly the same episode trajectory."""

  def __init__(self, obs_dim: int, env_name: str):
    if env_name not in TASK58_SUCCESS_SPECS:
      raise ValueError(
          f'Paired Task-5/Task-8 scoring does not support {env_name!r}.')
    self._obs_dim = int(obs_dim)
    self._env_name = env_name
    self._legacy_successes = []
    self._axis_successes = []
    self._legacy_distances = []
    self._axis_distances = []
    self._reward_mismatches = 0

  def observe_first(self, env, timestep):
    del env, timestep
    self._legacy_successes = []
    self._axis_successes = []
    self._legacy_distances = []
    self._axis_distances = []
    self._reward_mismatches = 0

  def observe(self, env, timestep, action):
    del env, action
    legacy_distance, axis_distance = success_distances(
        timestep.observation, self._obs_dim, self._env_name)
    threshold = TASK58_SUCCESS_SPECS[self._env_name].threshold
    legacy_success = legacy_distance < threshold
    axis_success = axis_distance <= threshold
    self._legacy_distances.append(legacy_distance)
    self._axis_distances.append(axis_distance)
    self._legacy_successes.append(legacy_success)
    self._axis_successes.append(axis_success)

    # The evaluation environment deliberately runs in legacy mode.  This
    # verifies that the observation-based paired scorer exactly reproduces
    # the old metric before comparing it with the corrected metric.
    emitted_success = float(timestep.reward or 0.0) > 0.5
    self._reward_mismatches += int(emitted_success != legacy_success)

  def get_metrics(self):
    legacy_success = any(self._legacy_successes)
    axis_success = any(self._axis_successes)
    return {
        'legacy_success': float(legacy_success),
        'task_axis_success': float(axis_success),
        'axis_rescued_success': float(axis_success and not legacy_success),
        'legacy_min_distance': float(min(
            self._legacy_distances, default=np.inf)),
        'task_axis_min_distance': float(min(
            self._axis_distances, default=np.inf)),
        'legacy_reward_mismatch_steps': float(self._reward_mismatches),
    }
