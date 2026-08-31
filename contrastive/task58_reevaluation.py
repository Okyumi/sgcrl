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


def success_distances(observation, obs_dim: int, env_name: str,
                      mechanism_target=None):
  """Return historical full-3D and corrected task-axis distances.

  ``observation`` must use the historical full-state contract
  ``[state, goal]``.  The mechanism occupies coordinates ``state[4:7]`` and
  the corresponding goal coordinates ``goal[4:7]`` for both tasks.

  ``mechanism_target`` is the authoritative task target used by the wrapper.
  It must be supplied when the exposed full-state conditioning goal is a
  captured reachable state whose mechanism coordinates need not equal that
  target exactly.
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
  if mechanism_target is None:
    target = observation[
        obs_dim + MECHANISM_START:obs_dim + MECHANISM_END]
  else:
    target = np.asarray(mechanism_target)
    if target.shape != (3,):
      raise ValueError(
          f'mechanism_target must have shape (3,), got {target.shape}.')
  spec = TASK58_SUCCESS_SPECS[env_name]
  legacy_distance = float(np.linalg.norm(mechanism - target))
  axis_distance = abs(float(mechanism[spec.axis] - target[spec.axis]))
  return legacy_distance, axis_distance


def success_flags(observation, obs_dim: int, env_name: str,
                  mechanism_target=None):
  """Return ``(legacy_success, task_axis_success)`` for one state.

  The strict historical comparison matches ``env_utils.py``.  The inclusive
  axis comparison matches the official Task-5/Task-8 thresholds and the new
  ``task_axis`` wrapper mode.
  """
  legacy_distance, axis_distance = success_distances(
      observation, obs_dim, env_name, mechanism_target)
  threshold = TASK58_SUCCESS_SPECS[env_name].threshold
  return legacy_distance < threshold, axis_distance <= threshold


class PairedTask58SuccessObserver:
  """Score success and failure stage on one Task-5/Task-8 trajectory."""

  def __init__(self, obs_dim: int, env_name: str,
               emitted_success_mode: str = 'legacy_distance',
               interaction_threshold: float = 0.09,
               movement_threshold: float = 0.005,
               mechanism_target=None):
    if env_name not in TASK58_SUCCESS_SPECS:
      raise ValueError(
          f'Paired Task-5/Task-8 scoring does not support {env_name!r}.')
    self._obs_dim = int(obs_dim)
    self._env_name = env_name
    if emitted_success_mode not in ('legacy_distance', 'corrected'):
      raise ValueError(
          'emitted_success_mode must be legacy_distance or corrected; got '
          f'{emitted_success_mode!r}.')
    self._emitted_success_mode = emitted_success_mode
    self._mechanism_target = (
        None if mechanism_target is None
        else np.asarray(mechanism_target, dtype=np.float32).copy())
    if (emitted_success_mode == 'corrected'
        and self._mechanism_target is None):
      raise ValueError(
          'Corrected Task-5/Task-8 scoring requires the wrapper mechanism '
          'target; the full-state conditioning goal is not a success target.')
    if (self._mechanism_target is not None
        and self._mechanism_target.shape != (3,)):
      raise ValueError(
          'mechanism_target must have shape (3,), got '
          f'{self._mechanism_target.shape}.')
    self._interaction_threshold = float(interaction_threshold)
    self._movement_threshold = float(movement_threshold)
    self._legacy_successes = []
    self._axis_successes = []
    self._legacy_distances = []
    self._axis_distances = []
    self._hand_mechanism_distances = []
    self._mechanism_axis_displacements = []
    self._initial_axis_distance = np.inf
    self._initial_mechanism_axis = 0.0
    self._reward_mismatches = 0

  def observe_first(self, env, timestep):
    del env
    self._legacy_successes = []
    self._axis_successes = []
    self._legacy_distances = []
    self._axis_distances = []
    self._hand_mechanism_distances = []
    self._mechanism_axis_displacements = []
    self._reward_mismatches = 0
    observation = np.asarray(timestep.observation)
    _, self._initial_axis_distance = success_distances(
        observation, self._obs_dim, self._env_name, self._mechanism_target)
    spec = TASK58_SUCCESS_SPECS[self._env_name]
    self._initial_mechanism_axis = float(
        observation[MECHANISM_START + spec.axis])

  def observe(self, env, timestep, action):
    del env, action
    legacy_distance, axis_distance = success_distances(
        timestep.observation, self._obs_dim, self._env_name,
        self._mechanism_target)
    threshold = TASK58_SUCCESS_SPECS[self._env_name].threshold
    legacy_success = legacy_distance < threshold
    axis_success = axis_distance <= threshold
    self._legacy_distances.append(legacy_distance)
    self._axis_distances.append(axis_distance)
    self._legacy_successes.append(legacy_success)
    self._axis_successes.append(axis_success)
    observation = np.asarray(timestep.observation)
    hand = observation[:3]
    mechanism = observation[MECHANISM_START:MECHANISM_END]
    spec = TASK58_SUCCESS_SPECS[self._env_name]
    self._hand_mechanism_distances.append(float(
        np.linalg.norm(hand - mechanism)))
    self._mechanism_axis_displacements.append(abs(
        float(mechanism[spec.axis]) - self._initial_mechanism_axis))

    emitted_success = float(timestep.reward or 0.0) > 0.5
    expected_success = (
        legacy_success if self._emitted_success_mode == 'legacy_distance'
        else axis_success)
    self._reward_mismatches += int(emitted_success != expected_success)

  def get_metrics(self):
    legacy_success = any(self._legacy_successes)
    axis_success = any(self._axis_successes)
    interaction_steps = sum(
        distance <= self._interaction_threshold
        for distance in self._hand_mechanism_distances)
    max_axis_displacement = max(
        self._mechanism_axis_displacements, default=0.0)
    min_axis_distance = min(self._axis_distances, default=np.inf)
    return {
        'legacy_success': float(legacy_success),
        'task_axis_success': float(axis_success),
        'axis_rescued_success': float(axis_success and not legacy_success),
        'initial_task_axis_distance': float(self._initial_axis_distance),
        'legacy_min_distance': float(min(
            self._legacy_distances, default=np.inf)),
        'task_axis_min_distance': float(min(
            self._axis_distances, default=np.inf)),
        'success_reward_mismatch_steps': float(self._reward_mismatches),
        # Backward-compatible key consumed by the historical-checkpoint
        # reevaluator, whose emitted mode is always legacy_distance.
        'legacy_reward_mismatch_steps': float(self._reward_mismatches),
        'approach_success': float(interaction_steps > 0),
        'interaction_step_fraction': float(
            interaction_steps / max(len(self._hand_mechanism_distances), 1)),
        'minimum_hand_mechanism_distance': float(min(
            self._hand_mechanism_distances, default=np.inf)),
        'mechanism_moved': float(
            max_axis_displacement >= self._movement_threshold),
        'max_mechanism_axis_displacement': float(max_axis_displacement),
        'max_task_axis_progress': float(
            self._initial_axis_distance - min_axis_distance),
    }
