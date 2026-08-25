"""Goal contracts for the custom goal-conditioned Sawyer wrappers.

The continual wrappers expose an observation as ``state || desired_goal``.
Historically the desired goal copied the full seven-coordinate Sawyer state,
including a hand pose and gripper opening that are *not* part of MetaWorld's
success predicate.  This module makes the two supported contracts explicit
and keeps the task-success mechanism slice dependency-light and testable.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


GOAL_CONDITIONING_MODES = ('full_state', 'success_mechanism')
SUCCESS_MECHANISM_TASKS = (
    'sawyer_handle_press_side',
    'sawyer_window_close',
)
MECHANISM_STATE_SLICE = slice(4, 7)
VALIDITY_TASKS = {
    'sawyer_handle_press_side': {
        'native_env_name': 'handle-press-side-v2',
        'fixed_goal': np.asarray([-0.07, 0.68, 0.07], dtype=np.float32),
        'success_threshold': 0.02,
        'policy_class': 'SawyerHandlePressSideV2Policy',
    },
    'sawyer_window_close': {
        'native_env_name': 'window-close-v2',
        'fixed_goal': np.asarray([0.0, 0.80, 0.20], dtype=np.float32),
        'success_threshold': 0.05,
        'policy_class': 'SawyerWindowCloseV2Policy',
    },
}


def resolve_goal_slice(mode: str, env_name: str) -> Tuple[int, int]:
  """Return the state slice used as a desired/achieved goal.

  ``full_state`` preserves the historical wrapper contract.  The
  ``success_mechanism`` validity ablation exposes only state coordinates
  ``[4:7]``, which are the handle coordinates used by the Task-5 and Task-8
  success predicates.
  """
  if mode not in GOAL_CONDITIONING_MODES:
    raise ValueError(
        f'Unknown goal conditioning mode {mode!r}; expected one of '
        f'{GOAL_CONDITIONING_MODES}.')
  if mode == 'full_state':
    return 0, -1
  if env_name not in SUCCESS_MECHANISM_TASKS:
    raise ValueError(
        'success_mechanism is currently validated only for '
        f'{SUCCESS_MECHANISM_TASKS}; got {env_name!r}.')
  return MECHANISM_STATE_SLICE.start, MECHANISM_STATE_SLICE.stop


def split_observation(observation: np.ndarray,
                      obs_dim: int) -> Tuple[np.ndarray, np.ndarray]:
  """Split one or more ``state || goal`` observations along the last axis."""
  observation = np.asarray(observation)
  if observation.shape[-1] <= obs_dim:
    raise ValueError(
        f'Observation width {observation.shape[-1]} has no goal after '
        f'obs_dim={obs_dim}.')
  return observation[..., :obs_dim], observation[..., obs_dim:]


def mechanism_from_state(state: np.ndarray) -> np.ndarray:
  """Extract Sawyer mechanism coordinates from a state vector."""
  state = np.asarray(state)
  if state.shape[-1] < MECHANISM_STATE_SLICE.stop:
    raise ValueError(
        f'Sawyer state needs at least 7 coordinates; got {state.shape[-1]}.')
  return state[..., MECHANISM_STATE_SLICE]


def mechanism_from_goal(goal: np.ndarray) -> np.ndarray:
  """Extract mechanism coordinates from either supported goal contract."""
  goal = np.asarray(goal)
  if goal.shape[-1] == 3:
    return goal
  if goal.shape[-1] >= MECHANISM_STATE_SLICE.stop:
    return goal[..., MECHANISM_STATE_SLICE]
  raise ValueError(
      'Goal must be either a 3-D success mechanism or a full state with at '
      f'least 7 coordinates; got {goal.shape[-1]}.')


def mechanism_distance(observation: np.ndarray, obs_dim: int) -> float:
  """Distance between achieved and desired success-mechanism coordinates."""
  state, goal = split_observation(observation, obs_dim)
  return float(np.linalg.norm(
      mechanism_from_state(state) - mechanism_from_goal(goal)))


def goal_contract_metrics(observation: np.ndarray, obs_dim: int,
                          internal_target: np.ndarray) -> dict[str, float]:
  """Measure whether an exposed goal agrees with the wrapper target.

  The hand and gripper mismatches are descriptive rather than pass/fail
  criteria: those coordinates are intentionally *not* in the benchmark
  success predicate and are the suspected invalid part of the old contract.
  """
  state, goal = split_observation(observation, obs_dim)
  mechanism_goal = mechanism_from_goal(goal)
  internal_target = np.asarray(internal_target)
  metrics = {
      'target_linf_error': float(np.max(np.abs(
          mechanism_goal - internal_target))),
      'mechanism_distance': float(np.linalg.norm(
          mechanism_from_state(state) - mechanism_goal)),
  }
  if goal.shape[-1] >= 7:
    metrics.update({
        'hand_goal_distance': float(np.linalg.norm(state[:3] - goal[:3])),
        'gripper_goal_error': float(abs(float(state[3] - goal[3]))),
    })
  return metrics
