"""Shared outcome semantics for counterfactual Sawyer experiments.

The legacy causal probe and the task-goal rank collector used different
success predicates and different progress summaries.  This module keeps the
definitions dependency-light and explicit so training, diagnostics, oracle
selection, and closed-loop execution can share them.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


SUCCESS_MODES = ('goal_distance', 'zero_reward', 'positive_reward')


def goal_distances(observation: np.ndarray, obs_dim: int) -> Tuple[float, float]:
  """Return full-state and mechanism-coordinate distances to the goal."""
  observation = np.asarray(observation, dtype=np.float64)
  state = observation[:obs_dim]
  goal = observation[obs_dim:]
  shared = min(state.shape[0], goal.shape[0])
  full = float(np.linalg.norm(state[:shared] - goal[:shared]))
  if shared < 7:
    return full, full
  mechanism = float(np.linalg.norm(state[4:7] - goal[4:7]))
  return full, mechanism


def interaction_distance(observation: np.ndarray, obs_dim: int) -> float:
  """Return the Sawyer hand-to-mechanism distance."""
  state = np.asarray(observation, dtype=np.float64)[:obs_dim]
  if state.shape[0] < 7:
    return float('nan')
  return float(np.linalg.norm(state[:3] - state[4:7]))


def interaction_phase(distance: float, contact_threshold: float,
                      precontact_margin: float = 0.03) -> str:
  """Coarse approach/pre-contact/contact phase used only for diagnostics."""
  if not np.isfinite(distance):
    return 'unknown'
  if distance <= contact_threshold:
    return 'contact'
  if distance <= contact_threshold + precontact_margin:
    return 'precontact'
  return 'approach'


def mechanism_proxy_success(observation: np.ndarray, obs_dim: int,
                            threshold: float) -> float:
  """Distance-proxy success used by earlier counterfactual labels."""
  _, mechanism = goal_distances(observation, obs_dim)
  return float(mechanism <= float(threshold))


def benchmark_success(timestep, observation: np.ndarray, obs_dim: int,
                      threshold: float, mode: str) -> Tuple[float, float]:
  """Return selected success and whether an independent signal was available.

  ``zero_reward`` matches this project's sparse step-penalty convention:
  failure has reward -1 and success has reward 0.  ``positive_reward`` is
  retained for environments that emit 0/+1.  ``goal_distance`` deliberately
  selects the mechanism proxy and reports availability zero because it is not
  independent of that proxy.
  """
  if mode not in SUCCESS_MODES:
    raise ValueError(
        f'Unknown success mode {mode!r}; expected one of {SUCCESS_MODES}.')
  proxy = mechanism_proxy_success(observation, obs_dim, threshold)
  if mode == 'goal_distance':
    return proxy, 0.0
  reward = getattr(timestep, 'reward', None)
  if reward is None:
    return proxy, 0.0
  reward = float(reward)
  if mode == 'zero_reward':
    return float(reward >= -1e-8), 1.0
  return float(reward > 0.0), 1.0

