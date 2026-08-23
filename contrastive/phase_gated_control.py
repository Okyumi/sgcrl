"""Phase-gated repeated-action controller for counterfactual rankers.

The wrapper preserves Acme's actor interface.  It always advances the wrapped
actor, but it may override the selected action with a scripted reach action or
with the highest-scoring supported candidate at contact.  The chosen contact
action is then executed for exactly the chunk length used by the labeler.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from contrastive import counterfactual_outcomes
from contrastive.counterfactual_ranking import scripted_contact_action


class PhaseGatedChunkActor:
  """Acme-compatible actor wrapper with aligned chunk execution."""

  def __init__(
      self,
      actor,
      *,
      obs_dim: int,
      action_spec,
      score_actions_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
      rng: np.random.Generator,
      reach_mode: str = 'policy',
      interaction_threshold: float = 0.09,
      chunk_length: int = 5,
      num_candidates: int = 16,
      local_noise_std: float = 0.10,
      contact_gain: float = 5.0,
  ):
    if reach_mode not in ('policy', 'scripted_contact'):
      raise ValueError('reach_mode must be policy or scripted_contact')
    if chunk_length <= 0:
      raise ValueError('chunk_length must be positive')
    if num_candidates < 2:
      raise ValueError('num_candidates must be at least two')
    self._actor = actor
    self._obs_dim = int(obs_dim)
    self._score_actions_fn = score_actions_fn
    self._rng = rng
    self._reach_mode = reach_mode
    self._interaction_threshold = float(interaction_threshold)
    self._chunk_length = int(chunk_length)
    self._num_candidates = int(num_candidates)
    self._local_noise_std = float(local_noise_std)
    self._contact_gain = float(contact_gain)
    self._action_min = np.broadcast_to(
        np.asarray(action_spec.minimum, dtype=np.float32), action_spec.shape)
    self._action_max = np.broadcast_to(
        np.asarray(action_spec.maximum, dtype=np.float32), action_spec.shape)
    self._cached_action = None
    self._remaining = 0
    self._steps = 0
    self._contact_steps = 0
    self._episodes = 0
    self._contact_episodes = 0
    self._episode_steps = 0
    self._episode_contact = False
    self._first_contact_steps = []
    self._scripted_steps = 0
    self._chunk_selections = 0
    self._selected_scores = []
    self._candidate_score_stds = []

  def observe_first(self, timestep):
    self._cached_action = None
    self._remaining = 0
    self._episodes += 1
    self._episode_steps = 0
    self._episode_contact = False
    return self._actor.observe_first(timestep)

  def _candidates(self, base_action: np.ndarray) -> np.ndarray:
    remaining = self._num_candidates - 1
    local_count = (remaining + 1) // 2
    uniform_count = remaining - local_count
    local = np.clip(
        base_action[None, :] + self._rng.normal(
            0.0, self._local_noise_std,
            size=(local_count, base_action.shape[0])),
        self._action_min, self._action_max)
    pieces = [base_action[None, :], local]
    if uniform_count:
      pieces.append(self._rng.uniform(
          self._action_min, self._action_max,
          size=(uniform_count, base_action.shape[0])))
    return np.concatenate(pieces, axis=0).astype(np.float32)

  def select_action(self, observation):
    observation = np.asarray(observation)
    # Advance the wrapped feed-forward actor and its RNG on every environment
    # step, including overridden/repeated steps.
    base_action = np.asarray(
        self._actor.select_action(observation), dtype=np.float32)
    self._steps += 1
    self._episode_steps += 1
    distance = counterfactual_outcomes.interaction_distance(
        observation, self._obs_dim)
    in_contact = bool(
        np.isfinite(distance) and distance <= self._interaction_threshold)
    if not in_contact:
      self._cached_action = None
      self._remaining = 0
      if self._reach_mode == 'scripted_contact':
        self._scripted_steps += 1
        return scripted_contact_action(
            observation, obs_dim=self._obs_dim,
            action_min=self._action_min, action_max=self._action_max,
            gain=self._contact_gain)
      return np.clip(base_action, self._action_min, self._action_max)

    self._contact_steps += 1
    if not self._episode_contact:
      self._episode_contact = True
      self._contact_episodes += 1
      self._first_contact_steps.append(float(self._episode_steps))
    if self._remaining > 0 and self._cached_action is not None:
      self._remaining -= 1
      return self._cached_action.copy()

    candidates = self._candidates(base_action)
    scores = np.asarray(
        self._score_actions_fn(observation, candidates), dtype=np.float64)
    if scores.shape != (candidates.shape[0],):
      raise ValueError(
          f'Expected {candidates.shape[0]} candidate scores, got {scores.shape}')
    selected = int(np.argmax(scores)) if np.all(np.isfinite(scores)) else 0
    self._cached_action = candidates[selected].copy()
    self._remaining = self._chunk_length - 1
    self._chunk_selections += 1
    self._selected_scores.append(float(scores[selected]))
    self._candidate_score_stds.append(float(np.std(scores)))
    return self._cached_action.copy()

  def observe(self, action, next_timestep):
    return self._actor.observe(action, next_timestep)

  def update(self, *args, **kwargs):
    return self._actor.update(*args, **kwargs)

  def get_and_reset_metrics(self) -> Dict[str, float]:
    metrics = {
        'phase_control/contact_step_fraction': (
            self._contact_steps / max(self._steps, 1)),
        'phase_control/contact_episode_reach_rate': (
            self._contact_episodes / max(self._episodes, 1)),
        'phase_control/first_contact_step_mean': (
            float(np.mean(self._first_contact_steps))
            if self._first_contact_steps else 0.0),
        'phase_control/scripted_reach_step_fraction': (
            self._scripted_steps / max(self._steps, 1)),
        'phase_control/chunk_selections': float(self._chunk_selections),
        'phase_control/selected_score_mean': (
            float(np.mean(self._selected_scores))
            if self._selected_scores else 0.0),
        'phase_control/candidate_score_std': (
            float(np.mean(self._candidate_score_stds))
            if self._candidate_score_stds else 0.0),
    }
    self._steps = 0
    self._contact_steps = 0
    self._episodes = 0
    self._contact_episodes = 0
    self._first_contact_steps = []
    self._scripted_steps = 0
    self._chunk_selections = 0
    self._selected_scores = []
    self._candidate_score_stds = []
    return metrics
