"""Task-goal same-state counterfactual action-ranking utilities.

The Stage-2 outcome head was trained observationally: each replay state had
only the behavior action, so a head could predict trajectory difficulty from
``(state, goal)`` while ignoring action credit.  This module instead restores
the *same* MuJoCo state for several candidate actions, measures their outcomes
under a common continuation policy, and returns within-state ranking batches.

This file intentionally depends only on NumPy plus the already isolated
MuJoCo snapshot helpers.  It can therefore be unit-tested without JAX, Acme,
TensorFlow, Meta-World, or mujoco-py installed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from contrastive import action_ranking_diagnostics
from contrastive import counterfactual_outcomes


@dataclass(frozen=True)
class CounterfactualRankingBatch:
  """Candidate actions and task-goal outcomes grouped by anchor state."""

  observations: np.ndarray  # [anchors, candidates, observation_dim]
  actions: np.ndarray       # [anchors, candidates, action_dim]
  outcomes: np.ndarray      # progress + success_bonus * success
  progress: np.ndarray      # best H-step task-goal mechanism progress
  success: np.ndarray       # task-goal success observed in the rollout
  informative: np.ndarray   # at least one candidate pair differs sufficiently
  near_interaction: np.ndarray
  interaction_distance: np.ndarray
  proxy_success: Optional[np.ndarray] = None
  benchmark_success: Optional[np.ndarray] = None
  benchmark_success_available: Optional[np.ndarray] = None

  @property
  def num_anchors(self) -> int:
    return int(self.observations.shape[0])

  @property
  def num_candidates(self) -> int:
    return int(self.observations.shape[1])


class CounterfactualRankingBuffer:
  """Task-local anchor buffer; each item retains all same-state candidates."""

  def __init__(self, capacity: int):
    if capacity <= 0:
      raise ValueError('counterfactual ranking capacity must be positive')
    self._items = deque(maxlen=int(capacity))

  def __len__(self) -> int:
    return len(self._items)

  def add(self, batch: CounterfactualRankingBatch) -> None:
    for index in range(batch.num_anchors):
      self._items.append((
          batch.observations[index].copy(),
          batch.actions[index].copy(),
          batch.outcomes[index].copy(),
          batch.progress[index].copy(),
          batch.success[index].copy(),
          bool(batch.informative[index]),
          bool(batch.near_interaction[index]),
          float(batch.interaction_distance[index]),
      ))

  def sample(self, batch_size: int,
             rng: np.random.Generator) -> CounterfactualRankingBatch:
    if not self._items:
      raise ValueError('cannot sample an empty counterfactual ranking buffer')
    if batch_size <= 0:
      raise ValueError('counterfactual ranking batch size must be positive')
    # Replacement keeps the JAX shape fixed from the first event onward.
    indices = rng.integers(0, len(self._items), size=int(batch_size))
    selected = [self._items[int(index)] for index in indices]
    return CounterfactualRankingBatch(
        observations=np.stack([item[0] for item in selected]),
        actions=np.stack([item[1] for item in selected]),
        outcomes=np.stack([item[2] for item in selected]),
        progress=np.stack([item[3] for item in selected]),
        success=np.stack([item[4] for item in selected]),
        informative=np.asarray([item[5] for item in selected], dtype=bool),
        near_interaction=np.asarray([item[6] for item in selected], dtype=bool),
        interaction_distance=np.asarray(
            [item[7] for item in selected], dtype=np.float32),
    )


def _goal_distances(observation: np.ndarray,
                    obs_dim: int) -> tuple[float, float]:
  return counterfactual_outcomes.goal_distances(observation, obs_dim)


def _interaction_distance(observation: np.ndarray, obs_dim: int) -> float:
  return counterfactual_outcomes.interaction_distance(observation, obs_dim)


def scripted_contact_action(
    observation: np.ndarray,
    *,
    obs_dim: int,
    action_min: np.ndarray,
    action_max: np.ndarray,
    gain: float = 5.0,
) -> np.ndarray:
  """Move the Sawyer hand toward the mechanism; this supplies anchors only."""
  state = np.asarray(observation, dtype=np.float32)[:obs_dim]
  action_min = np.asarray(action_min, dtype=np.float32)
  action_max = np.asarray(action_max, dtype=np.float32)
  action = np.zeros_like(action_min, dtype=np.float32)
  if state.shape[0] < 7 or action.shape[0] < 3:
    raise ValueError(
        'scripted_contact requires Sawyer hand xyz and mechanism xyz')
  action[:3] = float(gain) * (state[4:7] - state[:3])
  # Neutral gripper avoids injecting task-specific open/close knowledge.
  if action.shape[0] > 3:
    action[3] = 0.0
  return np.clip(action, action_min, action_max)


def _candidate_actions(
    *,
    observation: np.ndarray,
    obs_dim: int,
    action_min: np.ndarray,
    action_max: np.ndarray,
    replay_observations: np.ndarray,
    replay_actions: np.ndarray,
    policy_action_fn: Callable[[np.ndarray, np.random.Generator, bool],
                               np.ndarray],
    candidates_per_family: int,
    local_noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
  k = int(candidates_per_family)
  if k < 2:
    raise ValueError('candidates_per_family must be at least 2')
  deterministic = np.asarray(
      policy_action_fn(observation, rng, False), dtype=np.float32)
  policy = [deterministic]
  for _ in range(k - 1):
    policy.append(np.asarray(
        policy_action_fn(observation, rng, True), dtype=np.float32))
  policy = np.stack(policy)
  local = np.clip(
      deterministic[None, :] + rng.normal(
          0.0, local_noise_std, size=(k, deterministic.shape[0])),
      action_min, action_max)
  uniform = rng.uniform(
      action_min, action_max, size=(k, deterministic.shape[0]))

  replay_observations = np.asarray(replay_observations)
  replay_actions = np.asarray(replay_actions)
  if replay_actions.shape[0] < k:
    raise ValueError(f'need at least {k} replay actions')
  replay_state = replay_observations[:, :obs_dim]
  anchor_state = np.asarray(observation[:obs_dim])
  nearest = np.argsort(
      np.linalg.norm(replay_state - anchor_state[None, :], axis=-1))[:k]
  replay = replay_actions[nearest]

  actions = np.concatenate([policy, local, replay, uniform], axis=0)
  families = np.asarray(
      ['policy'] * k + ['local'] * k + ['replay'] * k + ['uniform'] * k)
  return actions.astype(np.float32), families


def collect_counterfactual_ranking_batch(
    *,
    environment: Any,
    obs_dim: int,
    replay_observations: np.ndarray,
    replay_actions: np.ndarray,
    policy_action_fn: Callable[[np.ndarray, np.random.Generator, bool],
                               np.ndarray],
    rng: np.random.Generator,
    num_anchors: int = 4,
    candidates_per_family: int = 4,
    rollout_horizon: int = 100,
    action_repeat: int = 5,
    local_noise_std: float = 0.10,
    anchor_mode: str = 'scripted_contact',
    anchor_search_steps: int = 150,
    interaction_threshold: float = 0.09,
    contact_gain: float = 5.0,
    success_threshold: float = 0.05,
    success_mode: str = 'goal_distance',
    success_bonus: float = 1.0,
    min_outcome_gap: float = 0.002,
) -> tuple[CounterfactualRankingBatch, Dict[str, float]]:
  """Collect task-goal candidate outcomes from identical restored states."""
  if anchor_mode not in ('policy', 'scripted_contact'):
    raise ValueError('anchor_mode must be policy or scripted_contact')
  if rollout_horizon <= 0 or action_repeat <= 0:
    raise ValueError('rollout_horizon and action_repeat must be positive')
  action_spec = environment.action_spec()
  action_min = np.broadcast_to(
      np.asarray(action_spec.minimum, dtype=np.float32), action_spec.shape)
  action_max = np.broadcast_to(
      np.asarray(action_spec.maximum, dtype=np.float32), action_spec.shape)

  anchor_observations = []
  all_actions = []
  all_progress = []
  all_success = []
  all_proxy_success = []
  all_benchmark_success = []
  all_benchmark_available = []
  all_outcomes = []
  all_informative = []
  all_near = []
  all_distance = []
  all_families = []

  for _ in range(int(num_anchors)):
    timestep = environment.reset()
    best_observation = np.asarray(timestep.observation).copy()
    best_snapshot = action_ranking_diagnostics.snapshot_environment(environment)
    best_distance = _interaction_distance(best_observation, obs_dim)
    for _ in range(int(anchor_search_steps)):
      observation = np.asarray(timestep.observation)
      if anchor_mode == 'scripted_contact':
        action = scripted_contact_action(
            observation, obs_dim=obs_dim, action_min=action_min,
            action_max=action_max, gain=contact_gain)
      else:
        action = policy_action_fn(observation, rng, True)
      timestep = environment.step(np.clip(action, action_min, action_max))
      observation = np.asarray(timestep.observation).copy()
      distance = _interaction_distance(observation, obs_dim)
      is_last = bool(hasattr(timestep, 'last') and timestep.last())
      if not is_last and np.isfinite(distance) and (
          not np.isfinite(best_distance) or distance < best_distance):
        best_distance = distance
        best_observation = observation
        best_snapshot = action_ranking_diagnostics.snapshot_environment(
            environment)
      if np.isfinite(best_distance) and best_distance <= interaction_threshold:
        break
      if is_last:
        break

    action_ranking_diagnostics.restore_environment(best_snapshot)
    anchor = best_observation
    _, initial_mechanism = _goal_distances(anchor, obs_dim)
    actions, families = _candidate_actions(
        observation=anchor,
        obs_dim=obs_dim,
        action_min=action_min,
        action_max=action_max,
        replay_observations=replay_observations,
        replay_actions=replay_actions,
        policy_action_fn=policy_action_fn,
        candidates_per_family=candidates_per_family,
        local_noise_std=local_noise_std,
        rng=rng)

    progress_values = []
    success_values = []
    proxy_success_values = []
    benchmark_success_values = []
    benchmark_available_values = []
    continuation_seed = int(rng.integers(0, 2**31 - 1))
    for candidate in actions:
      action_ranking_diagnostics.restore_environment(best_snapshot)
      candidate_rng = np.random.default_rng(continuation_seed)
      candidate_timestep = None
      best_mechanism = initial_mechanism
      success = 0.0
      proxy_success = 0.0
      benchmark = 0.0
      benchmark_available = 0.0
      steps = 0
      for _ in range(min(int(action_repeat), int(rollout_horizon))):
        candidate_timestep = environment.step(candidate)
        steps += 1
        _, mechanism = _goal_distances(
            np.asarray(candidate_timestep.observation), obs_dim)
        best_mechanism = min(best_mechanism, mechanism)
        candidate_observation = np.asarray(candidate_timestep.observation)
        proxy_success = max(
            proxy_success,
            counterfactual_outcomes.mechanism_proxy_success(
                candidate_observation, obs_dim, success_threshold))
        current_benchmark, current_available = (
            counterfactual_outcomes.benchmark_success(
                candidate_timestep, candidate_observation, obs_dim,
                success_threshold, success_mode))
        benchmark = max(benchmark, current_benchmark)
        benchmark_available = max(benchmark_available, current_available)
        success = benchmark if success_mode != 'goal_distance' else proxy_success
        if hasattr(candidate_timestep, 'last') and candidate_timestep.last():
          break
      while steps < int(rollout_horizon):
        if (candidate_timestep is not None
            and hasattr(candidate_timestep, 'last')
            and candidate_timestep.last()):
          break
        observation = np.asarray(candidate_timestep.observation)
        continuation = policy_action_fn(
            observation, candidate_rng, False)
        candidate_timestep = environment.step(
            np.clip(continuation, action_min, action_max))
        steps += 1
        _, mechanism = _goal_distances(
            np.asarray(candidate_timestep.observation), obs_dim)
        best_mechanism = min(best_mechanism, mechanism)
        candidate_observation = np.asarray(candidate_timestep.observation)
        proxy_success = max(
            proxy_success,
            counterfactual_outcomes.mechanism_proxy_success(
                candidate_observation, obs_dim, success_threshold))
        current_benchmark, current_available = (
            counterfactual_outcomes.benchmark_success(
                candidate_timestep, candidate_observation, obs_dim,
                success_threshold, success_mode))
        benchmark = max(benchmark, current_benchmark)
        benchmark_available = max(benchmark_available, current_available)
        success = benchmark if success_mode != 'goal_distance' else proxy_success
      progress_values.append(max(0.0, initial_mechanism - best_mechanism))
      success_values.append(success)
      proxy_success_values.append(proxy_success)
      benchmark_success_values.append(benchmark)
      benchmark_available_values.append(benchmark_available)

    action_ranking_diagnostics.restore_environment(best_snapshot)
    progress = np.asarray(progress_values, dtype=np.float32)
    success = np.asarray(success_values, dtype=np.float32)
    proxy_success = np.asarray(proxy_success_values, dtype=np.float32)
    benchmark_success = np.asarray(
        benchmark_success_values, dtype=np.float32)
    benchmark_available = np.asarray(
        benchmark_available_values, dtype=np.float32)
    outcome = progress + float(success_bonus) * success
    informative = bool(
        float(np.max(outcome) - np.min(outcome)) >= min_outcome_gap)
    anchor_observations.append(np.repeat(
        anchor[None, :], actions.shape[0], axis=0))
    all_actions.append(actions)
    all_progress.append(progress)
    all_success.append(success)
    all_proxy_success.append(proxy_success)
    all_benchmark_success.append(benchmark_success)
    all_benchmark_available.append(benchmark_available)
    all_outcomes.append(outcome)
    all_informative.append(informative)
    all_near.append(
        bool(np.isfinite(best_distance)
             and best_distance <= interaction_threshold))
    all_distance.append(best_distance)
    all_families.append(families)

  batch = CounterfactualRankingBatch(
      observations=np.stack(anchor_observations).astype(np.float32),
      actions=np.stack(all_actions).astype(np.float32),
      outcomes=np.stack(all_outcomes).astype(np.float32),
      progress=np.stack(all_progress).astype(np.float32),
      success=np.stack(all_success).astype(np.float32),
      informative=np.asarray(all_informative, dtype=bool),
      near_interaction=np.asarray(all_near, dtype=bool),
      interaction_distance=np.asarray(all_distance, dtype=np.float32),
      proxy_success=np.stack(all_proxy_success).astype(np.float32),
      benchmark_success=np.stack(all_benchmark_success).astype(np.float32),
      benchmark_success_available=np.stack(
          all_benchmark_available).astype(np.float32),
  )
  proxy_success = batch.proxy_success
  benchmark_success = batch.benchmark_success
  benchmark_available = batch.benchmark_success_available
  available_mask = benchmark_available > 0.5
  if np.any(available_mask):
    agreement = float(np.mean(
        proxy_success[available_mask] == benchmark_success[available_mask]))
    proxy_false_positive = float(np.mean(
        (proxy_success[available_mask] > 0.5)
        & (benchmark_success[available_mask] <= 0.5)))
    proxy_false_negative = float(np.mean(
        (proxy_success[available_mask] <= 0.5)
        & (benchmark_success[available_mask] > 0.5)))
  else:
    agreement = 0.0
    proxy_false_positive = 0.0
    proxy_false_negative = 0.0
  metrics = {
      'counterfactual_rank/informative_anchor_fraction': float(
          np.mean(batch.informative)),
      'counterfactual_rank/near_interaction_fraction': float(
          np.mean(batch.near_interaction)),
      'counterfactual_rank/anchor_interaction_distance': float(
          np.nanmean(batch.interaction_distance)),
      'counterfactual_rank/candidate_progress_std': float(
          np.mean(np.std(batch.progress, axis=1))),
      'counterfactual_rank/nonzero_progress_fraction': float(
          np.mean(batch.progress > min_outcome_gap)),
      'counterfactual_rank/task_success_fraction': float(
          np.mean(batch.success)),
      'counterfactual_rank/proxy_success_fraction': float(
          np.mean(proxy_success)),
      'counterfactual_rank/benchmark_success_fraction': float(
          np.mean(benchmark_success)),
      'counterfactual_rank/benchmark_success_available_fraction': float(
          np.mean(benchmark_available)),
      'counterfactual_rank/success_predicate_agreement': agreement,
      'counterfactual_rank/proxy_false_positive_fraction':
          proxy_false_positive,
      'counterfactual_rank/proxy_false_negative_fraction':
          proxy_false_negative,
      'counterfactual_rank/task_success_variation_fraction': float(np.mean(
          np.max(batch.success, axis=1) > np.min(batch.success, axis=1))),
      'counterfactual_rank/buffer_candidate_count': float(
          batch.num_anchors * batch.num_candidates),
  }
  family_array = np.stack(all_families)
  for family_name in ('policy', 'local', 'replay', 'uniform'):
    mask = family_array == family_name
    metrics[f'counterfactual_rank/{family_name}_progress_mean'] = float(
        np.mean(batch.progress[mask]))
    metrics[f'counterfactual_rank/{family_name}_success_mean'] = float(
        np.mean(batch.success[mask]))
    metrics[f'counterfactual_rank/{family_name}_outcome_mean'] = float(
        np.mean(batch.outcomes[mask]))
  best_family = np.take_along_axis(
      family_array, np.argmax(batch.outcomes, axis=1)[:, None], axis=1)[:, 0]
  metrics['counterfactual_rank/policy_is_best_fraction'] = float(
      np.mean(best_family == 'policy'))
  return batch, metrics


def _pairwise_accuracy(scores: np.ndarray, outcomes: np.ndarray,
                       min_gap: float) -> float:
  correct = []
  for anchor_scores, anchor_outcomes in zip(scores, outcomes):
    for left in range(anchor_scores.shape[0]):
      for right in range(left + 1, anchor_scores.shape[0]):
        delta = float(anchor_outcomes[left] - anchor_outcomes[right])
        if abs(delta) < min_gap:
          continue
        predicted = float(anchor_scores[left] - anchor_scores[right])
        correct.append(float(predicted * delta > 0.0))
  return float(np.mean(correct)) if correct else 0.0


def summarize_counterfactual_scores(
    scores: np.ndarray,
    batch: CounterfactualRankingBatch,
    *,
    min_outcome_gap: float,
) -> Dict[str, float]:
  """Metrics used for promotion gates; constants produce zero correlation."""
  scores = np.asarray(scores, dtype=np.float64)
  correlations = []
  regrets = []
  for anchor_scores, outcomes, informative in zip(
      scores, batch.outcomes, batch.informative):
    if not informative:
      continue
    correlations.append(action_ranking_diagnostics.spearman(
        anchor_scores, outcomes))
    regrets.append(float(
        np.max(outcomes) - outcomes[int(np.argmax(anchor_scores))]))
  permutation_drops = []
  for anchor_scores, outcomes, informative in zip(
      scores, batch.outcomes, batch.informative):
    if not informative or anchor_scores.shape[0] < 2:
      continue
    original = action_ranking_diagnostics.spearman(anchor_scores, outcomes)
    permuted = [
        action_ranking_diagnostics.spearman(
            np.roll(anchor_scores, shift), outcomes)
        for shift in range(1, anchor_scores.shape[0])]
    permutation_drops.append(original - float(np.mean(permuted)))
  rank_spearman = float(np.mean(correlations)) if correlations else 0.0
  return {
      # Keep the historical key as an alias for old dashboards.  The labels
      # are progress + success bonus, so ``score_vs_outcome_spearman`` is the
      # accurate name used by all new promotion gates.
      'counterfactual_rank/score_vs_task_progress_spearman': rank_spearman,
      'counterfactual_rank/score_vs_outcome_spearman': rank_spearman,
      'counterfactual_rank/pairwise_accuracy': _pairwise_accuracy(
          scores, batch.outcomes, min_outcome_gap),
      'counterfactual_rank/top_action_regret': (
          float(np.mean(regrets)) if regrets else 0.0),
      'counterfactual_rank/fixed_state_score_std': float(
          np.mean(np.std(scores, axis=1))),
      'counterfactual_rank/action_permutation_spearman_drop': (
          float(np.mean(permutation_drops)) if permutation_drops else 0.0),
  }


def summarize_oracle(batch: CounterfactualRankingBatch) -> Dict[str, float]:
  """Summarize the best outcome available in the tested candidate class."""
  oracle_success = np.max(batch.success, axis=1)
  random_success = np.mean(batch.success, axis=1)
  oracle_outcome = np.max(batch.outcomes, axis=1)
  random_outcome = np.mean(batch.outcomes, axis=1)
  return {
      'oracle/best_success_fraction': float(np.mean(oracle_success)),
      'oracle/random_success_fraction': float(np.mean(random_success)),
      'oracle/success_gain': float(np.mean(
          oracle_success - random_success)),
      'oracle/best_outcome_mean': float(np.mean(oracle_outcome)),
      'oracle/random_outcome_mean': float(np.mean(random_outcome)),
      'oracle/outcome_gain': float(np.mean(oracle_outcome - random_outcome)),
      'oracle/informative_anchor_fraction': float(np.mean(batch.informative)),
      'oracle/near_interaction_fraction': float(
          np.mean(batch.near_interaction)),
  }


def _self_test() -> None:
  rng = np.random.default_rng(7)
  observations = np.zeros((2, 4, 14), dtype=np.float32)
  actions = rng.normal(size=(2, 4, 4)).astype(np.float32)
  outcomes = np.asarray([[0.0, 0.1, 0.2, 0.3],
                         [0.3, 0.2, 0.1, 0.0]], dtype=np.float32)
  batch = CounterfactualRankingBatch(
      observations=observations,
      actions=actions,
      outcomes=outcomes,
      progress=outcomes,
      success=np.zeros_like(outcomes),
      informative=np.ones((2,), dtype=bool),
      near_interaction=np.ones((2,), dtype=bool),
      interaction_distance=np.zeros((2,), dtype=np.float32),
  )
  metrics = summarize_counterfactual_scores(
      outcomes, batch, min_outcome_gap=0.01)
  if metrics['counterfactual_rank/score_vs_task_progress_spearman'] < 0.999:
    raise AssertionError('counterfactual ranking self-test failed')
  buffer = CounterfactualRankingBuffer(capacity=3)
  buffer.add(batch)
  sampled = buffer.sample(4, rng)
  if sampled.observations.shape != (4, 4, 14):
    raise AssertionError('counterfactual buffer self-test failed')
  print('counterfactual-ranking dependency-light self-test passed')


if __name__ == '__main__':
  _self_test()
