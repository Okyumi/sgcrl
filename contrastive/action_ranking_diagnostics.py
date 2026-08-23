"""Causal diagnostics for DCC action ranking and actor exploitation.

The learner-side shortcut diagnostics compare actions across different replay
states.  That is useful for measuring action sensitivity, but it cannot answer
the control question: for one fixed simulator state and goal, does the DCC
score order alternative actions by their observed consequences?

This module answers that question outside the learner hot path.  It snapshots
one Meta-World/MuJoCo state, restores that exact state for every candidate
action, and measures one-step and multi-step goal progress under a shared
continuation policy.  Candidate actions come from four families: current
policy, local perturbations, replay-neighbour actions, and uniform actions.

No training objective is changed.  With the runner flag disabled (the
default), this module is never invoked.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from contrastive import counterfactual_outcomes


_WRAPPER_LINKS = ('_environment', 'environment', '_env', 'env', '_gym_env')
_STATEFUL_ATTRIBUTES = (
    '_elapsed_steps',
    '_episode_steps',
    '_step_count',
    'curr_path_length',
    '_curr_path_length',
    'path_length',
    '_path_length',
    'timestep',
    '_timestep',
    '_reset_next_step',
    '_needs_reset',
    '_done',
    '_last_observation',
    '_last_obs',
    '_prev_obs',
    '_observation',
)
_RNG_ATTRIBUTES = ('np_random', '_np_random', 'rng', '_rng')


@dataclasses.dataclass
class _ObjectState:
  obj: Any
  attributes: Dict[str, Any]
  rng_states: Dict[str, tuple[str, Any]]


@dataclasses.dataclass
class EnvironmentSnapshot:
  """Recoverable state for an Acme-wrapped mujoco-py environment."""

  sim: Any
  sim_state: Any
  mocap_pos: np.ndarray | None
  mocap_quat: np.ndarray | None
  ctrl: np.ndarray | None
  qfrc_applied: np.ndarray | None
  xfrc_applied: np.ndarray | None
  objects: List[_ObjectState]
  numpy_global_state: Any


def iter_environment_chain(environment: Any) -> Iterable[Any]:
  """Yield wrapper objects down to the underlying Gym/Meta-World env."""
  current = environment
  seen = set()
  while current is not None and id(current) not in seen:
    seen.add(id(current))
    yield current
    child = None
    for name in _WRAPPER_LINKS:
      try:
        candidate = getattr(current, name)
      except (AttributeError, TypeError):
        continue
      if candidate is not None and candidate is not current:
        child = candidate
        break
    current = child


def _copy_array_attribute(obj: Any, name: str) -> np.ndarray | None:
  try:
    value = getattr(obj, name)
  except (AttributeError, TypeError):
    return None
  try:
    return np.asarray(value).copy()
  except (TypeError, ValueError):
    return None


def _capture_rng(rng: Any) -> tuple[str, Any] | None:
  if hasattr(rng, 'get_state') and hasattr(rng, 'set_state'):
    return ('random_state', copy.deepcopy(rng.get_state()))
  bit_generator = getattr(rng, 'bit_generator', None)
  if bit_generator is not None and hasattr(bit_generator, 'state'):
    return ('generator', copy.deepcopy(bit_generator.state))
  return None


def _restore_rng(rng: Any, state: tuple[str, Any]) -> None:
  kind, payload = state
  if kind == 'random_state':
    rng.set_state(copy.deepcopy(payload))
  elif kind == 'generator':
    rng.bit_generator.state = copy.deepcopy(payload)
  else:
    raise ValueError(f'Unknown RNG state kind: {kind!r}')


def snapshot_environment(environment: Any) -> EnvironmentSnapshot:
  """Capture MuJoCo, wrapper, controller, cached-observation and RNG state."""
  chain = list(iter_environment_chain(environment))
  sim = next((getattr(obj, 'sim', None) for obj in chain
              if getattr(obj, 'sim', None) is not None), None)
  if sim is None or not hasattr(sim, 'get_state'):
    raise RuntimeError(
        'Action-landscape diagnostics require a mujoco-py environment with '
        'sim.get_state()/sim.set_state().')

  object_states = []
  for obj in chain:
    attributes = {}
    for name in _STATEFUL_ATTRIBUTES:
      if hasattr(obj, name):
        try:
          attributes[name] = copy.deepcopy(getattr(obj, name))
        except (TypeError, ValueError):
          pass
    rng_states = {}
    for name in _RNG_ATTRIBUTES:
      if not hasattr(obj, name):
        continue
      captured = _capture_rng(getattr(obj, name))
      if captured is not None:
        rng_states[name] = captured
    action_space = getattr(obj, 'action_space', None)
    if action_space is not None and hasattr(action_space, 'np_random'):
      captured = _capture_rng(action_space.np_random)
      if captured is not None:
        rng_states['action_space.np_random'] = captured
    object_states.append(_ObjectState(obj, attributes, rng_states))

  data = sim.data
  return EnvironmentSnapshot(
      sim=sim,
      sim_state=copy.deepcopy(sim.get_state()),
      mocap_pos=_copy_array_attribute(data, 'mocap_pos'),
      mocap_quat=_copy_array_attribute(data, 'mocap_quat'),
      ctrl=_copy_array_attribute(data, 'ctrl'),
      qfrc_applied=_copy_array_attribute(data, 'qfrc_applied'),
      xfrc_applied=_copy_array_attribute(data, 'xfrc_applied'),
      objects=object_states,
      numpy_global_state=copy.deepcopy(np.random.get_state()),
  )


def _restore_array_attribute(obj: Any, name: str,
                             value: np.ndarray | None) -> None:
  if value is None or not hasattr(obj, name):
    return
  target = getattr(obj, name)
  target[...] = value


def restore_environment(snapshot: EnvironmentSnapshot) -> None:
  """Restore an :class:`EnvironmentSnapshot` in place."""
  sim = snapshot.sim
  sim.set_state(copy.deepcopy(snapshot.sim_state))
  _restore_array_attribute(sim.data, 'mocap_pos', snapshot.mocap_pos)
  _restore_array_attribute(sim.data, 'mocap_quat', snapshot.mocap_quat)
  _restore_array_attribute(sim.data, 'ctrl', snapshot.ctrl)
  _restore_array_attribute(
      sim.data, 'qfrc_applied', snapshot.qfrc_applied)
  _restore_array_attribute(sim.data, 'xfrc_applied', snapshot.xfrc_applied)

  for object_state in snapshot.objects:
    for name, value in object_state.attributes.items():
      try:
        setattr(object_state.obj, name, copy.deepcopy(value))
      except (AttributeError, TypeError):
        pass
    for name, rng_state in object_state.rng_states.items():
      if name == 'action_space.np_random':
        rng = object_state.obj.action_space.np_random
      else:
        rng = getattr(object_state.obj, name)
      _restore_rng(rng, rng_state)

  np.random.set_state(copy.deepcopy(snapshot.numpy_global_state))
  if hasattr(sim, 'forward'):
    sim.forward()


def _timestep_payload(timestep: Any) -> tuple[np.ndarray, float, float, Any]:
  reward = 0.0 if timestep.reward is None else float(timestep.reward)
  discount = 1.0 if timestep.discount is None else float(timestep.discount)
  return (
      np.asarray(timestep.observation).copy(), reward, discount,
      getattr(timestep, 'step_type', None))


def assert_restore_step_reproducible(
    environment: Any,
    action: np.ndarray,
    *,
    atol: float = 1e-7,
) -> None:
  """Abort unless restore--step reproduces the same transition exactly."""
  environment.reset()
  snapshot = snapshot_environment(environment)
  first = _timestep_payload(environment.step(np.asarray(action)))
  restore_environment(snapshot)
  second = _timestep_payload(environment.step(np.asarray(action)))
  np.testing.assert_allclose(first[0], second[0], atol=atol, rtol=0.0)
  np.testing.assert_allclose(first[1:3], second[1:3], atol=atol, rtol=0.0)
  if first[3] != second[3]:
    raise AssertionError(
        f'restore--step changed step_type: {first[3]!r} != {second[3]!r}')


def _rankdata(values: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64)
  order = np.argsort(values, kind='mergesort')
  ranks = np.empty(values.shape[0], dtype=np.float64)
  ranks[order] = np.arange(values.shape[0], dtype=np.float64)
  # Average exact ties; this matters for binary success outcomes.
  unique, inverse, counts = np.unique(values, return_inverse=True,
                                      return_counts=True)
  del unique
  for group in np.flatnonzero(counts > 1):
    members = inverse == group
    ranks[members] = np.mean(ranks[members])
  return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
  """Dependency-light Spearman correlation; returns 0 for constants."""
  a = _rankdata(np.asarray(values_a))
  b = _rankdata(np.asarray(values_b))
  a -= np.mean(a)
  b -= np.mean(b)
  denominator = np.linalg.norm(a) * np.linalg.norm(b)
  if denominator <= 1e-12:
    return 0.0
  return float(np.dot(a, b) / denominator)


def _goal_distances(observation: np.ndarray, obs_dim: int) -> tuple[float, float]:
  return counterfactual_outcomes.goal_distances(observation, obs_dim)


def _interaction_distance(observation: np.ndarray, obs_dim: int) -> float:
  """Hand-to-mechanism distance for unified Sawyer observations."""
  return counterfactual_outcomes.interaction_distance(observation, obs_dim)


def _percentile(value: float, reference: np.ndarray) -> float:
  reference = np.asarray(reference, dtype=np.float64)
  if not reference.size:
    return 0.5
  return float(np.mean(reference <= value))


def _family_metrics(
    scores: np.ndarray,
    outcomes: Mapping[str, np.ndarray],
    family: np.ndarray,
) -> Dict[str, float]:
  metrics: Dict[str, float] = {}
  for family_name in ('policy', 'local', 'replay', 'uniform'):
    mask = family == family_name
    if not np.any(mask):
      continue
    metrics[f'action_landscape/{family_name}_score_mean'] = float(
        np.mean(scores[mask]))
    metrics[
        f'action_landscape/{family_name}_rollout_mechanism_progress_mean'
    ] = float(np.mean(outcomes['rollout_mechanism_progress'][mask]))
    metrics[f'action_landscape/{family_name}_success_mean'] = float(
        np.mean(outcomes['success'][mask]))
  return metrics


def summarize_action_ranking(
    scores: Sequence[float],
    outcomes: Mapping[str, Sequence[float]],
    family: Sequence[str],
    actions: np.ndarray,
    replay_actions: np.ndarray,
) -> Dict[str, float]:
  """Summarize critic calibration and the actor-exploitation signature."""
  scores = np.asarray(scores, dtype=np.float64)
  family = np.asarray(family)
  outcomes_np = {
      name: np.asarray(values, dtype=np.float64)
      for name, values in outcomes.items()
  }
  metrics: Dict[str, float] = {}
  for outcome_name, values in outcomes_np.items():
    metrics[f'action_landscape/score_vs_{outcome_name}_spearman'] = (
        spearman(scores, values))
    best_outcome = float(np.max(values))
    score_choice_outcome = float(values[int(np.argmax(scores))])
    metrics[f'action_landscape/top_score_{outcome_name}_regret'] = (
        best_outcome - score_choice_outcome)
    metrics[f'action_landscape/{outcome_name}_std'] = float(np.std(values))

  replay_mask = family == 'replay'
  if np.sum(replay_mask) >= 2:
    replay_outcome = outcomes_np['rollout_mechanism_progress'][replay_mask]
    replay_score = scores[replay_mask]
    metrics[
        'action_landscape/replay_score_vs_rollout_mechanism_spearman'
    ] = spearman(replay_score, replay_outcome)
    metrics['action_landscape/replay_top_score_regret'] = float(
        np.max(replay_outcome) - replay_outcome[int(np.argmax(replay_score))])

  policy_mask = family == 'policy'
  comparison_mask = ~policy_mask
  if np.any(policy_mask):
    policy_scores = scores[policy_mask]
    policy_outcomes = outcomes_np['rollout_mechanism_progress'][policy_mask]
    score_percentiles = [
        _percentile(value, scores[comparison_mask]) for value in policy_scores]
    outcome_percentiles = [
        _percentile(value,
                    outcomes_np['rollout_mechanism_progress'][comparison_mask])
        for value in policy_outcomes]
    score_percentile = float(np.mean(score_percentiles))
    outcome_percentile = float(np.mean(outcome_percentiles))
    metrics['action_landscape/policy_score_percentile'] = score_percentile
    metrics['action_landscape/policy_outcome_percentile'] = outcome_percentile
    metrics['action_landscape/policy_score_outcome_percentile_gap'] = (
        score_percentile - outcome_percentile)

    if np.asarray(replay_actions).size:
      policy_actions = np.asarray(actions)[policy_mask]
      support_distances = np.linalg.norm(
          policy_actions[:, None, :] - np.asarray(replay_actions)[None, :, :],
          axis=-1)
      metrics['action_landscape/policy_replay_support_distance'] = float(
          np.mean(np.min(support_distances, axis=1)))

  metrics.update(_family_metrics(scores, outcomes_np, family))
  metrics['action_landscape/candidate_score_std'] = float(np.std(scores))
  metrics['action_landscape/num_candidates'] = float(scores.shape[0])
  return metrics


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
  """Build matched action families for one fixed state and goal."""
  k = int(candidates_per_family)
  if k < 2:
    raise ValueError('candidates_per_family must be at least 2.')
  deterministic_action = np.asarray(
      policy_action_fn(observation, rng, False), dtype=np.float32)
  policy_actions = [deterministic_action]
  for _ in range(k - 1):
    policy_actions.append(np.asarray(
        policy_action_fn(observation, rng, True), dtype=np.float32))
  policy_actions = np.stack(policy_actions)

  local_actions = deterministic_action[None, :] + rng.normal(
      0.0, local_noise_std, size=(k, deterministic_action.shape[0]))
  local_actions = np.clip(local_actions, action_min, action_max)
  uniform_actions = rng.uniform(action_min, action_max,
                                size=(k, deterministic_action.shape[0]))

  replay_observations = np.asarray(replay_observations)
  replay_actions = np.asarray(replay_actions)
  if replay_actions.shape[0] < k:
    raise ValueError(
        f'Need at least {k} replay actions, got {replay_actions.shape[0]}.')
  replay_state = replay_observations[:, :obs_dim]
  anchor_state = np.asarray(observation[:obs_dim])
  nearest = np.argsort(
      np.linalg.norm(replay_state - anchor_state[None, :], axis=-1))[:k]
  nearest_actions = replay_actions[nearest]

  actions = np.concatenate(
      [policy_actions, local_actions, nearest_actions, uniform_actions], axis=0)
  family = np.asarray(
      ['policy'] * k + ['local'] * k + ['replay'] * k + ['uniform'] * k)
  return actions.astype(np.float32), family


def run_causal_action_ranking_probe(
    *,
    environment: Any,
    obs_dim: int,
    replay_observations: np.ndarray,
    replay_actions: np.ndarray,
    policy_action_fn: Callable[[np.ndarray, np.random.Generator, bool],
                               np.ndarray],
    score_actions_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rng: np.random.Generator,
    num_anchors: int = 1,
    candidates_per_family: int = 4,
    rollout_horizon: int = 25,
    anchor_prefix_steps: int = 20,
    local_noise_std: float = 0.10,
    interaction_aware_anchor: bool = False,
    interaction_threshold: float = 0.09,
    anchor_search_steps: int = 200,
    action_repeat: int = 1,
    use_best_progress: bool = False,
    success_threshold: float = 0.05,
    success_mode: str = 'goal_distance',
) -> Dict[str, float]:
  """Run same-state action interventions and return W&B-ready metrics."""
  action_spec = environment.action_spec()
  action_min = np.broadcast_to(
      np.asarray(action_spec.minimum, dtype=np.float32), action_spec.shape)
  action_max = np.broadcast_to(
      np.asarray(action_spec.maximum, dtype=np.float32), action_spec.shape)
  per_anchor: List[Dict[str, float]] = []

  for anchor_id in range(int(num_anchors)):
    timestep = environment.reset()
    if interaction_aware_anchor:
      best_observation = np.asarray(timestep.observation).copy()
      best_snapshot = snapshot_environment(environment)
      best_distance = _interaction_distance(best_observation, obs_dim)
      for _ in range(int(anchor_search_steps)):
        action = policy_action_fn(
            np.asarray(timestep.observation), rng, True)
        timestep = environment.step(np.clip(action, action_min, action_max))
        observation = np.asarray(timestep.observation).copy()
        distance = _interaction_distance(observation, obs_dim)
        if np.isfinite(distance) and (
            not np.isfinite(best_distance) or distance < best_distance):
          best_distance = distance
          best_observation = observation
          best_snapshot = snapshot_environment(environment)
        if np.isfinite(best_distance) and best_distance <= interaction_threshold:
          break
        if hasattr(timestep, 'last') and timestep.last():
          break
      restore_environment(best_snapshot)
      anchor_observation = best_observation
      snapshot = best_snapshot
    else:
      prefix = int(anchor_prefix_steps) + 5 * anchor_id
      for _ in range(prefix):
        action = policy_action_fn(
            np.asarray(timestep.observation), rng, True)
        timestep = environment.step(np.clip(action, action_min, action_max))
        if hasattr(timestep, 'last') and timestep.last():
          timestep = environment.reset()
      anchor_observation = np.asarray(timestep.observation).copy()
      snapshot = snapshot_environment(environment)
      best_distance = _interaction_distance(anchor_observation, obs_dim)
    initial_full, initial_mechanism = _goal_distances(
        anchor_observation, obs_dim)
    actions, family = _candidate_actions(
        observation=anchor_observation,
        obs_dim=obs_dim,
        action_min=action_min,
        action_max=action_max,
        replay_observations=replay_observations,
        replay_actions=replay_actions,
        policy_action_fn=policy_action_fn,
        candidates_per_family=candidates_per_family,
        local_noise_std=local_noise_std,
        rng=rng,
    )
    scores = np.asarray(
        score_actions_fn(anchor_observation, actions), dtype=np.float64)
    outcomes = {
        'one_step_full_progress': [],
        'one_step_mechanism_progress': [],
        'rollout_full_progress': [],
        'rollout_mechanism_progress': [],
        'success': [],
        'best_rollout_full_progress': [],
        'best_rollout_mechanism_progress': [],
        'benchmark_success': [],
        'benchmark_success_available': [],
        'proxy_success': [],
    }

    # The continuation key stream is reset for every candidate.  Thus only
    # the intervened first action differs across counterfactual rollouts.
    continuation_seed = int(rng.integers(0, 2**31 - 1))
    for action in actions:
      restore_environment(snapshot)
      candidate_rng = np.random.default_rng(continuation_seed)
      candidate_timestep = None
      one_full = initial_full
      one_mechanism = initial_mechanism
      best_full = initial_full
      best_mechanism = initial_mechanism
      selected_success = 0.0
      benchmark_success = 0.0
      benchmark_available = 0.0
      proxy_success = 0.0
      steps = 0
      for repeat_index in range(min(
          max(1, int(action_repeat)), int(rollout_horizon))):
        candidate_timestep = environment.step(action)
        steps += 1
        observation = np.asarray(candidate_timestep.observation)
        current_full, current_mechanism = _goal_distances(
            observation, obs_dim)
        if repeat_index == 0:
          one_full = current_full
          one_mechanism = current_mechanism
        best_full = min(best_full, current_full)
        best_mechanism = min(best_mechanism, current_mechanism)
        proxy_success = max(
            proxy_success,
            counterfactual_outcomes.mechanism_proxy_success(
                observation, obs_dim, success_threshold))
        current_benchmark, current_available = (
            counterfactual_outcomes.benchmark_success(
                candidate_timestep, observation, obs_dim,
                success_threshold, success_mode))
        benchmark_success = max(benchmark_success, current_benchmark)
        benchmark_available = max(benchmark_available, current_available)
        selected_success = (
            benchmark_success if success_mode != 'goal_distance'
            else proxy_success)
        if (hasattr(candidate_timestep, 'last')
            and candidate_timestep.last()):
          break
      for _ in range(max(0, int(rollout_horizon) - steps)):
        if (hasattr(candidate_timestep, 'last')
            and candidate_timestep.last()):
          break
        continuation_action = policy_action_fn(
            np.asarray(candidate_timestep.observation),
            candidate_rng,
            False)
        candidate_timestep = environment.step(
            np.clip(continuation_action, action_min, action_max))
        observation = np.asarray(candidate_timestep.observation)
        current_full, current_mechanism = _goal_distances(
            observation, obs_dim)
        best_full = min(best_full, current_full)
        best_mechanism = min(best_mechanism, current_mechanism)
        proxy_success = max(
            proxy_success,
            counterfactual_outcomes.mechanism_proxy_success(
                observation, obs_dim, success_threshold))
        current_benchmark, current_available = (
            counterfactual_outcomes.benchmark_success(
                candidate_timestep, observation, obs_dim,
                success_threshold, success_mode))
        benchmark_success = max(benchmark_success, current_benchmark)
        benchmark_available = max(benchmark_available, current_available)
        selected_success = (
            benchmark_success if success_mode != 'goal_distance'
            else proxy_success)
      final_full, final_mechanism = _goal_distances(
          np.asarray(candidate_timestep.observation), obs_dim)
      outcomes['one_step_full_progress'].append(initial_full - one_full)
      outcomes['one_step_mechanism_progress'].append(
          initial_mechanism - one_mechanism)
      outcomes['rollout_full_progress'].append(initial_full - final_full)
      outcomes['rollout_mechanism_progress'].append(
          initial_mechanism - final_mechanism)
      outcomes['best_rollout_full_progress'].append(
          initial_full - best_full)
      outcomes['best_rollout_mechanism_progress'].append(
          initial_mechanism - best_mechanism)
      outcomes['success'].append(selected_success)
      outcomes['benchmark_success'].append(benchmark_success)
      outcomes['benchmark_success_available'].append(benchmark_available)
      outcomes['proxy_success'].append(proxy_success)

    restore_environment(snapshot)
    anchor_metrics = summarize_action_ranking(
        scores=scores,
        outcomes=outcomes,
        family=family,
        actions=actions,
        replay_actions=replay_actions,
    )
    anchor_metrics['action_landscape/anchor_interaction_distance'] = float(
        best_distance)
    anchor_metrics['action_landscape/anchor_near_interaction'] = float(
        np.isfinite(best_distance) and best_distance <= interaction_threshold)
    phase = counterfactual_outcomes.interaction_phase(
        best_distance, interaction_threshold)
    for phase_name in ('approach', 'precontact', 'contact', 'unknown'):
      anchor_metrics[f'action_landscape/anchor_phase_{phase_name}'] = float(
          phase == phase_name)
    anchor_metrics['action_landscape/action_repeat'] = float(action_repeat)
    anchor_metrics['action_landscape/use_best_progress'] = float(
        use_best_progress)
    primary_progress = (
        'best_rollout_mechanism_progress'
        if use_best_progress else 'rollout_mechanism_progress')
    anchor_metrics[
        'action_landscape/aligned_score_vs_progress_spearman'] = (
            anchor_metrics[
                f'action_landscape/score_vs_{primary_progress}_spearman'])
    anchor_metrics['action_landscape/benchmark_success_available_fraction'] = (
        float(np.mean(outcomes['benchmark_success_available'])))
    available = np.asarray(outcomes['benchmark_success_available']) > 0.5
    if np.any(available):
      benchmark = np.asarray(outcomes['benchmark_success'])[available] > 0.5
      proxy = np.asarray(outcomes['proxy_success'])[available] > 0.5
      anchor_metrics['action_landscape/success_predicate_agreement'] = float(
          np.mean(benchmark == proxy))
      anchor_metrics['action_landscape/proxy_false_positive_fraction'] = float(
          np.mean(proxy & ~benchmark))
      anchor_metrics['action_landscape/proxy_false_negative_fraction'] = float(
          np.mean(~proxy & benchmark))
    else:
      anchor_metrics['action_landscape/success_predicate_agreement'] = 0.0
      anchor_metrics['action_landscape/proxy_false_positive_fraction'] = 0.0
      anchor_metrics['action_landscape/proxy_false_negative_fraction'] = 0.0
    per_anchor.append(anchor_metrics)

  keys = sorted(set().union(*(metrics.keys() for metrics in per_anchor)))
  aggregated = {
      key: float(np.mean([metrics[key] for metrics in per_anchor
                          if key in metrics]))
      for key in keys
  }
  aggregated['action_landscape/num_anchors'] = float(num_anchors)
  return aggregated


_SELF_TEST_GOALS = {
    'sawyer_handle_press_side': np.array([-0.07, 0.68, 0.07]),
    'sawyer_window_close': np.array([0.0, 0.80, 0.2]),
}


def _self_test_environment(env_name: str, seed: int) -> None:
  # Imports stay local so dependency-light unit tests can import this module
  # without TensorFlow, Acme, Meta-World, JAX, or mujoco-py installed.
  from contrastive import utils as contrastive_utils  # pylint: disable=g-import-not-at-top
  environment, _ = contrastive_utils.make_environment(
      env_name, 0, -1, seed,
      fixed_start_end=_SELF_TEST_GOALS[env_name])
  try:
    action = np.zeros(environment.action_spec().shape, dtype=np.float32)
    assert_restore_step_reproducible(environment, action)
  finally:
    try:
      environment.close()
    except Exception:
      pass


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Validate MuJoCo restore reproducibility for DCC probes.')
  parser.add_argument(
      '--self-test-env', action='append', choices=sorted(_SELF_TEST_GOALS),
      required=True)
  parser.add_argument('--seed', type=int, default=5)
  args = parser.parse_args()
  for env_name in args.self_test_env:
    _self_test_environment(env_name, args.seed)
    print(f'[action-landscape self-test] {env_name}: PASS', flush=True)


if __name__ == '__main__':
  main()
