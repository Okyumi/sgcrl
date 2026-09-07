"""Online/offline critic probe for Task-5/8 peak-then-drop diagnosis.

Hypothesis under test
---------------------
After early success discovery, the contrastive critic may assign high scores
to non-success phases (hover near the mechanism, mid-reach) when scored
against the *fixed task goal*.  If hover ≈ success ≫ mid-reach, the actor
can exploit a nearby non-success mode.  If success scores collapse relative
to hover/reach while distances worsen, the failure looks more like phase
drowning / nonstationary critic fitting than a geometric fake-goal trap.

This module never changes the training objective.  It only rolls out the
current deterministic policy, classifies transitions into three families,
and scores ``φ(s,a)ᵀψ(g_task)`` for each family.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


FAMILY_SUCCESS = 'success'
FAMILY_HOVER = 'hover'
FAMILY_MID_REACH = 'mid_reach'
FAMILIES = (FAMILY_SUCCESS, FAMILY_HOVER, FAMILY_MID_REACH)


def _as_float_array(values: Sequence[float]) -> np.ndarray:
  return np.asarray(list(values), dtype=np.float32)


def classify_transition(
    state: np.ndarray,
    *,
    interaction_threshold: float = 0.09,
    mid_reach_threshold: float = 0.15,
    axis_index: int = 6,
    axis_target: float = 0.07,
    axis_threshold: float = 0.02,
    episode_succeeded: bool = False,
) -> Optional[str]:
  """Assign a transition to at most one diagnostic family.

  Priority: success > hover > mid_reach.  Ambiguous / idle states return
  ``None`` and are ignored by the probe aggregate.
  """
  state = np.asarray(state, dtype=np.float32).reshape(-1)
  if state.shape[0] < 7:
    raise ValueError('Task-5/8 probe expects at least a 7-D Sawyer state.')
  hand = state[:3]
  mechanism = state[4:7]
  hand_mechanism = float(np.linalg.norm(hand - mechanism))
  axis_ok = abs(float(state[axis_index]) - float(axis_target)) <= float(
      axis_threshold)

  if episode_succeeded and axis_ok and hand_mechanism <= interaction_threshold:
    return FAMILY_SUCCESS
  if (hand_mechanism <= interaction_threshold) and (not axis_ok):
    return FAMILY_HOVER
  if hand_mechanism >= mid_reach_threshold:
    return FAMILY_MID_REACH
  return None


def task58_axis_spec(env_name: str) -> Tuple[int, float, float]:
  """Return (axis_index, axis_target, axis_threshold) for Task 5/8."""
  if env_name == 'sawyer_handle_press_side':
    return 6, 0.07, 0.02
  if env_name == 'sawyer_window_close':
    # Window mechanism target is (x=0, y=0.80, z=0.20); success is on y.
    return 5, 0.80, 0.02
  raise ValueError(
      f'critic_phase_probe only supports Task-5/8 env names; got {env_name!r}')


def build_task_goal_observation(
    state: np.ndarray,
    task_goal: np.ndarray,
    *,
    obs_dim: int,
) -> np.ndarray:
  """Concatenate state and padded task goal into the actor/critic obs layout."""
  state = np.asarray(state, dtype=np.float32).reshape(-1)
  goal = np.asarray(task_goal, dtype=np.float32).reshape(-1)
  if state.shape[0] < obs_dim:
    state = np.pad(state, (0, obs_dim - state.shape[0]))
  else:
    state = state[:obs_dim]
  if goal.shape[0] < obs_dim:
    goal = np.pad(goal, (0, obs_dim - goal.shape[0]))
  else:
    goal = goal[:obs_dim]
  return np.concatenate([state, goal], axis=0).astype(np.float32)


def aggregate_family_scores(
    family_scores: Mapping[str, Sequence[float]],
) -> Dict[str, float]:
  """Reduce per-family score lists into W&B-friendly scalars."""
  metrics: Dict[str, float] = {}
  means = {}
  for family in FAMILIES:
    values = _as_float_array(family_scores.get(family, []))
    metrics[f'probe/n_{family}'] = float(values.size)
    if values.size:
      means[family] = float(np.mean(values))
      metrics[f'probe/score_{family}_vs_task_goal_mean'] = means[family]
      metrics[f'probe/score_{family}_vs_task_goal_std'] = float(np.std(values))
    else:
      means[family] = float('nan')
      metrics[f'probe/score_{family}_vs_task_goal_mean'] = float('nan')
      metrics[f'probe/score_{family}_vs_task_goal_std'] = float('nan')

  if np.isfinite(means[FAMILY_SUCCESS]) and np.isfinite(means[FAMILY_HOVER]):
    metrics['probe/gap_success_minus_hover'] = (
        means[FAMILY_SUCCESS] - means[FAMILY_HOVER])
  else:
    metrics['probe/gap_success_minus_hover'] = float('nan')

  if (np.isfinite(means[FAMILY_SUCCESS])
      and np.isfinite(means[FAMILY_MID_REACH])):
    metrics['probe/gap_success_minus_mid_reach'] = (
        means[FAMILY_SUCCESS] - means[FAMILY_MID_REACH])
  else:
    metrics['probe/gap_success_minus_mid_reach'] = float('nan')

  # Soft confirmation of the user's geometric "fake goal" story.
  if (np.isfinite(means[FAMILY_SUCCESS]) and np.isfinite(means[FAMILY_HOVER])
      and abs(means[FAMILY_SUCCESS]) > 1e-6):
    metrics['probe/hover_to_success_score_ratio'] = (
        means[FAMILY_HOVER] / means[FAMILY_SUCCESS])
  else:
    metrics['probe/hover_to_success_score_ratio'] = float('nan')
  return metrics


def _mean_finite(values: Sequence[float]) -> float:
  arr = _as_float_array(values)
  if arr.size == 0:
    return float('nan')
  return float(np.mean(arr))


def aggregate_goal_embedding_metrics(
    *,
    psi_task: np.ndarray,
    psi_hover: np.ndarray,
    success_scores_under_hover_goal: Sequence[float],
    hover_scores_under_hover_goal: Sequence[float],
    success_scores_under_task_goal: Sequence[float],
    hover_scores_under_task_goal: Sequence[float],
) -> Dict[str, float]:
  """Metrics for ψ(g_task) vs ψ(g_hover) and cross-goal score swaps.

  Distinguishes two claims:
  - phase ranking under fixed g_task (already in aggregate_family_scores)
  - goal-embedding aliasing / anti-press under hover-like goals
  """
  psi_task = np.asarray(psi_task, dtype=np.float32).reshape(-1)
  psi_hover = np.asarray(psi_hover, dtype=np.float32).reshape(-1)
  diff = psi_task - psi_hover
  l2 = float(np.linalg.norm(diff))
  denom = (
      float(np.linalg.norm(psi_task)) * float(np.linalg.norm(psi_hover)))
  cosine = float(np.dot(psi_task, psi_hover) / denom) if denom > 1e-8 else float(
      'nan')

  s_task = _mean_finite(success_scores_under_task_goal)
  h_task = _mean_finite(hover_scores_under_task_goal)
  s_hover_g = _mean_finite(success_scores_under_hover_goal)
  h_hover_g = _mean_finite(hover_scores_under_hover_goal)

  metrics = {
      'probe/psi_l2_task_vs_hover': l2,
      'probe/psi_cosine_task_vs_hover': cosine,
      'probe/score_success_vs_hover_goal_mean': s_hover_g,
      'probe/score_hover_vs_hover_goal_mean': h_hover_g,
      'probe/gap_success_minus_hover_under_hover_goal': (
          s_hover_g - h_hover_g
          if np.isfinite(s_hover_g) and np.isfinite(h_hover_g)
          else float('nan')),
      # Same (success) transitions scored under the two goals.
      'probe/gap_success_task_goal_minus_hover_goal': (
          s_task - s_hover_g
          if np.isfinite(s_task) and np.isfinite(s_hover_g)
          else float('nan')),
      # Same (hover) transitions scored under the two goals.
      'probe/gap_hover_task_goal_minus_hover_goal': (
          h_task - h_hover_g
          if np.isfinite(h_task) and np.isfinite(h_hover_g)
          else float('nan')),
  }
  return metrics


def score_observations_actions(
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
  """Score (obs, action) rows with the learner's paired contrastive score."""
  observations = np.asarray(observations, dtype=np.float32)
  actions = np.asarray(actions, dtype=np.float32)
  if observations.shape[0] == 0:
    return np.zeros((0,), dtype=np.float32)
  outs = []
  for start in range(0, observations.shape[0], batch_size):
    end = start + batch_size
    chunk_obs = observations[start:end]
    chunk_act = actions[start:end]
    # score_fn is defined on a single observation with many candidate
    # actions in the learner API.  Score one transition at a time for a
    # matched (s,a) pair to avoid accidental broadcasting bugs.
    chunk_scores = []
    for obs_i, act_i in zip(chunk_obs, chunk_act):
      score = np.asarray(
          score_fn(obs_i, act_i.reshape(1, -1)), dtype=np.float32).reshape(-1)
      chunk_scores.append(float(score[0]))
    outs.append(np.asarray(chunk_scores, dtype=np.float32))
  return np.concatenate(outs, axis=0)


def collect_policy_transitions(
    environment: Any,
    actor: Any,
    *,
    num_episodes: int,
    max_episode_steps: int = 150,
) -> Tuple[list, list, list]:
  """Roll deterministic policy episodes; return states, actions, successes."""
  states = []
  actions = []
  episode_success_flags = []
  for _ in range(int(num_episodes)):
    timestep = environment.reset()
    episode_states = []
    episode_actions = []
    episode_reward = 0.0
    steps = 0
    while not timestep.last() and steps < max_episode_steps:
      action = actor.select_action(timestep.observation)
      episode_states.append(np.asarray(timestep.observation, dtype=np.float32))
      episode_actions.append(np.asarray(action, dtype=np.float32))
      timestep = environment.step(action)
      episode_reward += float(timestep.reward or 0.0)
      steps += 1
    succeeded = episode_reward > 0.5
    states.extend(episode_states)
    actions.extend(episode_actions)
    episode_success_flags.extend([succeeded] * len(episode_states))
  return states, actions, episode_success_flags


def run_critic_phase_probe(
    *,
    environment: Any,
    actor: Any,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    task_goal: np.ndarray,
    env_name: str,
    obs_dim: int,
    num_episodes: int = 10,
    interaction_threshold: float = 0.09,
    mid_reach_threshold: float = 0.15,
    max_episode_steps: int = 150,
    max_per_family: int = 256,
    embed_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, float]:
  """Collect policy rollouts, classify phases, score against task goal.

  When ``embed_fn`` is provided and hover samples exist, also logs
  ``ψ(g_task)`` vs ``ψ(g_hover)`` geometry and re-scores success/hover
  transitions under a hover-state goal (HER-like near-contact future).
  """
  axis_index, axis_target, axis_threshold = task58_axis_spec(env_name)
  raw_obs, raw_actions, episode_flags = collect_policy_transitions(
      environment, actor, num_episodes=num_episodes,
      max_episode_steps=max_episode_steps)

  buckets = {
      family: {'state': [], 'obs': [], 'act': []} for family in FAMILIES}
  for obs, action, ep_success in zip(raw_obs, raw_actions, episode_flags):
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    state = obs[:obs_dim]
    family = classify_transition(
        state,
        interaction_threshold=interaction_threshold,
        mid_reach_threshold=mid_reach_threshold,
        axis_index=axis_index,
        axis_target=axis_target,
        axis_threshold=axis_threshold,
        episode_succeeded=bool(ep_success),
    )
    if family is None:
      continue
    if len(buckets[family]['obs']) >= max_per_family:
      continue
    task_obs = build_task_goal_observation(state, task_goal, obs_dim=obs_dim)
    buckets[family]['state'].append(np.asarray(state, dtype=np.float32))
    buckets[family]['obs'].append(task_obs)
    buckets[family]['act'].append(np.asarray(action, dtype=np.float32))

  family_scores = {}
  for family in FAMILIES:
    obs_arr = np.asarray(buckets[family]['obs'], dtype=np.float32)
    act_arr = np.asarray(buckets[family]['act'], dtype=np.float32)
    if obs_arr.size == 0:
      family_scores[family] = []
      continue
    scores = score_observations_actions(score_fn, obs_arr, act_arr)
    family_scores[family] = scores.tolist()

  metrics = aggregate_family_scores(family_scores)
  metrics['probe/num_episodes'] = float(num_episodes)
  metrics['probe/num_transitions_seen'] = float(len(raw_obs))

  hover_states = buckets[FAMILY_HOVER]['state']
  if embed_fn is not None and hover_states:
    hover_goal = np.mean(
        np.asarray(hover_states, dtype=np.float32), axis=0)
    # Dummy state; apply_psi reads only the goal half of the observation.
    zero_state = np.zeros((obs_dim,), dtype=np.float32)
    task_goal_obs = build_task_goal_observation(
        zero_state, task_goal, obs_dim=obs_dim)
    hover_goal_obs = build_task_goal_observation(
        zero_state, hover_goal, obs_dim=obs_dim)
    psi_task = np.asarray(embed_fn(task_goal_obs), dtype=np.float32).reshape(-1)
    psi_hover = np.asarray(
        embed_fn(hover_goal_obs), dtype=np.float32).reshape(-1)

    success_under_hover = []
    hover_under_hover = []
    for family, out_list in (
        (FAMILY_SUCCESS, success_under_hover),
        (FAMILY_HOVER, hover_under_hover),
    ):
      for state, action in zip(
          buckets[family]['state'], buckets[family]['act']):
        hover_obs = build_task_goal_observation(
            state, hover_goal, obs_dim=obs_dim)
        score = np.asarray(
            score_fn(hover_obs, np.asarray(action).reshape(1, -1)),
            dtype=np.float32).reshape(-1)
        out_list.append(float(score[0]))

    metrics.update(aggregate_goal_embedding_metrics(
        psi_task=psi_task,
        psi_hover=psi_hover,
        success_scores_under_hover_goal=success_under_hover,
        hover_scores_under_hover_goal=hover_under_hover,
        success_scores_under_task_goal=family_scores.get(FAMILY_SUCCESS, []),
        hover_scores_under_task_goal=family_scores.get(FAMILY_HOVER, []),
    ))
  else:
    metrics.update({
        'probe/psi_l2_task_vs_hover': float('nan'),
        'probe/psi_cosine_task_vs_hover': float('nan'),
        'probe/score_success_vs_hover_goal_mean': float('nan'),
        'probe/score_hover_vs_hover_goal_mean': float('nan'),
        'probe/gap_success_minus_hover_under_hover_goal': float('nan'),
        'probe/gap_success_task_goal_minus_hover_goal': float('nan'),
        'probe/gap_hover_task_goal_minus_hover_goal': float('nan'),
    })
  return metrics
