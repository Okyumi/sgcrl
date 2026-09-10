"""Diagnostics for post-discovery success propagation into HER / actor data.

D2 success-trace: does φ(s,π)ᵀψ(g_task) rise on successful episodes and stay
lower on failures after first discovery? (Demystifying “success trace”)

D3 actor-follow: at near-object unsolved states, does π match the critic’s
preferred action under g_task?

D4 success-inject: at first eval success spike, write N transitions from
successful episodes into Reverb (no Success-BC) and watch whether HER success
mass / eval then rise.

These helpers never change the training objective unless D4 is explicitly
enabled by the caller.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import dm_env
import numpy as np

from contrastive import critic_phase_probe as phase_probe
from contrastive import her_future_phase as her_phase


def resolve_task_goal(
    env_name: str,
    obs_dim: int,
    *,
    observation: Optional[np.ndarray] = None,
    reachable_goals: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
  """Return a length-``obs_dim`` task goal vector for scoring under g_task."""
  if reachable_goals and env_name in reachable_goals:
    goal = np.asarray(reachable_goals[env_name], dtype=np.float32).reshape(-1)
    if goal.shape[0] < obs_dim:
      goal = np.pad(goal, (0, obs_dim - goal.shape[0]))
    return goal[:obs_dim].astype(np.float32)
  if observation is None:
    raise ValueError(
        f'No reachable goal for {env_name!r}; pass an observation to take '
        'its goal half.')
  obs = np.asarray(observation, dtype=np.float32).reshape(-1)
  if obs.shape[0] < 2 * obs_dim:
    raise ValueError(
        f'Observation too short for goal half: shape={obs.shape}, '
        f'obs_dim={obs_dim}')
  return obs[obs_dim:2 * obs_dim].astype(np.float32)


def is_near_object_unsolved(
    state: np.ndarray,
    env_name: str,
) -> bool:
  """True for hover / near-object unsolved states used as D3 anchors."""
  if env_name not in her_phase.TASK_GEOMETRY:
    return False
  return her_phase.classify_her_goal(state, env_name) == her_phase.PHASE_HOVER


def aggregate_success_trace_metrics(
    scores: Sequence[float],
    episode_success_flags: Sequence[bool],
    *,
    near_unsolved_flags: Optional[Sequence[bool]] = None,
) -> Dict[str, float]:
  """Reduce per-transition similarity into success- vs fail-episode means."""
  scores_arr = np.asarray(list(scores), dtype=np.float32)
  flags = np.asarray(list(episode_success_flags), dtype=bool)
  if scores_arr.shape[0] != flags.shape[0]:
    raise ValueError('scores and episode_success_flags length mismatch')

  succ = scores_arr[flags]
  fail = scores_arr[~flags]
  metrics = {
      'trace/n_success_ep_transitions': float(succ.size),
      'trace/n_fail_ep_transitions': float(fail.size),
      'trace/score_success_ep_mean': (
          float(np.mean(succ)) if succ.size else float('nan')),
      'trace/score_fail_ep_mean': (
          float(np.mean(fail)) if fail.size else float('nan')),
      'trace/gap_success_minus_fail': float('nan'),
  }
  if succ.size and fail.size:
    metrics['trace/gap_success_minus_fail'] = (
        metrics['trace/score_success_ep_mean']
        - metrics['trace/score_fail_ep_mean'])

  if near_unsolved_flags is not None:
    near = np.asarray(list(near_unsolved_flags), dtype=bool)
    if near.shape[0] != flags.shape[0]:
      raise ValueError('near_unsolved_flags length mismatch')
    fail_hover = scores_arr[(~flags) & near]
    metrics['trace/n_fail_hover_transitions'] = float(fail_hover.size)
    metrics['trace/score_fail_hover_mean'] = (
        float(np.mean(fail_hover)) if fail_hover.size else float('nan'))
    if succ.size and fail_hover.size:
      metrics['trace/gap_success_minus_fail_hover'] = (
          metrics['trace/score_success_ep_mean']
          - metrics['trace/score_fail_hover_mean'])
    else:
      metrics['trace/gap_success_minus_fail_hover'] = float('nan')
  else:
    metrics['trace/n_fail_hover_transitions'] = 0.0
    metrics['trace/score_fail_hover_mean'] = float('nan')
    metrics['trace/gap_success_minus_fail_hover'] = float('nan')
  return metrics


def run_success_trace_probe(
    *,
    environment: Any,
    actor: Any,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    task_goal: np.ndarray,
    env_name: str,
    obs_dim: int,
    num_episodes: int = 10,
    max_episode_steps: int = 150,
) -> Dict[str, float]:
  """D2: mean φ(s,π)ᵀψ(g_task) on successful vs failed eval episodes."""
  raw_obs, raw_actions, episode_flags = phase_probe.collect_policy_transitions(
      environment, actor, num_episodes=num_episodes,
      max_episode_steps=max_episode_steps)
  if not raw_obs:
    return aggregate_success_trace_metrics([], [])

  observations = []
  actions = []
  near_flags = []
  for obs, action in zip(raw_obs, raw_actions):
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    state = obs[:obs_dim]
    observations.append(
        phase_probe.build_task_goal_observation(
            state, task_goal, obs_dim=obs_dim))
    actions.append(np.asarray(action, dtype=np.float32))
    near_flags.append(is_near_object_unsolved(state, env_name))

  scores = phase_probe.score_observations_actions(
      score_fn,
      np.asarray(observations, dtype=np.float32),
      np.asarray(actions, dtype=np.float32),
  )
  metrics = aggregate_success_trace_metrics(
      scores.tolist(), episode_flags, near_unsolved_flags=near_flags)
  metrics['trace/num_episodes'] = float(num_episodes)
  metrics['trace/num_transitions'] = float(len(raw_obs))
  metrics['trace/frac_success_episodes'] = float(
      np.mean(np.asarray(episode_flags, dtype=np.float32)))
  return metrics


def _sample_candidate_actions(
    pi_action: np.ndarray,
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
    num_candidates: int,
    local_noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
  """Build [K, A] candidates: π, local noise around π, and uniform."""
  pi_action = np.asarray(pi_action, dtype=np.float32).reshape(-1)
  k = max(int(num_candidates), 3)
  n_local = max((k - 1) // 2, 1)
  n_uniform = k - 1 - n_local
  local = pi_action[None, :] + rng.normal(
      0.0, local_noise_std, size=(n_local, pi_action.shape[0]))
  local = np.clip(local, action_min, action_max)
  uniform = rng.uniform(
      action_min, action_max, size=(n_uniform, pi_action.shape[0]))
  return np.concatenate(
      [pi_action.reshape(1, -1), local, uniform], axis=0).astype(np.float32)


def aggregate_actor_follow_metrics(
    pi_ranks: Sequence[int],
    score_gaps: Sequence[float],
    action_l2s: Sequence[float],
) -> Dict[str, float]:
  ranks = np.asarray(list(pi_ranks), dtype=np.float32)
  gaps = np.asarray(list(score_gaps), dtype=np.float32)
  l2s = np.asarray(list(action_l2s), dtype=np.float32)
  if ranks.size == 0:
    return {
        'follow/n_anchors': 0.0,
        'follow/pi_rank_mean': float('nan'),
        'follow/pi_is_argmax_frac': float('nan'),
        'follow/score_gap_argmax_minus_pi_mean': float('nan'),
        'follow/action_l2_pi_vs_argmax_mean': float('nan'),
    }
  return {
      'follow/n_anchors': float(ranks.size),
      'follow/pi_rank_mean': float(np.mean(ranks)),
      'follow/pi_is_argmax_frac': float(np.mean(ranks <= 0.5)),
      'follow/score_gap_argmax_minus_pi_mean': float(np.mean(gaps)),
      'follow/action_l2_pi_vs_argmax_mean': float(np.mean(l2s)),
  }


def run_actor_follow_probe(
    *,
    environment: Any,
    actor: Any,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    task_goal: np.ndarray,
    env_name: str,
    obs_dim: int,
    num_episodes: int = 10,
    max_anchors: int = 16,
    num_candidates: int = 32,
    local_noise_std: float = 0.10,
    max_episode_steps: int = 150,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
  """D3: critic argmax vs π at near-object unsolved states under g_task."""
  rng = rng or np.random.default_rng(0)
  action_spec = environment.action_spec()
  action_min = np.broadcast_to(
      np.asarray(action_spec.minimum, dtype=np.float32), action_spec.shape)
  action_max = np.broadcast_to(
      np.asarray(action_spec.maximum, dtype=np.float32), action_spec.shape)

  raw_obs, raw_actions, _ = phase_probe.collect_policy_transitions(
      environment, actor, num_episodes=num_episodes,
      max_episode_steps=max_episode_steps)

  ranks: List[int] = []
  gaps: List[float] = []
  l2s: List[float] = []
  for obs, pi_action in zip(raw_obs, raw_actions):
    if len(ranks) >= max_anchors:
      break
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    state = obs[:obs_dim]
    if not is_near_object_unsolved(state, env_name):
      continue
    task_obs = phase_probe.build_task_goal_observation(
        state, task_goal, obs_dim=obs_dim)
    pi_action = np.asarray(pi_action, dtype=np.float32).reshape(-1)
    candidates = _sample_candidate_actions(
        pi_action,
        action_min=action_min,
        action_max=action_max,
        num_candidates=num_candidates,
        local_noise_std=local_noise_std,
        rng=rng,
    )
    # Ensure index 0 is exactly π.
    candidates[0] = pi_action
    scores = np.asarray(
        score_fn(task_obs, candidates), dtype=np.float32).reshape(-1)
    if scores.shape[0] != candidates.shape[0]:
      raise ValueError(
          f'score_fn returned {scores.shape}, expected [{candidates.shape[0]}]')
    argmax = int(np.argmax(scores))
    order = np.argsort(-scores)
    pi_rank = int(np.where(order == 0)[0][0])
    ranks.append(pi_rank)
    gaps.append(float(scores[argmax] - scores[0]))
    l2s.append(float(np.linalg.norm(candidates[argmax] - pi_action)))

  metrics = aggregate_actor_follow_metrics(ranks, gaps, l2s)
  metrics['follow/num_episodes'] = float(num_episodes)
  metrics['follow/num_candidates'] = float(num_candidates)
  return metrics


def _copy_timestep(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  obs = None if timestep.observation is None else np.array(
      timestep.observation, copy=True)
  return dm_env.TimeStep(
      step_type=timestep.step_type,
      reward=timestep.reward,
      discount=timestep.discount,
      observation=obs,
  )


def collect_episode_record(
    environment: Any,
    actor: Any,
    *,
    max_episode_steps: int = 150,
) -> Tuple[dm_env.TimeStep, List[Tuple[np.ndarray, dm_env.TimeStep]], bool, int]:
  """Roll one episode; return (first_ts, [(action, next_ts)...], success, n)."""
  first = _copy_timestep(environment.reset())
  timestep = first
  records: List[Tuple[np.ndarray, dm_env.TimeStep]] = []
  episode_reward = 0.0
  steps = 0
  while not timestep.last() and steps < max_episode_steps:
    action = np.asarray(actor.select_action(timestep.observation),
                        dtype=np.float32)
    timestep = environment.step(action)
    records.append((action.copy(), _copy_timestep(timestep)))
    episode_reward += float(timestep.reward or 0.0)
    steps += 1
  return first, records, episode_reward > 0.5, steps


def write_episode_to_adder(
    adder: Any,
    first_timestep: dm_env.TimeStep,
    records: Sequence[Tuple[np.ndarray, dm_env.TimeStep]],
) -> int:
  """Write one complete episode into an EpisodeAdder. Returns #transitions."""
  if hasattr(adder, 'reset'):
    adder.reset()
  adder.add_first(first_timestep)
  for action, timestep in records:
    adder.add(action, timestep)
  return len(records)


def classify_state_stage(state: np.ndarray, env_name: str) -> str:
  """Label one state for dwell counting (handle vs push)."""
  state = np.asarray(state, dtype=np.float32).reshape(-1)
  hand = state[:3]
  obj = state[4:7]
  hand_obj = float(np.linalg.norm(hand - obj[:3]))
  if env_name == 'sawyer_handle_press_side':
    axis_ok = abs(float(state[6]) - 0.07) <= 0.02
    if axis_ok and hand_obj <= 0.09:
      return 'success'
    if hand_obj <= 0.09:
      return 'hover_contact'
    if hand_obj <= 0.15:
      return 'near_approach'
    return 'far'
  # push (and push-like): continuous object distance bands
  target = np.array([0.02, 0.89, 0.02], dtype=np.float32)
  if env_name in her_phase.TASK_GEOMETRY:
    spec = her_phase.TASK_GEOMETRY[env_name]
    if 'fixed_object_target' in spec:
      target = np.asarray(spec['fixed_object_target'], dtype=np.float32)
  obj_dist = float(np.linalg.norm(obj[:3] - target[:3]))
  if obj_dist <= 0.05:
    return 'success'
  if obj_dist <= 0.15:
    return 'object_progress'
  if hand_obj <= 0.09:
    return 'hover_contact'
  if hand_obj <= 0.15:
    return 'near_approach'
  return 'far'


def summarize_stage_dwell(
    labels: Sequence[str],
) -> Dict[str, float]:
  """Count how many of 150 steps were spent in each stage."""
  stages = (
      'far', 'near_approach', 'hover_contact', 'object_progress', 'success')
  n = max(len(labels), 1)
  out: Dict[str, float] = {'dwell/horizon': float(len(labels))}
  for stage in stages:
    count = float(list(labels).count(stage))
    out[f'dwell/steps_{stage}'] = count
    out[f'dwell/frac_{stage}'] = count / n
  return out


def aggregate_successful_episode_dwell(
    *,
    environment: Any,
    actor: Any,
    env_name: str,
    obs_dim: int,
    num_episodes: int = 20,
    max_episode_steps: int = 150,
) -> Dict[str, float]:
  """Roll policy; average stage dwell over *successful* episodes only."""
  succ_summaries: List[Dict[str, float]] = []
  n_success = 0
  for _ in range(int(num_episodes)):
    timestep = environment.reset()
    labels = []
    reward_sum = 0.0
    steps = 0
    while not timestep.last() and steps < max_episode_steps:
      obs = np.asarray(timestep.observation, dtype=np.float32).reshape(-1)
      labels.append(classify_state_stage(obs[:obs_dim], env_name))
      action = actor.select_action(timestep.observation)
      timestep = environment.step(action)
      reward_sum += float(timestep.reward or 0.0)
      steps += 1
    if reward_sum > 0.5 and labels:
      n_success += 1
      succ_summaries.append(summarize_stage_dwell(labels))
  metrics = {
      'dwell/n_success_episodes': float(n_success),
      'dwell/n_rollouts': float(num_episodes),
  }
  if not succ_summaries:
    for stage in (
        'far', 'near_approach', 'hover_contact', 'object_progress', 'success'):
      metrics[f'dwell/steps_{stage}'] = float('nan')
      metrics[f'dwell/frac_{stage}'] = float('nan')
    metrics['dwell/horizon'] = float('nan')
    return metrics
  keys = succ_summaries[0].keys()
  for key in keys:
    metrics[key] = float(np.mean([row[key] for row in succ_summaries]))
  return metrics


def press_bias_action(pi_action: np.ndarray, env_name: str) -> np.ndarray:
  """Simple press / push-forward action for landscape comparison."""
  action = np.asarray(pi_action, dtype=np.float32).reshape(-1).copy()
  if env_name == 'sawyer_handle_press_side':
    action[2] = -1.0
  else:
    action[1] = 1.0
  return np.clip(action, -1.0, 1.0)


def run_press_vs_pi_probe(
    *,
    environment: Any,
    actor: Any,
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    task_goal: np.ndarray,
    env_name: str,
    obs_dim: int,
    num_episodes: int = 10,
    max_anchors: int = 32,
    max_episode_steps: int = 150,
) -> Dict[str, float]:
  """At hover states: score(press_bias) - score(π) under g_task."""
  gaps = []
  raw_obs, raw_actions, _ = phase_probe.collect_policy_transitions(
      environment, actor, num_episodes=num_episodes,
      max_episode_steps=max_episode_steps)
  for obs, pi_action in zip(raw_obs, raw_actions):
    if len(gaps) >= max_anchors:
      break
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    state = obs[:obs_dim]
    if classify_state_stage(state, env_name) != 'hover_contact':
      continue
    task_obs = phase_probe.build_task_goal_observation(
        state, task_goal, obs_dim=obs_dim)
    pi_action = np.asarray(pi_action, dtype=np.float32).reshape(-1)
    cand = np.stack([pi_action, press_bias_action(pi_action, env_name)], 0)
    scores = np.asarray(score_fn(task_obs, cand), dtype=np.float32).reshape(-1)
    gaps.append(float(scores[1] - scores[0]))
  if not gaps:
    return {
        'presspi/n_hover_anchors': 0.0,
        'presspi/gap_press_minus_pi_mean': float('nan'),
        'presspi/press_better_frac': float('nan'),
    }
  arr = np.asarray(gaps, dtype=np.float32)
  return {
      'presspi/n_hover_anchors': float(arr.size),
      'presspi/gap_press_minus_pi_mean': float(np.mean(arr)),
      'presspi/press_better_frac': float(np.mean(arr > 0.0)),
  }


def inject_successful_episodes(
    *,
    environment: Any,
    actor: Any,
    adder: Any,
    n_transitions: int,
    max_attempts: int = 40,
    max_episode_steps: int = 150,
    clone: bool = False,
) -> Dict[str, float]:
  """Inject successful episodes into replay.

  If ``clone`` is True, keep rewriting the collected successful episode(s)
  until ``n_transitions`` is reached (mass-matched counterfactual).
  """
  injected = 0
  success_episodes_unique = 0
  clones_written = 0
  attempts = 0
  bank: List[Tuple[Any, List[Tuple[np.ndarray, dm_env.TimeStep]]]] = []

  while injected < int(n_transitions) and attempts < int(max_attempts):
    attempts += 1
    first, records, succeeded, _ = collect_episode_record(
        environment, actor, max_episode_steps=max_episode_steps)
    if not succeeded or not records:
      continue
    bank.append((first, records))
    written = write_episode_to_adder(adder, first, records)
    injected += written
    success_episodes_unique += 1

  if clone and bank and injected < int(n_transitions):
    idx = 0
    # Enough clones to hit the target even for short episodes.
    ep_len = max(len(bank[0][1]), 1)
    max_clone_writes = int(n_transitions) // ep_len + len(bank) + 5
    while injected < int(n_transitions) and clones_written < max_clone_writes:
      first, records = bank[idx % len(bank)]
      idx += 1
      written = write_episode_to_adder(adder, first, records)
      injected += written
      clones_written += 1

  return {
      'inject/n_transitions': float(injected),
      'inject/n_success_episodes': float(success_episodes_unique),
      'inject/n_clones': float(clones_written),
      'inject/n_attempts': float(attempts),
      'inject/triggered': 1.0 if injected > 0 else 0.0,
      'inject/target_n_transitions': float(n_transitions),
  }
