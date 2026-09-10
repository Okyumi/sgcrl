#!/usr/bin/env python3
"""Same-state action-advice audit for contrastive critic at hover.

Answers: does liking success *states* imply preferring press *actions* at hover?

For each hover (and optionally success) state under a fixed goal embedding:
  - score(π), score(press-biased), score(random actions)
  - action-sensitivity: std_a score(s, a, g) vs state-sensitivity across hover states
  - under g_task AND under a hover-goal (state‖state as desired goal)
  - local gradient of score w.r.t. action at a_π: does it point toward press?

This is offline over a mid-task checkpoint (no training).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

FIXED_GOALS = {
    'sawyer_handle_press_side': np.array([-0.07, 0.68, 0.07], dtype=np.float32),
    'sawyer_push': np.array([0.02, 0.89, 0.02], dtype=np.float32),
}
PUSH_TARGET = FIXED_GOALS['sawyer_push']


class _PolicyActor:
  def __init__(self, apply_mode):
    self._apply_mode = apply_mode

  def select_action(self, observation):
    return np.asarray(self._apply_mode(observation), dtype=np.float32)


def classify_state(state: np.ndarray, env_name: str) -> str:
  state = np.asarray(state, dtype=np.float32).reshape(-1)
  hand = state[:3]
  obj = state[4:7]
  hand_obj = float(np.linalg.norm(hand - obj))
  if env_name == 'sawyer_handle_press_side':
    axis_ok = abs(float(state[6]) - 0.07) <= 0.02
    if axis_ok and hand_obj <= 0.09:
      return 'success'
    if hand_obj <= 0.09:
      return 'hover_contact'
    if hand_obj <= 0.15:
      return 'near_approach'
    return 'far'
  obj_dist = float(np.linalg.norm(obj - PUSH_TARGET))
  if obj_dist <= 0.05:
    return 'success'
  if obj_dist <= 0.15:
    return 'object_progress'
  if hand_obj <= 0.09:
    return 'hover_contact'
  if hand_obj <= 0.15:
    return 'near_approach'
  return 'far'


def biased_action(pi_action: np.ndarray, env_name: str) -> np.ndarray:
  a = np.array(pi_action, dtype=np.float32, copy=True)
  if env_name == 'sawyer_handle_press_side':
    a[2] = -1.0
  else:
    a[1] = 1.0
  return np.clip(a, -1.0, 1.0)


def press_direction(env_name: str) -> np.ndarray:
  d = np.zeros(4, dtype=np.float32)
  if env_name == 'sawyer_handle_press_side':
    d[2] = -1.0
  else:
    d[1] = 1.0
  return d


def summarize_gaps(rows: List[Dict[str, float]]) -> Dict[str, float]:
  if not rows:
    return {}
  keys = rows[0].keys()
  out = {}
  for k in keys:
    vals = np.asarray([r[k] for r in rows], dtype=np.float64)
    out[f'{k}_mean'] = float(np.mean(vals))
    out[f'{k}_std'] = float(np.std(vals))
    out[f'{k}_median'] = float(np.median(vals))
  out['n'] = float(len(rows))
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--env-name', required=True, choices=sorted(FIXED_GOALS))
  parser.add_argument('--seed', type=int, default=6)
  parser.add_argument('--episodes', type=int, default=40)
  parser.add_argument('--n-random-actions', type=int, default=32)
  parser.add_argument('--max-hover-per-ep', type=int, default=8)
  parser.add_argument('--max-success-per-ep', type=int, default=4)
  parser.add_argument('--network-width', type=int, default=1024)
  parser.add_argument('--critic-depth', type=int, default=4)
  parser.add_argument('--actor-depth', type=int, default=4)
  parser.add_argument('--phi-task-width', type=int, default=256)
  parser.add_argument('--phi-task-depth', type=int, default=4)
  parser.add_argument('--output', default='')
  args = parser.parse_args()

  import jax
  import jax.numpy as jnp
  from acme import specs

  import contrastive
  from contrastive import critic_phase_probe
  from contrastive import utils as contrastive_utils
  from contrastive.decomposed_networks import make_decomposed_networks
  import env_utils

  ckpt_path = Path(args.checkpoint).expanduser().resolve()
  with ckpt_path.open('rb') as handle:
    ckpt = pickle.load(handle)
  if 'decomposed_training_state' not in ckpt:
    raise KeyError('Need mid-task decomposed checkpoint')

  env_name = args.env_name
  start_index = int(ckpt.get('goal_start_index', 0))
  end_index = int(ckpt.get('goal_end_index', -1))
  success_mode = ckpt.get('sawyer_success_mode', 'corrected')
  environment, obs_dim = contrastive_utils.make_environment(
      env_name, start_index, end_index, args.seed,
      fixed_start_end=FIXED_GOALS[env_name],
      sawyer_success_mode=success_mode)
  obs_dim = int(ckpt.get('obs_dim') or obs_dim)
  action_dim = int(environment.action_spec().shape[0])

  env_spec = specs.make_environment_spec(environment)
  networks = contrastive.make_networks(
      env_spec, obs_dim=obs_dim,
      hidden_layer_sizes=(args.network_width, args.network_width),
      use_residual=True, network_width=args.network_width,
      critic_depth=args.critic_depth, actor_depth=args.actor_depth)
  decomp_nets = make_decomposed_networks(
      env_spec, obs_dim=obs_dim, repr_dim=64,
      hidden_layer_sizes=(args.network_width, args.network_width),
      use_residual=True, network_width=args.network_width,
      critic_depth=args.critic_depth,
      phi_task_width=args.phi_task_width, phi_task_depth=args.phi_task_depth,
      combine_mode='add', goal_encoder_mode='shared')

  state = ckpt['decomposed_training_state']
  policy_params = ckpt.get('decomposed_policy_params') or ckpt.get(
      'composed_policy')
  b_shared = ckpt['decomposed_b_shared_params']
  h_phi = ckpt['decomposed_h_phi_params']
  psi = ckpt['decomposed_psi_params']
  phi_task = state.phi_task_params

  def _mode_or_mean(observation):
    dist_params = networks.policy_network.apply(policy_params, observation)
    if hasattr(dist_params, 'mode'):
      return dist_params.mode()
    return dist_params.mean()

  _mode_jit = jax.jit(_mode_or_mean)

  @jax.jit
  def paired_score(observation, actions):
    actions = jnp.asarray(actions)
    obs = jnp.repeat(
        jnp.asarray(observation)[None, :], actions.shape[0], axis=0)
    return decomp_nets.apply_paired_score(
        b_shared, h_phi, phi_task, psi, obs, actions)

  def score_grad_at_action(observation, action):
    obs = jnp.asarray(observation)
    a0 = jnp.asarray(action)

    def _score(a):
      return decomp_nets.apply_paired_score(
          b_shared, h_phi, phi_task, psi, obs[None, :], a[None, :])[0]

    return jax.grad(_score)(a0)

  score_grad_jit = jax.jit(score_grad_at_action)

  actor = _PolicyActor(lambda obs: np.asarray(_mode_jit(obs)))
  rng = np.random.default_rng(args.seed)

  if env_name in env_utils.TASK58_REACHABLE_SUCCESS_GOALS:
    task_goal = env_utils.TASK58_REACHABLE_SUCCESS_GOALS[env_name]
  else:
    ts0 = environment.reset()
    task_goal = np.asarray(ts0.observation, dtype=np.float32)[obs_dim:]

  press_dir = press_direction(env_name)
  hover_rows: List[Dict[str, float]] = []
  success_rows: List[Dict[str, float]] = []
  # For state-vs-action sensitivity pooling
  hover_pi_task_scores: List[float] = []
  hover_action_stds_task: List[float] = []
  hover_action_stds_hover_g: List[float] = []

  def audit_state(
      state_vec: np.ndarray,
      pi_a: np.ndarray,
      stage: str,
  ) -> Dict[str, float]:
    task_obs = critic_phase_probe.build_task_goal_observation(
        state_vec, task_goal, obs_dim=obs_dim)
    # HER-style hover/future goal ≈ "stay here": desired_goal = current state
    hover_obs = np.concatenate([state_vec, state_vec], axis=0).astype(
        np.float32)

    press_a = biased_action(pi_a, env_name)
    rand_a = rng.uniform(-1.0, 1.0, size=(args.n_random_actions, action_dim)
                         ).astype(np.float32)
    cand = np.concatenate(
        [pi_a[None, :], press_a[None, :], rand_a], axis=0)

    scores_task = np.asarray(paired_score(task_obs, cand), dtype=np.float32)
    scores_hover_g = np.asarray(
        paired_score(hover_obs, cand), dtype=np.float32)

    pi_t, press_t = float(scores_task[0]), float(scores_task[1])
    rand_t = scores_task[2:]
    pi_h, press_h = float(scores_hover_g[0]), float(scores_hover_g[1])
    rand_h = scores_hover_g[2:]

    # Rank of press among {π, press, randoms} (0 = best)
    press_rank_task = int(np.sum(scores_task >= press_t))  # 1-best if unique
    press_rank_hover_g = int(np.sum(scores_hover_g >= press_h))

    grad = np.asarray(score_grad_jit(task_obs, pi_a), dtype=np.float32)
    grad_norm = float(np.linalg.norm(grad) + 1e-12)
    cos_press = float(np.dot(grad, press_dir) / (
        grad_norm * (np.linalg.norm(press_dir) + 1e-12)))
    # Finite-difference check: move π toward press on the press axis
    a_step = np.array(pi_a, dtype=np.float32, copy=True)
    if env_name == 'sawyer_handle_press_side':
      a_step[2] = float(np.clip(a_step[2] - 0.25, -1.0, 1.0))
    else:
      a_step[1] = float(np.clip(a_step[1] + 0.25, -1.0, 1.0))
    step_scores = np.asarray(
        paired_score(task_obs, np.stack([pi_a, a_step], axis=0)),
        dtype=np.float32)

    return {
        'stage_is_hover': 1.0 if stage == 'hover_contact' else 0.0,
        'score_pi_task': pi_t,
        'score_press_task': press_t,
        'gap_press_minus_pi_task': press_t - pi_t,
        'score_rand_mean_task': float(np.mean(rand_t)),
        'score_rand_std_task': float(np.std(rand_t)),
        'score_action_std_task': float(np.std(scores_task)),
        'press_beats_pi_task': 1.0 if press_t > pi_t else 0.0,
        'press_beats_rand_mean_task': (
            1.0 if press_t > float(np.mean(rand_t)) else 0.0),
        'press_rank_task': float(press_rank_task),
        'score_pi_hover_g': pi_h,
        'score_press_hover_g': press_h,
        'gap_press_minus_pi_hover_g': press_h - pi_h,
        'score_action_std_hover_g': float(np.std(scores_hover_g)),
        'press_beats_pi_hover_g': 1.0 if press_h > pi_h else 0.0,
        'press_rank_hover_g': float(press_rank_hover_g),
        'grad_norm_task': grad_norm,
        'grad_cos_press_dir': cos_press,
        'local_step_delta_toward_press': float(step_scores[1] - step_scores[0]),
        'pi_press_axis': float(
            pi_a[2] if env_name == 'sawyer_handle_press_side' else pi_a[1]),
    }

  for _ in range(int(args.episodes)):
    ts = environment.reset()
    steps = 0
    n_hov = 0
    n_suc = 0
    while not ts.last() and steps < 150:
      obs = np.asarray(ts.observation, dtype=np.float32)
      state_vec = obs[:obs_dim]
      action = actor.select_action(obs)
      lab = classify_state(state_vec, env_name)
      if lab == 'hover_contact' and n_hov < args.max_hover_per_ep:
        row = audit_state(state_vec, action, lab)
        hover_rows.append(row)
        hover_pi_task_scores.append(row['score_pi_task'])
        hover_action_stds_task.append(row['score_action_std_task'])
        hover_action_stds_hover_g.append(row['score_action_std_hover_g'])
        n_hov += 1
      elif lab == 'success' and n_suc < args.max_success_per_ep:
        success_rows.append(audit_state(state_vec, action, lab))
        n_suc += 1
      ts = environment.step(action)
      steps += 1

  # State sensitivity: std of π-scores across different hover states
  state_std_task = (
      float(np.std(hover_pi_task_scores)) if hover_pi_task_scores else float('nan'))
  action_std_task_mean = (
      float(np.mean(hover_action_stds_task)) if hover_action_stds_task
      else float('nan'))
  action_std_hover_g_mean = (
      float(np.mean(hover_action_stds_hover_g)) if hover_action_stds_hover_g
      else float('nan'))

  report = {
      'checkpoint': str(ckpt_path),
      'env_name': env_name,
      'env_steps': int(ckpt.get('env_steps', -1)),
      'episodes': int(args.episodes),
      'n_random_actions': int(args.n_random_actions),
      'n_hover_audits': len(hover_rows),
      'n_success_audits': len(success_rows),
      'hover_summary': summarize_gaps(hover_rows),
      'success_summary': summarize_gaps(success_rows),
      'sensitivity': {
          'hover_state_std_of_pi_score_task': state_std_task,
          'hover_mean_action_std_task': action_std_task_mean,
          'hover_mean_action_std_hover_g': action_std_hover_g_mean,
          'action_over_state_std_ratio_task': (
              action_std_task_mean / (state_std_task + 1e-12)
              if np.isfinite(state_std_task) else float('nan')),
      },
      'interpretation': {
          'likes_success_states_vs_action_advice': (
              'Compare success_summary.score_pi_task_mean to '
              'hover_summary.score_pi_task_mean (state ranking) vs '
              'hover_summary.gap_press_minus_pi_task_mean / '
              'grad_cos_press_dir_mean (action advice at same state).'),
          'action_insensitive_if': (
              'action_over_state_std_ratio_task << 1'),
          'her_goal_mismatch_if': (
              'gap under g_task > 0 but gap under hover_g <= 0'),
      },
  }
  print(json.dumps(report, indent=2, sort_keys=True))
  if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
  main()
