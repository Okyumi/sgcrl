#!/usr/bin/env python3
"""Measure stage dwell on successful trajectories + actor score at hover.

Reports, for successful 150-step episodes from a mid-task checkpoint:
  - steps spent in far / near_approach / hover_contact / object_progress / success
  - first hitting times and max consecutive success dwell
  - at hover states: critic score of π vs a press/push-biased action under g_task

Example:
  python scripts/measure_success_stage_dwell.py \\
    --checkpoint .../sawyer_handle_press_side/seed_6/task_0_step_100200.pkl \\
    --env-name sawyer_handle_press_side --episodes 60
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

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
  # push
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


def summarize(labels: List[str]) -> Dict[str, float]:
  n = max(len(labels), 1)
  stages = ('far', 'near_approach', 'hover_contact', 'object_progress', 'success')
  out = {f'steps_{s}': float(labels.count(s)) for s in stages}
  out['horizon'] = float(len(labels))
  for s in stages:
    out[f'frac_{s}'] = labels.count(s) / n
    out[f'first_{s}'] = float(labels.index(s) if s in labels else -1.0)
  max_run = run = 0
  for lab in labels:
    if lab == 'success':
      run += 1
      max_run = max(max_run, run)
    else:
      run = 0
  out['max_consecutive_success'] = float(max_run)
  # success onset → end occupancy
  if 'success' in labels:
    first = labels.index('success')
    out['steps_after_first_success'] = float(len(labels) - first)
    out['success_steps_after_onset'] = float(labels[first:].count('success'))
  else:
    out['steps_after_first_success'] = -1.0
    out['success_steps_after_onset'] = -1.0
  return out


def mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
  if not rows:
    return {}
  keys = rows[0].keys()
  return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def biased_action(pi_action: np.ndarray, env_name: str) -> np.ndarray:
  a = np.array(pi_action, dtype=np.float32, copy=True)
  if env_name == 'sawyer_handle_press_side':
    a[2] = -1.0  # press down
  else:
    a[1] = 1.0   # push toward +Y target
  return np.clip(a, -1.0, 1.0)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--env-name', required=True, choices=sorted(FIXED_GOALS))
  parser.add_argument('--seed', type=int, default=5)
  parser.add_argument('--episodes', type=int, default=60)
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

  actor = _PolicyActor(lambda obs: np.asarray(_mode_jit(obs)))

  if env_name in env_utils.TASK58_REACHABLE_SUCCESS_GOALS:
    task_goal = env_utils.TASK58_REACHABLE_SUCCESS_GOALS[env_name]
  else:
    # Push full-state goal: use goal half from a reset observation.
    ts0 = environment.reset()
    task_goal = np.asarray(ts0.observation, dtype=np.float32)[obs_dim:]

  success_rows = []
  fail_rows = []
  hover_gaps = []  # score(biased) - score(pi) under g_task at hover states

  for _ in range(int(args.episodes)):
    ts = environment.reset()
    labels = []
    reward_sum = 0.0
    steps = 0
    ep_hover_gaps = []
    while not ts.last() and steps < 150:
      obs = np.asarray(ts.observation, dtype=np.float32)
      state_vec = obs[:obs_dim]
      action = actor.select_action(obs)
      lab = classify_state(state_vec, env_name)
      labels.append(lab)
      if lab == 'hover_contact':
        task_obs = critic_phase_probe.build_task_goal_observation(
            state_vec, task_goal, obs_dim=obs_dim)
        cand = np.stack([action, biased_action(action, env_name)], axis=0)
        scores = np.asarray(paired_score(task_obs, cand), dtype=np.float32)
        ep_hover_gaps.append(float(scores[1] - scores[0]))
      ts = environment.step(action)
      reward_sum += float(ts.reward or 0.0)
      steps += 1
    row = summarize(labels)
    row['return'] = reward_sum
    if reward_sum > 0.5:
      success_rows.append(row)
      hover_gaps.extend(ep_hover_gaps)
    else:
      fail_rows.append(row)

  report = {
      'checkpoint': str(ckpt_path),
      'env_name': env_name,
      'env_steps': int(ckpt.get('env_steps', -1)),
      'episodes': int(args.episodes),
      'n_success': len(success_rows),
      'n_fail': len(fail_rows),
      'success_rate': len(success_rows) / max(args.episodes, 1),
      'success_stage_mean': mean_dict(success_rows),
      'fail_stage_mean': mean_dict(fail_rows),
      'hover_score_gap_biased_minus_pi_mean': (
          float(np.mean(hover_gaps)) if hover_gaps else float('nan')),
      'hover_score_gap_n': len(hover_gaps),
      'definitions': {
          'horizon': 150,
          'handle_success': '|z-0.07|<=0.02 and ||hand-handle||<=0.09',
          'push_success': '||obj-target||<=0.05',
          'push_progress': '0.05<||obj-target||<=0.15',
          'hover_contact': '||hand-obj||<=0.09 and not success/(progress)',
          'near_approach': '0.09<||hand-obj||<=0.15',
          'far': '||hand-obj||>0.15',
      },
  }
  print(json.dumps(report, indent=2, sort_keys=True))
  if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
  main()
