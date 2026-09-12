#!/usr/bin/env python3
"""Pin down which observation coordinates the contrastive critic uses.

Offline over a mid-task decomposed checkpoint. No training.

Three complementary measurements, all on the same rollouts:

1. Synthetic feature transplant at hover under g_task.
   Copy one coordinate of s onto its success value, keep a_π and g_task.
   Recovery fraction vs copying the full success state.

2. InfoNCE retrieval after shuffling one block of s, a, or g.
   HER (s_t, a_t, s_{t+k}) pairs from the same rollouts. A large accuracy
   drop identifies the retrieval feature.

3. One-step mechanism displacement at a frozen hover state.
   How much press / π / random actually move handle z vs cube xy.
   This is the empirical I(G; A | S_hover) proxy.

Also logs Pearson(score, coordinate) and the residual correlation of score
with the press/push axis after regressing out the candidate shortcut.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.feature_shortcut import (  # noqa: E402
    FEATURE_BLOCKS,
    FIXED_GOALS,
    PUSH_TARGET,
    biased_action,
    categorical_accuracy,
    classify_state,
    env_config_dir_matches,
    make_her_pairs,
    mechanism_density,
    occupancy_stats,
    pearson,
    recovery_fraction,
    residualize,
    shuffle_block,
    shuffle_rows,
    summarize,
    synthetic_progress_state,
    transplant_indices,
)
from contrastive import action_ranking_diagnostics as ard  # noqa: E402
from contrastive import critic_phase_probe  # noqa: E402


class _PolicyActor:
  def __init__(self, apply_mode):
    self._apply_mode = apply_mode

  def select_action(self, observation):
    return np.asarray(self._apply_mode(observation), dtype=np.float32)


def _pad_state(state: np.ndarray, obs_dim: int) -> np.ndarray:
  state = np.asarray(state, dtype=np.float32).reshape(-1)
  if state.shape[0] < obs_dim:
    return np.pad(state, (0, obs_dim - state.shape[0])).astype(np.float32)
  return state[:obs_dim].astype(np.float32)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--env-name', required=True, choices=sorted(FIXED_GOALS))
  parser.add_argument('--seed', type=int, default=6)
  parser.add_argument('--episodes', type=int, default=40)
  parser.add_argument('--max-hover-per-ep', type=int, default=6)
  parser.add_argument('--max-success-per-ep', type=int, default=4)
  parser.add_argument('--her-pairs', type=int, default=256)
  parser.add_argument('--one-step-hovers', type=int, default=24)
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
  from contrastive import utils as contrastive_utils
  from contrastive.decomposed_networks import make_decomposed_networks
  import env_utils

  ckpt_path = Path(args.checkpoint).expanduser().resolve()
  with ckpt_path.open('rb') as handle:
    ckpt = pickle.load(handle)
  if 'decomposed_training_state' not in ckpt:
    raise KeyError('Need mid-task decomposed checkpoint')
  if not env_config_dir_matches(ckpt_path, args.env_name):
    print(
        f'WARNING: checkpoint path does not end with _env_{args.env_name}: '
        f'{ckpt_path}',
        flush=True)

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
  def batch_paired_score(observations, actions):
    return decomp_nets.apply_paired_score(
        b_shared, h_phi, phi_task, psi,
        jnp.asarray(observations), jnp.asarray(actions))

  @jax.jit
  def batch_score_matrix(observations, actions):
    return decomp_nets.apply_score(
        b_shared, h_phi, phi_task, psi,
        jnp.asarray(observations), jnp.asarray(actions))

  actor = _PolicyActor(lambda obs: np.asarray(_mode_jit(obs)))
  rng = np.random.default_rng(args.seed)

  if env_name in env_utils.TASK58_REACHABLE_SUCCESS_GOALS:
    task_goal = env_utils.TASK58_REACHABLE_SUCCESS_GOALS[env_name]
  else:
    ts0 = environment.reset()
    task_goal = np.asarray(ts0.observation, dtype=np.float32)[obs_dim:]

  def obs_from_states(states_arr):
    return np.stack([
        critic_phase_probe.build_task_goal_observation(
            s, task_goal, obs_dim=obs_dim)
        for s in states_arr
    ], axis=0).astype(np.float32)

  def score_batch(states_arr, actions_arr):
    if len(states_arr) == 0:
      return np.zeros((0,), dtype=np.float32)
    return np.asarray(
        batch_paired_score(obs_from_states(states_arr), actions_arr),
        dtype=np.float32)

  episodes = []
  hover_states: List[np.ndarray] = []
  hover_actions: List[np.ndarray] = []
  success_states: List[np.ndarray] = []
  success_actions: List[np.ndarray] = []
  all_states: List[np.ndarray] = []
  all_actions: List[np.ndarray] = []
  all_labels: List[str] = []

  for _ in range(int(args.episodes)):
    ts = environment.reset()
    ep_states = []
    ep_actions = []
    steps = 0
    n_hov = 0
    n_suc = 0
    while not ts.last() and steps < 150:
      obs = np.asarray(ts.observation, dtype=np.float32)
      state_vec = _pad_state(obs[:obs_dim], obs_dim)
      action = actor.select_action(obs)
      lab = classify_state(state_vec, env_name)
      ep_states.append(state_vec)
      ep_actions.append(np.asarray(action, dtype=np.float32))
      all_states.append(state_vec)
      all_actions.append(np.asarray(action, dtype=np.float32))
      all_labels.append(lab)
      if lab == 'hover_contact' and n_hov < args.max_hover_per_ep:
        hover_states.append(state_vec)
        hover_actions.append(np.asarray(action, dtype=np.float32))
        n_hov += 1
      elif lab == 'success' and n_suc < args.max_success_per_ep:
        success_states.append(state_vec)
        success_actions.append(np.asarray(action, dtype=np.float32))
        n_suc += 1
      ts = environment.step(action)
      steps += 1
    if ep_states:
      episodes.append({
          'states': np.stack(ep_states, axis=0),
          'actions': np.stack(ep_actions, axis=0),
      })

  hover_states_np = (
      np.stack(hover_states, axis=0) if hover_states
      else np.zeros((0, obs_dim), dtype=np.float32))
  hover_actions_np = (
      np.stack(hover_actions, axis=0) if hover_actions
      else np.zeros((0, action_dim), dtype=np.float32))
  success_states_np = (
      np.stack(success_states, axis=0) if success_states
      else np.zeros((0, obs_dim), dtype=np.float32))
  all_states_np = (
      np.stack(all_states, axis=0) if all_states
      else np.zeros((0, obs_dim), dtype=np.float32))
  all_actions_np = (
      np.stack(all_actions, axis=0) if all_actions
      else np.zeros((0, action_dim), dtype=np.float32))
  all_scores_np = np.asarray(
      score_batch(all_states_np, all_actions_np), dtype=np.float64)
  donor = (
      np.mean(success_states_np, axis=0) if success_states_np.shape[0]
      else synthetic_progress_state(
          hover_states_np[0] if hover_states_np.shape[0]
          else np.zeros(obs_dim, dtype=np.float32),
          env_name))

  # --- 1. transplants at hover --------------------------------------------
  transplant_rows: Dict[str, List[float]] = {}
  recovery_rows: Dict[str, List[float]] = {}
  if hover_states_np.shape[0]:
    n_h = hover_states_np.shape[0]
    donor_rep = np.repeat(donor[None, :], n_h, axis=0)
    treatments = {
        'base': (hover_states_np, hover_actions_np),
        'full_success_state': (donor_rep, hover_actions_np),
        'synthetic_progress': (
            np.stack([synthetic_progress_state(s, env_name)
                      for s in hover_states_np], axis=0),
            hover_actions_np),
        'donor_mech_z': (
            np.stack([transplant_indices(s, donor, slice(6, 7))
                      for s in hover_states_np], axis=0),
            hover_actions_np),
        'donor_mech_xy': (
            np.stack([transplant_indices(s, donor, slice(4, 6))
                      for s in hover_states_np], axis=0),
            hover_actions_np),
        'donor_mech': (
            np.stack([transplant_indices(s, donor, slice(4, 7))
                      for s in hover_states_np], axis=0),
            hover_actions_np),
        'donor_hand': (
            np.stack([transplant_indices(s, donor, slice(0, 3))
                      for s in hover_states_np], axis=0),
            hover_actions_np),
        'donor_gripper': (
            np.stack([transplant_indices(s, donor, slice(3, 4))
                      for s in hover_states_np], axis=0),
            hover_actions_np),
        'press_action_only': (
            hover_states_np,
            np.stack([biased_action(a, env_name) for a in hover_actions_np],
                     axis=0)),
    }
    scored = {
        name: np.asarray(score_batch(states, acts), dtype=np.float64)
        for name, (states, acts) in treatments.items()
    }
    transplant_rows = {name: vals.tolist() for name, vals in scored.items()}
    base_vals = scored['base']
    full_vals = scored['full_success_state']
    for name, vals in scored.items():
      if name in ('base', 'full_success_state'):
        continue
      recovery_rows[name] = [
          recovery_fraction(b, v, f)
          for b, v, f in zip(base_vals, vals, full_vals)
      ]

  transplant_summary = {
      name: summarize(vals) for name, vals in transplant_rows.items()}
  recovery_summary = {
      name: summarize(vals) for name, vals in recovery_rows.items()}

  # --- 2. HER retrieval shuffles ------------------------------------------
  her_s, her_a, her_g = make_her_pairs(
      episodes, rng, max_pairs=int(args.her_pairs))
  retrieval = {
      'n_pairs': float(her_s.shape[0]),
      'baseline_accuracy': float('nan'),
  }
  if her_s.shape[0] >= 8:
    her_obs = np.concatenate([
        np.stack([_pad_state(s, obs_dim) for s in her_s], axis=0),
        np.stack([_pad_state(g, obs_dim) for g in her_g], axis=0),
    ], axis=1)
    base_logits = np.asarray(batch_score_matrix(her_obs, her_a))
    retrieval['baseline_accuracy'] = categorical_accuracy(base_logits)

    def _acc_from_obs(obs_mod, act_mod):
      logits = np.asarray(batch_score_matrix(obs_mod, act_mod))
      acc = categorical_accuracy(logits)
      return acc, retrieval['baseline_accuracy'] - acc

    state_part = her_obs[:, :obs_dim].copy()
    goal_part = her_obs[:, obs_dim:].copy()
    for name, slc in FEATURE_BLOCKS.items():
      sh_s = shuffle_block(state_part, slc, rng)
      obs_s = np.concatenate([sh_s, goal_part], axis=1)
      acc, drop = _acc_from_obs(obs_s, her_a)
      retrieval[f'state_{name}_accuracy'] = acc
      retrieval[f'state_{name}_drop'] = drop

      sh_g = shuffle_block(goal_part, slc, rng)
      obs_g = np.concatenate([state_part, sh_g], axis=1)
      acc, drop = _acc_from_obs(obs_g, her_a)
      retrieval[f'goal_{name}_accuracy'] = acc
      retrieval[f'goal_{name}_drop'] = drop

    sh_a = shuffle_rows(her_a, rng)
    acc, drop = _acc_from_obs(her_obs, sh_a)
    retrieval['action_accuracy'] = acc
    retrieval['action_drop'] = drop
    zero_a = np.zeros_like(her_a)
    acc, drop = _acc_from_obs(her_obs, zero_a)
    retrieval['zero_action_accuracy'] = acc
    retrieval['zero_action_drop'] = drop

  # --- 3. one-step mechanism movement at frozen hover ---------------------
  one_step = {
      'n': 0.0,
      'pi': {},
      'press': {},
      'random': {},
  }
  n_one = 0
  deltas = {'pi': [], 'press': [], 'random': []}
  try:
    ts = environment.reset()
    steps = 0
    while n_one < int(args.one_step_hovers) and steps < 8000:
      if ts.last():
        ts = environment.reset()
      obs = np.asarray(ts.observation, dtype=np.float32)
      state_vec = _pad_state(obs[:obs_dim], obs_dim)
      lab = classify_state(state_vec, env_name)
      if lab != 'hover_contact':
        action = actor.select_action(obs)
        ts = environment.step(action)
        steps += 1
        continue
      snap = ard.snapshot_environment(environment)
      a_pi = np.asarray(actor.select_action(obs), dtype=np.float32)
      candidates = {
          'pi': a_pi,
          'press': biased_action(a_pi, env_name),
          'random': rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32),
      }
      for name, act in candidates.items():
        ard.restore_environment(snap)
        ts_next = environment.step(act)
        nxt = _pad_state(
            np.asarray(ts_next.observation, dtype=np.float32)[:obs_dim],
            obs_dim)
        deltas[name].append({
            'd_mech_z': float(abs(nxt[6] - state_vec[6])),
            'd_mech_xy': float(np.linalg.norm(nxt[4:6] - state_vec[4:6])),
            'd_mech': float(np.linalg.norm(nxt[4:7] - state_vec[4:7])),
            'd_hand': float(np.linalg.norm(nxt[:3] - state_vec[:3])),
        })
      ard.restore_environment(snap)
      ts = environment.step(a_pi)
      n_one += 1
      steps += 1
  except Exception as exc:  # noqa: BLE001
    one_step['error'] = f'{type(exc).__name__}: {exc}'

  one_step['n'] = float(n_one)
  for name, rows in deltas.items():
    if not rows:
      continue
    one_step[name] = {
        key: summarize([row[key] for row in rows])
        for key in rows[0].keys()
    }

  # --- correlations -------------------------------------------------------
  press_axis = (
      all_actions_np[:, 2] if env_name == 'sawyer_handle_press_side'
      else all_actions_np[:, 1])
  corr = {}
  if all_states_np.shape[0] >= 8:
    corr['score_vs_hand_x'] = pearson(all_scores_np, all_states_np[:, 0])
    corr['score_vs_hand_y'] = pearson(all_scores_np, all_states_np[:, 1])
    corr['score_vs_hand_z'] = pearson(all_scores_np, all_states_np[:, 2])
    corr['score_vs_gripper'] = pearson(all_scores_np, all_states_np[:, 3])
    corr['score_vs_mech_x'] = pearson(all_scores_np, all_states_np[:, 4])
    corr['score_vs_mech_y'] = pearson(all_scores_np, all_states_np[:, 5])
    corr['score_vs_mech_z'] = pearson(all_scores_np, all_states_np[:, 6])
    corr['score_vs_press_axis'] = pearson(all_scores_np, press_axis)
    hand_obj = np.linalg.norm(
        all_states_np[:, :3] - all_states_np[:, 4:7], axis=-1)
    corr['score_vs_hand_obj_dist'] = pearson(all_scores_np, hand_obj)
    if env_name == 'sawyer_push':
      obj_dist = np.linalg.norm(
          all_states_np[:, 4:7] - PUSH_TARGET[None, :], axis=-1)
      corr['score_vs_obj_target_dist'] = pearson(all_scores_np, obj_dist)
      shortcut = obj_dist
    else:
      shortcut = all_states_np[:, 6]
    residual_score = residualize(all_scores_np, shortcut)
    corr['score_vs_press_axis_after_shortcut'] = pearson(
        residual_score, residualize(press_axis, shortcut))

  hover_mask = np.array([lab == 'hover_contact' for lab in all_labels])
  hover_corr = {}
  if np.count_nonzero(hover_mask) >= 8:
    hs = all_scores_np[hover_mask]
    hz = all_states_np[hover_mask]
    ha = press_axis[hover_mask]
    hover_corr['score_vs_mech_z'] = pearson(hs, hz[:, 6])
    hover_corr['score_vs_mech_x'] = pearson(hs, hz[:, 4])
    hover_corr['score_vs_mech_y'] = pearson(hs, hz[:, 5])
    hover_corr['score_vs_press_axis'] = pearson(hs, ha)
    if env_name == 'sawyer_push':
      od = np.linalg.norm(hz[:, 4:7] - PUSH_TARGET[None, :], axis=-1)
      hover_corr['score_vs_obj_target_dist'] = pearson(hs, od)

  z_vals = all_states_np[:, 6] if all_states_np.size else np.array([])
  density = {
      'all_states': mechanism_density(all_states_np, env_name),
      'hover_states': mechanism_density(hover_states_np, env_name),
      'success_states': mechanism_density(success_states_np, env_name),
      'handle_z_occupancy': occupancy_stats(z_vals, 0.05, 0.09),
  }

  predicted = (
      'handle: synthetic_progress / donor_mech_z recovery ~ 1, action drop ~ 0, '
      'one-step d_mech_z(press) small relative to scene gap; '
      'push: donor_mech_xy / synthetic_progress recovery high AND action drop '
      'or one-step d_mech_xy(press) large.'
      if env_name == 'sawyer_handle_press_side' else
      'push: object xy (mech_xy / obj-target dist) is the retrieval feature; '
      'action still moves that feature at hover, so InfoNCE keeps ∂φ/∂a.')

  report = {
      'checkpoint': str(ckpt_path),
      'env_name': env_name,
      'env_steps': int(ckpt.get('env_steps', -1)),
      'episodes': int(args.episodes),
      'n_hover': float(hover_states_np.shape[0]),
      'n_success': float(success_states_np.shape[0]),
      'n_transitions': float(all_states_np.shape[0]),
      'transplant_scores': transplant_summary,
      'transplant_recovery': recovery_summary,
      'retrieval': retrieval,
      'one_step_mechanism_delta': one_step,
      'correlation': corr,
      'hover_correlation': hover_corr,
      'density': density,
      'predicted_signature': predicted,
  }
  print(json.dumps(report, indent=2, sort_keys=True))
  if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {out}', flush=True)


if __name__ == '__main__':
  main()
