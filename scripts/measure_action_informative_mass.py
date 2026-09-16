#!/usr/bin/env python3
"""Extract critic action-sensitivity proxies from a decomposed DCC checkpoint.

Collects on-policy trajectories (a proxy for late-training replay occupancy;
Reverb buffers are not saved) and, at each state, scores counterfactual
actions under the fixed task goal. Optionally branches the simulator from
the same MuJoCo snapshot.

Writes JSON (summary) and NPZ (all plotted arrays). Does not train.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import action_ranking_diagnostics as ard  # noqa: E402
from contrastive import critic_phase_probe  # noqa: E402
from contrastive.action_informative_mass import (  # noqa: E402
    EPISODE_LEN,
    TASK_SPECS,
    biased_action,
    bootstrap_survival,
    contact_flag,
    default_eps_grid,
    family_range,
    make_candidate_actions,
    score_range,
    score_std,
    summarize_run,
    survival_curve,
    task_coordinate,
)
from contrastive.feature_shortcut import (  # noqa: E402
    categorical_accuracy,
    env_config_dir_matches,
    make_her_pairs,
)
from contrastive.her_future_phase import classify_her_goal  # noqa: E402


def _pad_state(state: np.ndarray, obs_dim: int) -> np.ndarray:
  state = np.asarray(state, dtype=np.float32).reshape(-1)
  if state.shape[0] < obs_dim:
    return np.pad(state, (0, obs_dim - state.shape[0])).astype(np.float32)
  return state[:obs_dim].astype(np.float32)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--env-name', required=True, choices=sorted(TASK_SPECS))
  parser.add_argument('--seed', type=int, default=6)
  parser.add_argument('--episodes', type=int, default=40)
  parser.add_argument('--n-gauss', type=int, default=8)
  parser.add_argument('--n-shuffle', type=int, default=8)
  parser.add_argument('--n-uniform', type=int, default=8)
  parser.add_argument('--env-branch-cap', type=int, default=64)
  parser.add_argument('--env-horizon', type=int, default=8)
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
    raise KeyError('Need a mid-task decomposed DCC checkpoint')
  if not env_config_dir_matches(ckpt_path, args.env_name):
    print(f'WARNING: path does not end with _env_{args.env_name}: {ckpt_path}',
          flush=True)

  env_name = args.env_name
  spec = TASK_SPECS[env_name]
  start_index = int(ckpt.get('goal_start_index', 0))
  end_index = int(ckpt.get('goal_end_index', -1))
  success_mode = ckpt.get('sawyer_success_mode', 'corrected')
  environment, obs_dim = contrastive_utils.make_environment(
      env_name, start_index, end_index, args.seed,
      fixed_start_end=spec['fixed_goal'],
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

  def _sample(observation, key):
    dist_params = networks.policy_network.apply(policy_params, observation)
    return networks.sample(dist_params, key)

  mode_jit = jax.jit(_mode_or_mean)
  sample_jit = jax.jit(_sample)

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

  rng = np.random.default_rng(args.seed)
  key = jax.random.PRNGKey(args.seed)

  if env_name in env_utils.TASK58_REACHABLE_SUCCESS_GOALS:
    task_goal = env_utils.TASK58_REACHABLE_SUCCESS_GOALS[env_name]
  else:
    ts0 = environment.reset()
    task_goal = np.asarray(ts0.observation, dtype=np.float32)[obs_dim:]

  def obs_from_states(states_arr, goal_vec):
    return np.stack([
        critic_phase_probe.build_task_goal_observation(
            s, goal_vec, obs_dim=obs_dim)
        for s in states_arr
    ], axis=0).astype(np.float32)

  episodes = []
  states = []
  actions_pi = []
  actions_mode = []
  t_fracs = []
  ep_ids = []
  for ep_i in range(int(args.episodes)):
    ts = environment.reset()
    ep_states = []
    ep_actions = []
    steps = 0
    while not ts.last() and steps < EPISODE_LEN:
      obs = np.asarray(ts.observation, dtype=np.float32)
      state_vec = _pad_state(obs[:obs_dim], obs_dim)
      key, sub = jax.random.split(key)
      a_stoch = np.asarray(sample_jit(obs, sub), dtype=np.float32)
      a_mode = np.asarray(mode_jit(obs), dtype=np.float32)
      states.append(state_vec)
      actions_pi.append(a_stoch)
      actions_mode.append(a_mode)
      t_fracs.append(steps / float(EPISODE_LEN))
      ep_ids.append(ep_i)
      ep_states.append(state_vec)
      ep_actions.append(a_stoch)
      ts = environment.step(a_stoch)
      steps += 1
    if ep_states:
      episodes.append({
          'states': np.stack(ep_states, axis=0),
          'actions': np.stack(ep_actions, axis=0),
      })

  states_np = np.stack(states, axis=0)
  actions_np = np.stack(actions_pi, axis=0)
  mode_np = np.stack(actions_mode, axis=0)
  t_frac_np = np.asarray(t_fracs, dtype=np.float64)
  ep_id_np = np.asarray(ep_ids, dtype=np.int32)
  n = states_np.shape[0]

  cand_rows = []
  fam_rows = []
  obs_rows = []
  for i in range(n):
    cands, families = make_candidate_actions(
        mode_np[i], actions_np, rng,
        n_gauss=args.n_gauss, n_shuffle=args.n_shuffle,
        n_uniform=args.n_uniform)
    obs_i = critic_phase_probe.build_task_goal_observation(
        states_np[i], task_goal, obs_dim=obs_dim)
    cand_rows.append(cands)
    fam_rows.append(families)
    obs_rows.append(np.repeat(obs_i[None, :], cands.shape[0], axis=0))
  n_cand = int(cand_rows[0].shape[0])
  obs_cat = np.concatenate(obs_rows, axis=0)
  act_cat = np.concatenate(cand_rows, axis=0)
  fam_cat = np.concatenate(fam_rows, axis=0)

  def _score_chunks(obs_all, act_all, chunk=2048):
    outs = []
    for start in range(0, obs_all.shape[0], chunk):
      outs.append(np.asarray(batch_paired_score(
          obs_all[start:start + chunk], act_all[start:start + chunk]),
          dtype=np.float64))
    return np.concatenate(outs, axis=0)

  scores_cat = _score_chunks(obs_cat, act_cat).reshape(n, n_cand)
  delta_task = np.array([score_range(row) for row in scores_cat])
  std_task = np.array([score_std(row) for row in scores_cat])
  delta_g010 = np.array([
      family_range(scores_cat[i], fam_cat[i * n_cand:(i + 1) * n_cand],
                   'gauss_0.10')
      for i in range(n)])
  delta_g025 = np.array([
      family_range(scores_cat[i], fam_cat[i * n_cand:(i + 1) * n_cand],
                   'gauss_0.25')
      for i in range(n)])
  f_pi = scores_cat[:, 0].copy()
  coord = np.array([task_coordinate(s, env_name) for s in states_np])
  contact = np.array([
      1.0 if contact_flag(s, env_name) else 0.0 for s in states_np])
  phase = np.array([classify_her_goal(s, env_name) for s in states_np])

  @jax.jit
  def grad_chunk(observations, actions):
    def one(obs, act):
      def _score(a):
        return decomp_nets.apply_paired_score(
            b_shared, h_phi, phi_task, psi, obs[None, :], a[None, :])[0]
      return jax.grad(_score)(act)
    return jax.vmap(one)(observations, actions)

  task_obs = obs_from_states(states_np, task_goal)
  grad_parts = []
  for start in range(0, n, 128):
    g = np.asarray(grad_chunk(
        task_obs[start:start + 128], mode_np[start:start + 128]),
        dtype=np.float32)
    grad_parts.append(np.linalg.norm(g, axis=-1))
  grad_norm = np.concatenate(grad_parts, axis=0)

  her_s, her_a, her_g = make_her_pairs(
      episodes, rng, max_pairs=256, discount=0.99)
  cat_acc = float('nan')
  shuf_acc = float('nan')
  if her_s.shape[0] >= 8:
    her_obs = np.concatenate([
        np.stack([_pad_state(s, obs_dim) for s in her_s], axis=0),
        np.stack([_pad_state(g, obs_dim) for g in her_g], axis=0),
    ], axis=1)
    logits = np.asarray(batch_score_matrix(her_obs, her_a))
    cat_acc = categorical_accuracy(logits)
    shuf_acc = categorical_accuracy(
        np.asarray(batch_score_matrix(
            her_obs, her_a[rng.permutation(her_a.shape[0])])))

  env_vals_1 = []
  env_vals_h = []
  env_error = ''
  try:
    ts = environment.reset()
    branched = 0
    steps = 0
    while branched < int(args.env_branch_cap) and steps < 8000:
      if ts.last():
        ts = environment.reset()
      obs = np.asarray(ts.observation, dtype=np.float32)
      state_vec = _pad_state(obs[:obs_dim], obs_dim)
      a_mode = np.asarray(mode_jit(obs), dtype=np.float32)
      do_branch = contact_flag(state_vec, env_name) or (steps % 12 == 0)
      if do_branch:
        snap = ard.snapshot_environment(environment)
        branch_actions = [
            a_mode,
            np.zeros(action_dim, dtype=np.float32),
            biased_action(a_mode, env_name),
        ]
        extra = rng.uniform(-1.0, 1.0, size=(3, action_dim)).astype(np.float32)
        branch_actions.extend(list(extra))
        coords_1 = []
        coords_h = []
        for act in branch_actions:
          ard.restore_environment(snap)
          ts1 = environment.step(act)
          s1 = _pad_state(
              np.asarray(ts1.observation, dtype=np.float32)[:obs_dim],
              obs_dim)
          coords_1.append(task_coordinate(s1, env_name))
          ts_h = ts1
          for _ in range(max(int(args.env_horizon) - 1, 0)):
            a_c = np.asarray(mode_jit(ts_h.observation), dtype=np.float32)
            ts_h = environment.step(a_c)
          s_h = _pad_state(
              np.asarray(ts_h.observation, dtype=np.float32)[:obs_dim],
              obs_dim)
          coords_h.append(task_coordinate(s_h, env_name))
        env_vals_1.append(score_range(np.asarray(coords_1)))
        env_vals_h.append(score_range(np.asarray(coords_h)))
        ard.restore_environment(snap)
        branched += 1
      ts = environment.step(a_mode)
      steps += 1
  except Exception as exc:  # noqa: BLE001
    env_error = f'{type(exc).__name__}: {exc}'

  env_delta_arr = np.asarray(env_vals_1, dtype=np.float64)
  env_delta_h_arr = np.asarray(env_vals_h, dtype=np.float64)

  eps_grid = default_eps_grid(
      max(np.nanmax(delta_task) if delta_task.size else 1.0, 1.0), n=61)
  surv, surv_lo, surv_hi = bootstrap_survival(delta_task, eps_grid, rng)
  surv010 = survival_curve(delta_g010, eps_grid)
  surv025 = survival_curve(delta_g025, eps_grid)

  npz = {
      'delta_f_task': delta_task,
      'delta_f_std': std_task,
      'delta_f_gauss_0.10': delta_g010,
      'delta_f_gauss_0.25': delta_g025,
      'grad_norm': grad_norm,
      'f_pi': f_pi,
      't_frac': t_frac_np,
      'episode_id': ep_id_np.astype(np.float64),
      'task_coord': coord,
      'contact': contact,
      'eps_grid': eps_grid,
      'survival': surv,
      'survival_lo': surv_lo,
      'survival_hi': surv_hi,
      'survival_gauss_0.10': surv010,
      'survival_gauss_0.25': surv025,
      'categorical_accuracy': np.array([cat_acc], dtype=np.float64),
      'action_shuffle_accuracy': np.array([shuf_acc], dtype=np.float64),
      'env_delta_taskcoord': env_delta_arr,
      'env_delta_horizon': env_delta_h_arr,
      'env_steps': np.array([int(ckpt.get('env_steps', -1))], dtype=np.float64),
      'seed': np.array([args.seed], dtype=np.float64),
  }

  summary = {
      'checkpoint': str(ckpt_path),
      'env_name': env_name,
      'task_index': int(spec['index']),
      'env_steps': int(ckpt.get('env_steps', -1)),
      'seed': int(args.seed),
      'episodes': int(args.episodes),
      'sawyer_success_mode': success_mode,
      'goal': 'task_goal',
      'occupancy': 'on_policy_stochastic_rollout',
      'delta_f_is': 'max_a f(s,a,g_task) - min_a f(s,a,g_task) over candidate families',
      'not_tv': True,
      'env_branch_error': env_error,
      'n_env_branch': float(env_delta_arr.size),
      **summarize_run(npz),
      'phase_mass': {
          name: float(np.mean(phase == name))
          for name in sorted(set(phase.tolist()))
      },
  }
  print(json.dumps(summary, indent=2, sort_keys=True))
  if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(out.with_suffix('.npz'), **npz)
    print(f'Wrote {out} and {out.with_suffix(".npz")}', flush=True)


if __name__ == '__main__':
  main()
