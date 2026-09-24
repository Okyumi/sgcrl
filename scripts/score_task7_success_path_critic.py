#!/usr/bin/env python3
"""Score Task-7 Success-BC buffers vs plain-DCC success rollouts.

The first 10% terminal-BC eval on shelf_place stores five full 150-step
episodes in D_succ (750 transitions). This script asks whether those
early cloned paths have lower φ(s,a)ᵀψ(g) than success paths from a
policy that actually solved the task.

Raw inner products are only compared under one critic. Cosine of the
same (φ, ψ) pair is logged as a scale-free check.
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

TERM_DIR = Path(
    '/scratch/yd2247/sgcrl/logs/terminal_bc_task47/checkpoints/'
    'actor_reset_critic_decomposed_tid_False_heads_True'
    '_success_native_info_dyn0.000_pt256x4'
    '_bridge_4c3aef3792be_ret_7d91520b6c16')
PAPER_PLAIN = Path(
    '/scratch/yd2247/sgcrl/logs/paper_dcc_success_bc_jubail_checkpoints/10seed/'
    'actor_reset_critic_decomposed_tid_False_heads_True'
    '_success_native_info_dyn0.000_pt256x4')
SHELF_TARGET = np.array([0.02, 0.89, 0.30], dtype=np.float32)
SUCCESS_RADIUS = 0.07


def _term_ckpt(seed: int, step: int) -> Path:
  return TERM_DIR / f'seed_{seed}' / f'task_7_step_{step}.pkl'


def _paper_plain(seed: int) -> Path:
  return PAPER_PLAIN / f'seed_{seed}' / 'task_7.pkl'


def load_pickle(path: Path) -> dict:
  with path.open('rb') as handle:
    return pickle.load(handle)


def buffer_from_ckpt(ckpt: dict):
  state = ckpt['decomposed_training_state']
  size = int(np.asarray(state.success_buffer_size))
  obs = np.asarray(state.success_buffer_observation[:size], dtype=np.float32)
  action = np.asarray(state.success_buffer_action[:size], dtype=np.float32)
  return obs, action


def object_distance(obs: np.ndarray) -> np.ndarray:
  obj = obs[:, 4:7]
  return np.linalg.norm(obj - SHELF_TARGET[None, :], axis=1)


def summarize(scores: np.ndarray, dist: np.ndarray) -> dict:
  scores = np.asarray(scores, dtype=np.float64)
  dist = np.asarray(dist, dtype=np.float64)
  in_goal = dist <= SUCCESS_RADIUS
  far = dist > 0.15
  out = {
      'n': int(scores.size),
      'mean': float(np.mean(scores)) if scores.size else float('nan'),
      'std': float(np.std(scores)) if scores.size else float('nan'),
      'p10': float(np.percentile(scores, 10)) if scores.size else float('nan'),
      'p50': float(np.percentile(scores, 50)) if scores.size else float('nan'),
      'p90': float(np.percentile(scores, 90)) if scores.size else float('nan'),
      'frac_in_goal': float(np.mean(in_goal)) if scores.size else float('nan'),
  }
  for name, mask in (('in_goal', in_goal), ('prefix_far', far)):
    vals = scores[mask]
    out[f'{name}_n'] = int(vals.size)
    out[f'{name}_mean'] = float(np.mean(vals)) if vals.size else float('nan')
  return out


def dummy_spec(obs_dim=11, action_dim=4):
  from acme import specs
  from dm_env import specs as dm_specs
  return specs.EnvironmentSpec(
      observations=dm_specs.Array(shape=(2 * obs_dim,), dtype=np.float32),
      actions=dm_specs.BoundedArray(
          shape=(action_dim,), dtype=np.float32, minimum=-1.0, maximum=1.0),
      rewards=dm_specs.Array(shape=(), dtype=np.float32),
      discounts=dm_specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0.0, maximum=1.0),
  )


def make_critic_scorer(ckpt: dict):
  import jax
  import jax.numpy as jnp
  from contrastive.decomposed_networks import make_decomposed_networks

  state = ckpt['decomposed_training_state']
  obs_dim = int(ckpt.get('obs_dim', 11))
  decomp = make_decomposed_networks(
      dummy_spec(obs_dim),
      obs_dim=obs_dim,
      repr_dim=64,
      hidden_layer_sizes=(1024, 1024),
      use_residual=True,
      network_width=1024,
      critic_depth=4,
      phi_task_width=256,
      phi_task_depth=4,
      combine_mode='add',
      goal_encoder_mode='shared',
  )
  b_shared = ckpt['decomposed_b_shared_params']
  h_phi = ckpt['decomposed_h_phi_params']
  psi = ckpt['decomposed_psi_params']
  phi_task = state.phi_task_params

  @jax.jit
  def _score(obs, action):
    inner = decomp.apply_paired_score(
        b_shared, h_phi, phi_task, psi, obs, action)
    sa = decomp.apply_sa_repr(b_shared, h_phi, phi_task, obs, action)
    g = decomp.apply_goal_for_score(decomp.apply_psi(psi, obs))
    sa_n = sa / jnp.maximum(jnp.linalg.norm(sa, axis=1, keepdims=True), 1e-8)
    g_n = g / jnp.maximum(jnp.linalg.norm(g, axis=1, keepdims=True), 1e-8)
    cosine = jnp.sum(sa_n * g_n, axis=1)
    return inner, cosine

  def score_batch(obs, action, batch=256):
    inners = []
    cosines = []
    for start in range(0, obs.shape[0], batch):
      inner, cosine = _score(
          jnp.asarray(obs[start:start + batch]),
          jnp.asarray(action[start:start + batch]))
      inners.append(np.asarray(inner))
      cosines.append(np.asarray(cosine))
    return np.concatenate(inners), np.concatenate(cosines)

  return score_batch


def make_actor(ckpt: dict, seed: int):
  import jax
  from acme import specs
  import contrastive
  from contrastive import utils as contrastive_utils

  policy_params = ckpt.get('decomposed_policy_params') or ckpt.get(
      'composed_policy')
  if policy_params is None:
    raise KeyError('checkpoint has no policy')
  env_name = 'sawyer_shelf_place'
  start_index = int(ckpt.get('goal_start_index', 0))
  end_index = int(ckpt.get('goal_end_index', -1))
  success_mode = ckpt.get('sawyer_success_mode', 'native_info')
  environment, obs_dim = contrastive_utils.make_environment(
      env_name, start_index, end_index, seed,
      fixed_start_end=SHELF_TARGET,
      sawyer_success_mode=success_mode)
  obs_dim = int(ckpt.get('obs_dim') or obs_dim)
  env_spec = specs.make_environment_spec(environment)
  networks = contrastive.make_networks(
      env_spec, obs_dim=obs_dim,
      hidden_layer_sizes=(1024, 1024),
      use_residual=True, network_width=1024,
      critic_depth=4, actor_depth=4)

  def _mode(observation):
    dist_params = networks.policy_network.apply(policy_params, observation)
    if hasattr(dist_params, 'mode'):
      return dist_params.mode()
    return dist_params.mean()

  mode_jit = jax.jit(_mode)

  def select_action(observation):
    return np.asarray(mode_jit(observation), dtype=np.float32)

  return environment, select_action, int(obs_dim)


def rollout_successes(environment, select_action, obs_dim, episodes, seed):
  obs_list = []
  act_list = []
  success_episodes = 0
  for _ in range(episodes):
    timestep = environment.reset()
    ep_obs = []
    ep_act = []
    ep_rew = []
    while not timestep.last():
      obs = np.asarray(timestep.observation, dtype=np.float32)
      action = select_action(obs)
      ep_obs.append(obs)
      ep_act.append(action)
      timestep = environment.step(action)
      reward = float(np.asarray(timestep.reward))
      ep_rew.append(reward)
    if any(r > 0.0 for r in ep_rew):
      success_episodes += 1
      obs_list.append(np.stack(ep_obs, axis=0))
      act_list.append(np.stack(ep_act, axis=0))
  n_trans = int(sum(x.shape[0] for x in obs_list)) if obs_list else 0
  if not obs_list:
    empty = np.zeros((0, 2 * obs_dim), dtype=np.float32)
    return empty, np.zeros((0, 4), dtype=np.float32), {
        'episodes': episodes,
        'success_episodes': 0,
        'success_rate': 0.0,
        'n_transitions': 0,
    }
  return np.concatenate(obs_list, axis=0), np.concatenate(act_list, axis=0), {
      'episodes': episodes,
      'success_episodes': success_episodes,
      'success_rate': success_episodes / episodes,
      'n_transitions': n_trans,
  }


def pack(name, inner, cosine, obs, extra=None):
  dist = object_distance(obs) if obs.shape[0] else np.zeros((0,), np.float32)
  payload = {
      'name': name,
      'inner': summarize(inner, dist),
      'cosine': summarize(cosine, dist),
  }
  if extra:
    payload.update(extra)
  return payload


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--episodes', type=int, default=20)
  parser.add_argument('--skip-rollouts', action='store_true')
  parser.add_argument(
      '--output',
      default='/scratch/yd2247/sgcrl/logs/task7_critic_score_gate/scores.json')
  args = parser.parse_args()

  print('Loading critics and buffers...', flush=True)
  early6 = load_pickle(_term_ckpt(6, 1_500_750))
  early10 = load_pickle(_term_ckpt(10, 1_100_550))
  late10 = load_pickle(_term_ckpt(10, 7_903_950))
  late6 = load_pickle(_term_ckpt(6, 7_903_950))

  buffers = {
      'seed6_first5': buffer_from_ckpt(early6),
      'seed10_first5': buffer_from_ckpt(early10),
      'seed10_late': buffer_from_ckpt(late10),
      'seed6_late': buffer_from_ckpt(late6),
  }
  print('Building seed-10 late critic (good Task-7 run)...', flush=True)
  score_good = make_critic_scorer(late10)
  print('Building seed-6 first-success critic (lock-in time)...', flush=True)
  score_early6 = make_critic_scorer(early6)

  results = []
  for critic_name, scorer in (
      ('seed10_late_critic', score_good),
      ('seed6_early_critic', score_early6),
  ):
    for buf_name, (obs, action) in buffers.items():
      print(f'Scoring {buf_name} with {critic_name} n={obs.shape[0]}',
            flush=True)
      inner, cosine = scorer(obs, action)
      results.append(pack(
          f'{buf_name} | {critic_name}', inner, cosine, obs,
          extra={'source': 'success_buffer', 'buffer': buf_name,
                 'critic': critic_name}))

  if not args.skip_rollouts:
    print('Rolling out paper DCC seed 6 and terminal first-success actors...',
          flush=True)
    plain6 = load_pickle(_paper_plain(6))
    env, actor, obs_dim = make_actor(plain6, seed=6)
    dcc_obs, dcc_act, dcc_meta = rollout_successes(
        env, actor, obs_dim, args.episodes, seed=6)
    print('DCC seed6', dcc_meta, flush=True)
    env_e, actor_e, obs_dim_e = make_actor(early6, seed=106)
    early_obs, early_act, early_meta = rollout_successes(
        env_e, actor_e, obs_dim_e, args.episodes, seed=106)
    print('terminal seed6 @1.5M', early_meta, flush=True)
    for critic_name, scorer in (
        ('seed10_late_critic', score_good),
        ('seed6_early_critic', score_early6),
    ):
      if dcc_obs.shape[0]:
        inner, cosine = scorer(dcc_obs, dcc_act)
        results.append(pack(
            f'plain_dcc_seed6_success_rollouts | {critic_name}',
            inner, cosine, dcc_obs,
            extra={'source': 'rollout', 'actor': 'plain_dcc_seed6',
                   'critic': critic_name, **dcc_meta}))
      if early_obs.shape[0]:
        inner, cosine = scorer(early_obs, early_act)
        results.append(pack(
            f'terminal_seed6_1.5M_success_rollouts | {critic_name}',
            inner, cosine, early_obs,
            extra={'source': 'rollout', 'actor': 'terminal_seed6_1.5M',
                   'critic': critic_name, **early_meta}))

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
  print(f'Wrote {output}', flush=True)
  for row in results:
    inn = row['inner']
    cos = row['cosine']
    print(
        f"{row['name']}: inner {inn['mean']:.4f} "
        f"(in-goal {inn['in_goal_mean']:.4f}, far {inn['prefix_far_mean']:.4f}, "
        f"n={inn['n']}, in-goal n={inn['in_goal_n']}) "
        f"cosine {cos['mean']:.4f}",
        flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
