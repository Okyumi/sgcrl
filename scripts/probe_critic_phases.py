#!/usr/bin/env python3
"""Offline critic phase probe from a mid-task Task-5/8 checkpoint.

Loads ``task_{id}_step_{N}.pkl`` written by ``mid_task_checkpoint_every``,
scores success/hover/mid-reach transitions against the fixed task goal, and
writes JSON metrics matching the online ``probe/*`` W&B keys.

Example:
  python scripts/probe_critic_phases.py \\
    --checkpoint .../task_0_step_100000.pkl \\
    --env-name sawyer_handle_press_side \\
    --episodes 10 \\
    --output /tmp/probe_100k.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


FIXED_GOALS = {
    'sawyer_handle_press_side': np.array([-0.07, 0.68, 0.07], dtype=np.float32),
    'sawyer_window_close': np.array([0.0, 0.80, 0.2], dtype=np.float32),
}


class _PolicyActor:
  """Minimal deterministic actor with a ``select_action`` method."""

  def __init__(self, apply_mode):
    self._apply_mode = apply_mode

  def select_action(self, observation):
    return np.asarray(self._apply_mode(observation), dtype=np.float32)


def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--env-name', required=True,
                      choices=sorted(FIXED_GOALS))
  parser.add_argument('--seed', type=int, default=5)
  parser.add_argument('--episodes', type=int, default=10)
  parser.add_argument('--output', default='')
  parser.add_argument('--interaction-threshold', type=float, default=0.09)
  parser.add_argument('--mid-reach-threshold', type=float, default=0.15)
  parser.add_argument('--network-width', type=int, default=1024)
  parser.add_argument('--critic-depth', type=int, default=4)
  parser.add_argument('--actor-depth', type=int, default=4)
  parser.add_argument('--phi-task-width', type=int, default=256)
  parser.add_argument('--phi-task-depth', type=int, default=4)
  return parser.parse_args()


def main():
  args = _parse_args()
  import jax
  import jax.numpy as jnp
  from acme import specs

  import contrastive
  from contrastive import critic_phase_probe
  from contrastive import utils as contrastive_utils
  from contrastive.decomposed_networks import make_decomposed_networks
  import env_utils

  checkpoint_path = Path(args.checkpoint).expanduser().resolve()
  with checkpoint_path.open('rb') as handle:
    ckpt = pickle.load(handle)
  if not isinstance(ckpt, Mapping):
    raise TypeError(f'Expected mapping checkpoint, got {type(ckpt)}')
  if 'decomposed_training_state' not in ckpt:
    raise KeyError(
        'Checkpoint missing decomposed_training_state. Save mid-task '
        'snapshots with mid_task_checkpoint_every > 0.')

  env_name = args.env_name
  start_index = int(ckpt.get('goal_start_index', 0))
  end_index = int(ckpt.get('goal_end_index', -1))
  success_mode = ckpt.get('sawyer_success_mode', 'corrected')
  environment, obs_dim = contrastive_utils.make_environment(
      env_name, start_index, end_index, args.seed,
      fixed_start_end=FIXED_GOALS[env_name],
      task_id=None, num_tasks=None,
      sawyer_success_mode=success_mode)
  obs_dim = int(ckpt.get('obs_dim') or obs_dim)

  env_spec = specs.make_environment_spec(environment)
  networks = contrastive.make_networks(
      env_spec,
      obs_dim=obs_dim,
      hidden_layer_sizes=(args.network_width, args.network_width),
      use_residual=True,
      network_width=args.network_width,
      critic_depth=args.critic_depth,
      actor_depth=args.actor_depth,
  )
  decomp_nets = make_decomposed_networks(
      env_spec,
      obs_dim=obs_dim,
      repr_dim=64,
      hidden_layer_sizes=(args.network_width, args.network_width),
      use_residual=True,
      network_width=args.network_width,
      critic_depth=args.critic_depth,
      phi_task_width=args.phi_task_width,
      phi_task_depth=args.phi_task_depth,
      combine_mode='add',
      goal_encoder_mode='shared',
  )
  policy_network = networks.policy_network

  state = ckpt['decomposed_training_state']
  policy_params = ckpt.get('decomposed_policy_params') or ckpt.get(
      'composed_policy')
  b_shared = ckpt['decomposed_b_shared_params']
  h_phi = ckpt['decomposed_h_phi_params']
  psi = ckpt['decomposed_psi_params']
  phi_task = state.phi_task_params

  def _mode_or_mean(observation):
    dist_params = policy_network.apply(policy_params, observation)
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

  def score_fn(observation, actions):
    return np.asarray(paired_score(observation, actions))

  metrics = critic_phase_probe.run_critic_phase_probe(
      environment=environment,
      actor=actor,
      score_fn=score_fn,
      task_goal=env_utils.TASK58_REACHABLE_SUCCESS_GOALS[env_name],
      env_name=env_name,
      obs_dim=obs_dim,
      num_episodes=args.episodes,
      interaction_threshold=args.interaction_threshold,
      mid_reach_threshold=args.mid_reach_threshold,
  )
  metrics['checkpoint'] = str(checkpoint_path)
  metrics['env_name'] = env_name
  metrics['env_steps'] = int(ckpt.get('env_steps', -1))
  print(json.dumps(metrics, indent=2, sort_keys=True))
  if args.output:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n',
                   encoding='utf-8')


if __name__ == '__main__':
  main()
