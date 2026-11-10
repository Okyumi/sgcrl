"""Success replay and local critic-gated BC for an undecomposed critic."""
from typing import NamedTuple

import jax
import jax.numpy as jnp


class SuccessBuffer(NamedTuple):
  observation: jnp.ndarray
  action: jnp.ndarray
  size: jnp.ndarray
  index: jnp.ndarray


def empty_buffer(capacity, observation_dim, action_dim):
  return SuccessBuffer(
      jnp.zeros((capacity, observation_dim), jnp.float32),
      jnp.zeros((capacity, action_dim), jnp.float32),
      jnp.asarray(0, jnp.int32), jnp.asarray(0, jnp.int32))


def insert(buffer, transitions):
  """Keep successful source-episode transitions with their ORIGINAL goal."""
  keep = transitions.extras['outcome_task_success'] > 0.5
  capacity = buffer.observation.shape[0]
  # At most one write per slot; enforce this at construction as well.
  if keep.shape[0] > capacity:
    raise ValueError('Success capacity must be at least the SGD batch size.')
  offsets = jnp.cumsum(keep.astype(jnp.int32)) - 1
  indices = jnp.where(keep, (buffer.index + offsets) % capacity, capacity)
  count = jnp.sum(keep.astype(jnp.int32))
  return SuccessBuffer(
      buffer.observation.at[indices].set(
          transitions.extras['outcome_retention_observation'], mode='drop'),
      buffer.action.at[indices].set(transitions.action, mode='drop'),
      jnp.minimum(buffer.size + count, capacity),
      (buffer.index + count) % capacity)


def local_gate(paired_score, observation, policy_action, key, obs_dim,
               state_noise=0.1):
  """DCC's eight-probe local gate, evaluated on the vanilla critic."""
  state_key, action_key = jax.random.split(key)
  state, goal = observation[:, :obs_dim], observation[:, obs_dim:]
  std = jnp.maximum(jax.lax.stop_gradient(
      jnp.std(state, axis=0, keepdims=True)), 1e-6)
  noise = jax.random.normal(state_key, (8,) + state.shape)
  states = state[None] + state_noise * std[None] * noise
  goals = jnp.broadcast_to(goal, (8,) + goal.shape)
  observations = jnp.concatenate([states, goals], axis=-1)
  actions = jnp.broadcast_to(policy_action, (8,) + policy_action.shape)
  sigma_s = jnp.std(jax.vmap(paired_score)(observations, actions), axis=0)
  probes = jax.random.uniform(action_key, actions.shape, minval=-1., maxval=1.)
  sigma_a = jnp.std(jax.vmap(lambda a: paired_score(observation, a))(probes), axis=0)
  denom = sigma_s + sigma_a
  gate = jnp.where(denom > 1e-6, sigma_s / jnp.maximum(denom, 1e-6), 1.)
  return jax.lax.stop_gradient(gate), sigma_a, sigma_s


def loss(networks, policy_params, q_params, buffer, key, obs_dim,
         batch_size=64, weight=0.1, state_noise=0.1):
  sample_key, policy_key, gate_key = jax.random.split(key, 3)
  indices = jax.random.randint(sample_key, (batch_size,), 0,
                               jnp.maximum(buffer.size, 1))
  observation, action = buffer.observation[indices], buffer.action[indices]
  distribution = networks.policy_network.apply(policy_params, observation)

  def paired_score(obs, actions):
    scores, _, _ = networks.q_network.apply(q_params, obs, actions)
    if scores.ndim == 3:
      scores = jnp.min(scores, axis=-1)
    return jnp.diag(scores)

  gate, sigma_a, sigma_s = local_gate(
      paired_score, observation, networks.sample(distribution, policy_key),
      gate_key, obs_dim, state_noise)
  bc_loss = jnp.mean(gate * -networks.log_prob(distribution, action))
  active = (buffer.size > 0).astype(jnp.float32)
  weighted = weight * active * bc_loss
  return weighted, {
      'retention/bc_loss': bc_loss,
      'retention/weighted_bc_loss': weighted,
      'retention/bc_active': active,
      'retention/bc_weight': jnp.asarray(weight),
      'retention/bc_critic_gate_mean': jnp.mean(gate),
      'retention/bc_critic_sigma_a': jnp.mean(sigma_a),
      'retention/bc_critic_sigma_s': jnp.mean(sigma_s),
      'retention/bc_effective_weight': weight * active * jnp.mean(gate),
      'retention/buffer_size': buffer.size.astype(jnp.float32),
  }
