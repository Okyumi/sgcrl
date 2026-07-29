"""Continual learner for Residual Bellman-Calibrated DCC (RBC-DCC).

This is a sibling of ``ContinualDecomposedLearner``. Keeping a distinct state
type preserves the legacy DCC execution path and checkpoint compatibility.
"""
from __future__ import annotations

import time
from typing import Dict, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.utils import counting
import jax
import jax.numpy as jnp
import optax

from contrastive import config as contrastive_config
from contrastive import rbc_networks
from contrastive import state_mask as sm
from default import make_default_logger
from sac import her


class RBCTrainingState(NamedTuple):
  """RBC state; transferred and reset groups are deliberately explicit."""

  policy_params: networks_lib.Params
  policy_opt_state: optax.OptState

  b_shared_params: networks_lib.Params
  b_shared_opt_state: optax.OptState
  h_phi_params: networks_lib.Params
  h_phi_opt_state: optax.OptState
  h_dyn_params: networks_lib.Params
  h_dyn_opt_state: optax.OptState
  phi_task_params: networks_lib.Params
  phi_task_opt_state: optax.OptState
  psi_params: networks_lib.Params
  psi_opt_state: optax.OptState

  residual_params: networks_lib.Params
  residual_opt_state: optax.OptState
  calibration_params: networks_lib.Params
  calibration_opt_state: optax.OptState
  target_params: networks_lib.Params

  alpha_params: jnp.ndarray
  alpha_optimizer_state: optax.OptState
  key: networks_lib.PRNGKey


def _online_target_bundle(state_or_params) -> Dict[str, networks_lib.Params]:
  """Extract exactly the online groups required by target hybrid Q."""
  if isinstance(state_or_params, dict):
    return {
        name: state_or_params[name]
        for name in (
            'b_shared', 'h_phi', 'phi_task', 'psi', 'residual',
            'calibration')
    }
  return {
      'b_shared': state_or_params.b_shared_params,
      'h_phi': state_or_params.h_phi_params,
      'phi_task': state_or_params.phi_task_params,
      'psi': state_or_params.psi_params,
      'residual': state_or_params.residual_params,
      'calibration': state_or_params.calibration_params,
  }


class ContinualRBCDecomposedLearner(acme.Learner):
  """DCC representation learning plus a resettable twin Bellman residual."""

  _state: RBCTrainingState

  def __init__(
      self,
      *,
      rbc_nets: rbc_networks.RBCNetworks,
      policy_network,
      sample_fn,
      log_prob_fn,
      rng,
      iterator,
      counter,
      logger,
      config: contrastive_config.ContrastiveConfig,
      continual_config,
      task_id: int = 0,
      prev_b_shared_params: Optional[networks_lib.Params] = None,
      prev_b_shared_opt_state: Optional[optax.OptState] = None,
      prev_h_phi_params: Optional[networks_lib.Params] = None,
      prev_h_phi_opt_state: Optional[optax.OptState] = None,
      prev_h_dyn_params: Optional[networks_lib.Params] = None,
      prev_h_dyn_opt_state: Optional[optax.OptState] = None,
      prev_psi_params: Optional[networks_lib.Params] = None,
      prev_psi_opt_state: Optional[optax.OptState] = None,
      log_alpha_init: float = 0.0,
  ):
    if config.use_td:
      raise ValueError(
          'critic_mode=rbc_decomposed requires config.use_td=False; '
          'RBC supplies its own scalar Bellman objective.')
    if getattr(config, 'twin_q', False):
      raise ValueError(
          'critic_mode=rbc_decomposed requires legacy twin_q=False; '
          'RBC supplies its own twin scalar Q values.')
    if config.entropy_coefficient is not None:
      raise ValueError('RBC-DCC requires adaptive entropy.')

    self._iterator = iterator
    self._counter = counter or counting.Counter()
    self._logger = logger or make_default_logger('learner')
    self._config = config
    self._continual_cfg = continual_config
    self._rbc_nets = rbc_nets
    self._decomp_nets = rbc_nets.decomposed
    self._policy_network = policy_network
    self._sample_fn = sample_fn
    self._log_prob_fn = log_prob_fn
    self._task_id = task_id
    self._timestamp = None
    self._last_metrics = {}

    lr = config.learning_rate
    self._b_shared_opt = optax.adam(lr)
    self._h_phi_opt = optax.adam(lr)
    self._h_dyn_opt = optax.adam(lr)
    self._phi_task_opt = optax.adam(lr)
    self._psi_opt = optax.adam(lr)
    self._residual_opt = optax.adam(lr)
    self._calibration_opt = optax.adam(lr)
    self._actor_opt = optax.adam(config.actor_learning_rate)
    self._alpha_opt = optax.adam(lr)

    rng, *keys = jax.random.split(rng, 11)
    shared_is_fresh = (
        task_id == 0
        or prev_b_shared_params is None
        or prev_h_phi_params is None
        or prev_h_dyn_params is None
        or prev_psi_params is None)
    decomp = rbc_nets.decomposed
    if shared_is_fresh:
      b_shared_params = decomp.init_b_shared(keys[0])
      h_phi_params = decomp.init_h_phi(keys[1])
      h_dyn_params = decomp.init_h_dyn(keys[2])
      psi_params = decomp.init_psi(keys[3])
      b_shared_opt_state = self._b_shared_opt.init(b_shared_params)
      h_phi_opt_state = self._h_phi_opt.init(h_phi_params)
      h_dyn_opt_state = self._h_dyn_opt.init(h_dyn_params)
      psi_opt_state = self._psi_opt.init(psi_params)
    else:
      b_shared_params = prev_b_shared_params
      h_phi_params = prev_h_phi_params
      h_dyn_params = prev_h_dyn_params
      psi_params = prev_psi_params
      b_shared_opt_state = (
          prev_b_shared_opt_state
          if prev_b_shared_opt_state is not None
          else self._b_shared_opt.init(b_shared_params))
      h_phi_opt_state = (
          prev_h_phi_opt_state
          if prev_h_phi_opt_state is not None
          else self._h_phi_opt.init(h_phi_params))
      h_dyn_opt_state = (
          prev_h_dyn_opt_state
          if prev_h_dyn_opt_state is not None
          else self._h_dyn_opt.init(h_dyn_params))
      psi_opt_state = (
          prev_psi_opt_state
          if prev_psi_opt_state is not None
          else self._psi_opt.init(psi_params))

    # Resettable groups are always fresh at a task boundary.
    phi_task_params = decomp.init_phi_task(keys[4])
    residual_params = rbc_nets.init_residual(keys[5])
    calibration_params = rbc_nets.init_calibration()
    policy_params = policy_network.init(keys[6])
    alpha_params = jnp.asarray(log_alpha_init, dtype=jnp.float32)

    online = {
        'b_shared': b_shared_params,
        'h_phi': h_phi_params,
        'phi_task': phi_task_params,
        'psi': psi_params,
        'residual': residual_params,
        'calibration': calibration_params,
    }
    target_params = jax.tree_util.tree_map(lambda x: x, online)

    self._state = RBCTrainingState(
        policy_params=policy_params,
        policy_opt_state=self._actor_opt.init(policy_params),
        b_shared_params=b_shared_params,
        b_shared_opt_state=b_shared_opt_state,
        h_phi_params=h_phi_params,
        h_phi_opt_state=h_phi_opt_state,
        h_dyn_params=h_dyn_params,
        h_dyn_opt_state=h_dyn_opt_state,
        phi_task_params=phi_task_params,
        phi_task_opt_state=self._phi_task_opt.init(phi_task_params),
        psi_params=psi_params,
        psi_opt_state=psi_opt_state,
        residual_params=residual_params,
        residual_opt_state=self._residual_opt.init(residual_params),
        calibration_params=calibration_params,
        calibration_opt_state=self._calibration_opt.init(calibration_params),
        target_params=target_params,
        alpha_params=alpha_params,
        alpha_optimizer_state=self._alpha_opt.init(alpha_params),
        key=keys[7],
    )
    self._update_step = self._make_update_step()

  def _make_update_step(self):
    config = self._config
    continual = self._continual_cfg
    rbc = self._rbc_nets
    decomp = rbc.decomposed
    policy = self._policy_network
    sample_fn = self._sample_fn
    log_prob_fn = self._log_prob_fn

    target_entropy = config.target_entropy
    reward_scaling = float(getattr(config, 'reward_scale', 1.0))
    dyn_weight = float(getattr(continual, 'dyn_aux_weight', 1.0))
    bellman_weight = float(getattr(continual, 'bellman_loss_weight', 1.0))
    residual_l2_weight = float(
        getattr(continual, 'bellman_residual_l2_weight', 0.0))
    bellman_discount = float(
        getattr(continual, 'bellman_discount', config.discount))
    bellman_tau = float(getattr(continual, 'bellman_tau', 0.005))
    step_penalty_reward = bool(
        getattr(continual, 'step_penalty_reward', True))
    stable_idx = jnp.asarray(sm.STABLE_INDICES)
    obs_dim = config.obs_dim
    use_cpc = bool(getattr(config, 'use_cpc', False))
    logsumexp_coeff = float(getattr(config, 'logsumexp_penalty', 0.0))

    def _critic_dict(state):
      return {
          'b_shared': state.b_shared_params,
          'h_phi': state.h_phi_params,
          'phi_task': state.phi_task_params,
          'psi': state.psi_params,
          'residual': state.residual_params,
          'calibration': state.calibration_params,
      }

    def _apply_q(params, obs, action):
      return rbc.apply_hybrid_q(
          params['b_shared'], params['h_phi'], params['phi_task'],
          params['psi'], params['residual'], params['calibration'],
          obs, action)

    def infonce_loss_fn(params, transitions):
      logits = decomp.apply_score(
          params['b_shared'], params['h_phi'], params['phi_task'],
          params['psi'], transitions.observation, transitions.action)
      batch_size = logits.shape[0]
      labels = jnp.eye(batch_size)
      if use_cpc:
        per_example = (
            optax.softmax_cross_entropy(logits=logits, labels=labels)
            + logsumexp_coeff
            * jnp.square(jax.nn.logsumexp(logits, axis=1)))
      else:
        per_example = optax.sigmoid_binary_cross_entropy(
            logits=logits, labels=labels)
      loss = jnp.mean(per_example)
      return loss, {
          'critic_loss': loss,
          'infonce': loss,
          'logsumexp': jnp.mean(
              jnp.square(jax.nn.logsumexp(logits, axis=1))),
          'binary_accuracy': jnp.mean((logits > 0) == labels),
          'categorical_accuracy': jnp.mean(
              (jnp.argmax(logits, axis=1)
               == jnp.arange(batch_size)).astype(jnp.float32)),
          'logits_pos': jnp.mean(jnp.diag(logits)),
          'logits_neg': (
              jnp.sum(logits * (1.0 - labels))
              / jnp.maximum(jnp.sum(1.0 - labels), 1.0)),
      }

    def dyn_loss_fn(p_b, p_dyn, transitions):
      hidden = decomp.apply_b_shared(
          p_b, transitions.observation, transitions.action)
      pred = decomp.apply_h_dyn(p_dyn, hidden)
      target = transitions.next_observation[:, :obs_dim][:, stable_idx]
      return jnp.mean(jnp.square(pred - target))

    def bellman_loss_fn(
        params, target_params, policy_params, alpha, transitions, key):
      next_dist = policy.apply(
          policy_params, transitions.next_observation)
      next_action = sample_fn(next_dist, key)
      next_log_prob = log_prob_fn(next_dist, next_action)
      target_q = _apply_q(
          target_params, transitions.next_observation, next_action)
      next_v = jnp.min(target_q, axis=-1) - alpha * next_log_prob
      target = jax.lax.stop_gradient(
          transitions.reward * reward_scaling
          + transitions.discount * bellman_discount * next_v)

      q_pred = _apply_q(
          params, transitions.observation, transitions.action)
      q_error = q_pred - target[:, None]
      td_loss = 0.5 * jnp.mean(jnp.square(q_error))

      z_shared, z_task, z_goal = rbc.apply_components(
          params['b_shared'], params['h_phi'], params['phi_task'],
          params['psi'], transitions.observation, transitions.action)
      delta = rbc.apply_residual(
          params['residual'],
          jax.lax.stop_gradient(z_shared),
          z_task,
          jax.lax.stop_gradient(z_goal))
      residual_l2 = jnp.mean(jnp.square(delta))
      base = rbc.apply_paired_score(
          params['b_shared'], params['h_phi'], params['phi_task'],
          params['psi'], transitions.observation, transitions.action)
      slopes = jax.nn.softplus(params['calibration']['rho'])
      reached = her.reached_from_reward(
          transitions.reward, step_penalty_reward)
      aux = {
          'td_loss': td_loss,
          'q1_mean': jnp.mean(q_pred[:, 0]),
          'q2_mean': jnp.mean(q_pred[:, 1]),
          'q_target_mean': jnp.mean(target),
          'q_target_std': jnp.std(target),
          'td_error_abs': jnp.mean(jnp.abs(q_error)),
          'her_success_rate': jnp.mean(reached.astype(jnp.float32)),
          'calibration_slope_1': slopes[0],
          'calibration_slope_2': slopes[1],
          'calibration_bias_1': params['calibration']['bias'][0],
          'calibration_bias_2': params['calibration']['bias'][1],
          'residual_rms_1': jnp.sqrt(jnp.mean(jnp.square(delta[:, 0]))),
          'residual_rms_2': jnp.sqrt(jnp.mean(jnp.square(delta[:, 1]))),
          'base_score_rms': jnp.sqrt(jnp.mean(jnp.square(base))),
          'action_score_std': jnp.std(jnp.min(q_pred, axis=-1)),
      }
      residual_rms = 0.5 * (
          aux['residual_rms_1'] + aux['residual_rms_2'])
      aux['residual_to_base_ratio'] = (
          residual_rms / jnp.maximum(aux['base_score_rms'], 1e-8))
      loss = (
          bellman_weight * td_loss
          + residual_l2_weight * residual_l2)
      return loss, aux

    def total_critic_loss_fn(
        params, target_params, policy_params, alpha, transitions, key):
      nce_loss, nce_aux = infonce_loss_fn(params, transitions)
      bellman_loss, bellman_aux = bellman_loss_fn(
          params, target_params, policy_params, alpha, transitions, key)
      return nce_loss + bellman_loss, (nce_aux, bellman_aux)

    def actor_loss_fn(policy_params, critic_params, alpha, transitions, key):
      dist = policy.apply(policy_params, transitions.observation)
      action = sample_fn(dist, key)
      log_prob = log_prob_fn(dist, action)
      q_min = jnp.min(
          _apply_q(critic_params, transitions.observation, action), axis=-1)
      loss = alpha * log_prob - q_min
      return jnp.mean(loss), {
          'actor_loss': jnp.mean(loss),
          'entropy_mean': jnp.mean(-log_prob),
          'q_pi_mean': jnp.mean(q_min),
      }

    def alpha_loss_fn(log_alpha, policy_params, transitions, key):
      dist = policy.apply(policy_params, transitions.observation)
      action = sample_fn(dist, key)
      log_prob = log_prob_fn(dist, action)
      alpha = jnp.exp(log_alpha)
      return jnp.mean(
          alpha * jax.lax.stop_gradient(-log_prob - target_entropy))

    def update_step(
        state: RBCTrainingState,
        transitions: types.Transition,
    ) -> Tuple[RBCTrainingState, Dict[str, jnp.ndarray]]:
      key, k_alpha, k_critic, k_actor = jax.random.split(state.key, 4)
      old_critic = _critic_dict(state)
      alpha = jnp.exp(state.alpha_params)

      alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
          state.alpha_params, state.policy_params, transitions, k_alpha)
      (critic_loss, (nce_aux, rbc_aux)), critic_grads = (
          jax.value_and_grad(total_critic_loss_fn, has_aux=True)(
              old_critic, state.target_params, state.policy_params, alpha,
              transitions, k_critic))
      dyn_loss, (b_dyn_grad, h_dyn_grad) = jax.value_and_grad(
          dyn_loss_fn, argnums=(0, 1))(
              state.b_shared_params, state.h_dyn_params, transitions)

      b_grad = jax.tree_util.tree_map(
          lambda nce_td, dyn: nce_td + dyn_weight * dyn,
          critic_grads['b_shared'], b_dyn_grad)
      h_dyn_grad = jax.tree_util.tree_map(
          lambda grad: dyn_weight * grad, h_dyn_grad)

      def _updated(opt, params, opt_state, grad):
        updates, new_opt_state = opt.update(grad, opt_state)
        return optax.apply_updates(params, updates), new_opt_state

      new_b, b_opt_state = _updated(
          self._b_shared_opt, state.b_shared_params,
          state.b_shared_opt_state, b_grad)
      new_h_phi, h_phi_opt_state = _updated(
          self._h_phi_opt, state.h_phi_params,
          state.h_phi_opt_state, critic_grads['h_phi'])
      new_h_dyn, h_dyn_opt_state = _updated(
          self._h_dyn_opt, state.h_dyn_params,
          state.h_dyn_opt_state, h_dyn_grad)
      new_task, task_opt_state = _updated(
          self._phi_task_opt, state.phi_task_params,
          state.phi_task_opt_state, critic_grads['phi_task'])
      new_psi, psi_opt_state = _updated(
          self._psi_opt, state.psi_params,
          state.psi_opt_state, critic_grads['psi'])
      new_residual, residual_opt_state = _updated(
          self._residual_opt, state.residual_params,
          state.residual_opt_state, critic_grads['residual'])
      new_calibration, calibration_opt_state = _updated(
          self._calibration_opt, state.calibration_params,
          state.calibration_opt_state, critic_grads['calibration'])

      new_online = {
          'b_shared': new_b,
          'h_phi': new_h_phi,
          'phi_task': new_task,
          'psi': new_psi,
          'residual': new_residual,
          'calibration': new_calibration,
      }
      new_target = rbc_networks.polyak_update(
          state.target_params, new_online, bellman_tau)

      # Match the standalone SAC update order: the actor sees the pre-update
      # critic while the target is Polyak-updated from the new critic.
      (actor_loss, actor_aux), actor_grad = jax.value_and_grad(
          actor_loss_fn, has_aux=True)(
              state.policy_params, old_critic, alpha, transitions, k_actor)
      new_policy, policy_opt_state = _updated(
          self._actor_opt, state.policy_params,
          state.policy_opt_state, actor_grad)
      new_alpha, alpha_opt_state = _updated(
          self._alpha_opt, state.alpha_params,
          state.alpha_optimizer_state, alpha_grad)

      new_state = RBCTrainingState(
          policy_params=new_policy,
          policy_opt_state=policy_opt_state,
          b_shared_params=new_b,
          b_shared_opt_state=b_opt_state,
          h_phi_params=new_h_phi,
          h_phi_opt_state=h_phi_opt_state,
          h_dyn_params=new_h_dyn,
          h_dyn_opt_state=h_dyn_opt_state,
          phi_task_params=new_task,
          phi_task_opt_state=task_opt_state,
          psi_params=new_psi,
          psi_opt_state=psi_opt_state,
          residual_params=new_residual,
          residual_opt_state=residual_opt_state,
          calibration_params=new_calibration,
          calibration_opt_state=calibration_opt_state,
          target_params=new_target,
          alpha_params=new_alpha,
          alpha_optimizer_state=alpha_opt_state,
          key=key,
      )
      metrics = {
          **nce_aux,
          'critic_loss': critic_loss,
          'decomp/L_dyn': dyn_loss,
          **{f'rbc/{name}': value for name, value in rbc_aux.items()},
          **actor_aux,
          'alpha_loss': alpha_loss,
          'alpha': jnp.exp(new_alpha),
      }
      return new_state, metrics

    num_sgd = int(getattr(config, 'num_sgd_steps_per_step', 1))

    def scan_step(state, transitions):
      batched = jax.tree_util.tree_map(
          lambda array: jnp.reshape(
              array, (num_sgd, -1, *array.shape[1:])),
          transitions)
      state, metrics = jax.lax.scan(
          update_step, state, batched, length=num_sgd)
      return state, jax.tree_util.tree_map(jnp.mean, metrics)

    return jax.jit(scan_step)

  def step(self):
    sample = next(self._iterator)
    transitions = types.Transition(*sample.data)
    self._last_transitions = transitions
    self._state, metrics = self._update_step(self._state, transitions)

    timestamp = time.time()
    elapsed = timestamp - self._timestamp if self._timestamp else 0.0
    self._timestamp = timestamp
    counts = self._counter.increment(steps=1, walltime=elapsed)
    if elapsed > 0:
      metrics = {
          **metrics,
          'steps_per_second': (
              self._config.num_sgd_steps_per_step / elapsed),
      }
    self._last_metrics = {
        **{name: float(value) for name, value in metrics.items()},
        **counts,
    }
    self._logger.write(self._last_metrics)

  def get_variables(self, names):
    critic = {
        **_online_target_bundle(self._state),
        'h_dyn': self._state.h_dyn_params,
    }
    table = {'policy': self._state.policy_params, 'critic': critic}
    return [table[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state

  def reset_actor(self, rng_key):
    if self._task_id != 0:
      raise ValueError('reset_actor is only supported during task 0.')
    policy_params = self._policy_network.init(rng_key)
    self._state = self._state._replace(
        policy_params=policy_params,
        policy_opt_state=self._actor_opt.init(policy_params))

  @property
  def last_metrics(self):
    return getattr(self, '_last_metrics', {})

  @property
  def last_transitions(self):
    return getattr(self, '_last_transitions', None)

  @property
  def q_params(self):
    return None

  @property
  def b_shared_params(self):
    return self._state.b_shared_params

  @property
  def b_shared_opt_state(self):
    return self._state.b_shared_opt_state

  @property
  def h_phi_params(self):
    return self._state.h_phi_params

  @property
  def h_phi_opt_state(self):
    return self._state.h_phi_opt_state

  @property
  def h_dyn_params(self):
    return self._state.h_dyn_params

  @property
  def h_dyn_opt_state(self):
    return self._state.h_dyn_opt_state

  @property
  def psi_params(self):
    return self._state.psi_params

  @property
  def psi_opt_state(self):
    return self._state.psi_opt_state

  @property
  def phi_task_params(self):
    return self._state.phi_task_params

  @property
  def policy_params(self):
    return self._state.policy_params
