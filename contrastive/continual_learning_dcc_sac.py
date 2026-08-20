"""DCC-SAC and action-contrastive DCC learners.

The implementation intentionally keeps the two value systems separate:

* DCC learns its contrastive representation with InfoNCE (plus the existing
  dynamics auxiliary).
* An independent twin scalar Q network learns sparse HER returns from raw
  [state, goal, action] inputs.
* TD gradients never enter DCC.
* The main dcc_sac actor retains the ordinary DCC objective and receives only
  a normalized Q action-ranking correction after the Q critic is stable.

The same learner also exposes two ablations and one reward-free algorithm:

* dcc_sac_separate: standard SAC actor, with DCC trained only as a diagnostic
  auxiliary.  This is intentionally not the proposed hybrid.
* action_dcc: DCC plus an action-contrastive transition loss, without Q.
* action_dcc_sac: action-sensitive DCC plus the stable-Q actor correction.
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

from contrastive import state_mask as sm
from contrastive import shortcut_diagnostics
from default import make_default_logger
from sac import her


_SUPPORTED_MODES = (
    'dcc_sac',
    'dcc_sac_separate',
    'action_dcc',
    'action_dcc_sac',
)


class DCCSACTrainingState(NamedTuple):
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

  q_params: Optional[networks_lib.Params]
  target_q_params: Optional[networks_lib.Params]
  q_opt_state: Optional[optax.OptState]

  alpha_params: jnp.ndarray
  alpha_opt_state: optax.OptState

  td_error_ema: jnp.ndarray
  twin_disagreement_ema: jnp.ndarray
  update_count: jnp.ndarray
  key: networks_lib.PRNGKey


class ContinualDCCSACLearner(acme.Learner):
  """DCC with an independent, confidence-gated SAC action correction."""

  _state: DCCSACTrainingState

  def __init__(
      self,
      *,
      hybrid_mode: str,
      decomp_nets,
      q_network,
      policy_network,
      sample_fn,
      log_prob_fn,
      rng,
      iterator,
      counter,
      logger,
      config,
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
  ):
    if hybrid_mode not in _SUPPORTED_MODES:
      raise ValueError(
          f'Unsupported DCC-SAC mode {hybrid_mode!r}; '
          f'expected one of {_SUPPORTED_MODES}.')
    if config.use_td:
      raise ValueError(
          f'critic_mode={hybrid_mode} requires legacy use_td=False.')
    if getattr(config, 'twin_q', False):
      raise ValueError(
          f'critic_mode={hybrid_mode} requires legacy twin_q=False; '
          'the hybrid supplies an independent twin scalar Q network.')
    if config.entropy_coefficient is not None:
      raise ValueError(
          f'critic_mode={hybrid_mode} requires adaptive entropy.')
    if getattr(continual_config, 'combine_mode', 'add') != 'add':
      raise ValueError(
          f'critic_mode={hybrid_mode} currently requires combine_mode=add.')
    if getattr(continual_config, 'goal_encoder_mode', 'shared') != 'shared':
      raise ValueError(
          f'critic_mode={hybrid_mode} currently requires '
          'goal_encoder_mode=shared.')

    self._mode = hybrid_mode
    self._use_q = hybrid_mode != 'action_dcc'
    self._fused_actor = hybrid_mode in ('dcc_sac', 'action_dcc_sac')
    self._separate_actor = hybrid_mode == 'dcc_sac_separate'
    self._use_action_contrast = hybrid_mode in (
        'action_dcc', 'action_dcc_sac')
    if self._use_q and q_network is None:
      raise ValueError(f'critic_mode={hybrid_mode} requires a Q network.')

    self._decomp_nets = decomp_nets
    self._q_network = q_network
    self._policy_network = policy_network
    self._sample_fn = sample_fn
    self._log_prob_fn = log_prob_fn
    self._iterator = iterator
    self._counter = counter or counting.Counter()
    self._logger = logger or make_default_logger('learner')
    self._config = config
    self._continual_cfg = continual_config
    self._task_id = task_id
    self._timestamp = None
    self._last_metrics: Dict[str, float] = {}

    critic_lr = float(config.learning_rate)
    q_lr = float(getattr(
        continual_config, 'dcc_sac_q_learning_rate', critic_lr))
    actor_lr = float(config.actor_learning_rate)
    self._b_shared_opt = optax.adam(critic_lr)
    self._h_phi_opt = optax.adam(critic_lr)
    self._h_dyn_opt = optax.adam(critic_lr)
    self._phi_task_opt = optax.adam(critic_lr)
    self._psi_opt = optax.adam(critic_lr)
    self._q_opt = optax.adam(q_lr, eps=1e-7)
    self._actor_opt = optax.adam(actor_lr, eps=1e-7)
    self._alpha_opt = optax.adam(q_lr)

    rng, *keys = jax.random.split(rng, 12)
    shared_is_fresh = (
        task_id == 0
        or prev_b_shared_params is None
        or prev_h_phi_params is None
        or prev_h_dyn_params is None
        or prev_psi_params is None)
    if shared_is_fresh:
      b_shared_params = decomp_nets.init_b_shared(keys[0])
      h_phi_params = decomp_nets.init_h_phi(keys[1])
      h_dyn_params = decomp_nets.init_h_dyn(keys[2])
      psi_params = decomp_nets.init_psi(keys[3])
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

    phi_task_params = decomp_nets.init_phi_task(keys[4])
    policy_params = policy_network.init(keys[5])
    if self._use_q:
      q_params = q_network.init(keys[6])
      target_q_params = jax.tree_util.tree_map(lambda x: x, q_params)
      q_opt_state = self._q_opt.init(q_params)
    else:
      q_params = None
      target_q_params = None
      q_opt_state = None
    alpha_params = jnp.asarray(0.0, dtype=jnp.float32)

    self._state = DCCSACTrainingState(
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
        q_params=q_params,
        target_q_params=target_q_params,
        q_opt_state=q_opt_state,
        alpha_params=alpha_params,
        alpha_opt_state=self._alpha_opt.init(alpha_params),
        td_error_ema=jnp.asarray(0.0, dtype=jnp.float32),
        twin_disagreement_ema=jnp.asarray(0.0, dtype=jnp.float32),
        update_count=jnp.asarray(0, dtype=jnp.int32),
        key=keys[7],
    )

    self._update_step = self._make_update_step()
    self._diagnostic_interval = int(getattr(
        continual_config, 'shortcut_diagnostic_interval', 0))
    self._diagnostic_counter = 0
    self._last_diagnostic_metrics: Dict[str, float] = {}
    self._diagnostic_fn = None
    if self._diagnostic_interval > 0:
      self._diagnostic_fn = shortcut_diagnostics.make_shortcut_diagnostic_fn(
          decomp_nets=decomp_nets,
          policy_network=policy_network,
          sample_fn=sample_fn,
          obs_dim=config.obs_dim,
          start_index=config.start_index,
          end_index=config.end_index,
          diagnostic_batch_size=int(getattr(
              continual_config, 'shortcut_diagnostic_batch_size', 32)),
          candidate_actions=int(getattr(
              continual_config, 'shortcut_candidate_actions', 16)),
          q_network=(q_network if self._use_q else None),
      )

  def _make_update_step(self):
    config = self._config
    continual = self._continual_cfg
    decomp = self._decomp_nets
    q_network = self._q_network
    policy = self._policy_network
    sample_fn = self._sample_fn
    log_prob_fn = self._log_prob_fn

    use_q = self._use_q
    fused_actor = self._fused_actor
    separate_actor = self._separate_actor
    use_action_contrast = self._use_action_contrast

    dyn_weight = float(getattr(continual, 'dyn_aux_weight', 1.0))
    q_loss_weight = float(getattr(
        continual, 'dcc_sac_q_loss_weight', 1.0))
    q_discount = float(getattr(
        continual, 'dcc_sac_discount', config.discount))
    q_tau = float(getattr(continual, 'dcc_sac_tau', 0.005))
    beta_max = float(getattr(continual, 'dcc_sac_beta_max', 0.1))
    warmup_updates = int(getattr(
        continual, 'dcc_sac_q_warmup_updates', 10000))
    ramp_updates = max(1, int(getattr(
        continual, 'dcc_sac_q_ramp_updates', 25000)))
    td_threshold = float(getattr(
        continual, 'dcc_sac_td_error_threshold', 0.5))
    twin_threshold = float(getattr(
        continual, 'dcc_sac_twin_disagreement_threshold', 0.1))
    ema_decay = float(getattr(continual, 'dcc_sac_ema_decay', 0.99))
    candidate_actions = int(getattr(
        continual, 'dcc_sac_candidate_actions', 8))
    normalization_eps = float(getattr(
        continual, 'dcc_sac_normalization_eps', 1e-3))
    correction_clip = float(getattr(
        continual, 'dcc_sac_correction_clip', 5.0))
    step_penalty_reward = bool(getattr(
        continual, 'step_penalty_reward', True))

    action_contrast_weight = (
        float(getattr(continual, 'action_contrast_weight', 1.0))
        if use_action_contrast else 0.0)
    action_contrast_temperature = float(getattr(
        continual, 'action_contrast_temperature', 1.0))
    action_contrast_batch_size = int(getattr(
        continual, 'action_contrast_batch_size', 32))

    target_entropy = config.target_entropy
    reward_scaling = float(getattr(config, 'reward_scale', 1.0))
    use_cpc = bool(getattr(config, 'use_cpc', False))
    logsumexp_coeff = float(getattr(
        config, 'logsumexp_penalty', 0.0))
    use_action_entropy = bool(getattr(
        config, 'use_action_entropy', True))
    random_goals = float(getattr(config, 'random_goals', 0.5))
    stable_idx = jnp.asarray(sm.STABLE_INDICES)
    obs_dim = config.obs_dim
    start_index = int(config.start_index)
    end_index = int(config.end_index)

    def _dcc_params(state):
      return {
          'b_shared': state.b_shared_params,
          'h_phi': state.h_phi_params,
          'phi_task': state.phi_task_params,
          'psi': state.psi_params,
      }

    def _paired(params, obs, action):
      return decomp.apply_paired_score(
          params['b_shared'], params['h_phi'], params['phi_task'],
          params['psi'], obs, action)

    def dcc_loss_fn(params, transitions):
      logits = decomp.apply_score(
          params['b_shared'], params['h_phi'], params['phi_task'],
          params['psi'], transitions.observation, transitions.action)
      batch_size = logits.shape[0]
      labels = jnp.eye(batch_size)
      if use_cpc:
        per_cell = (
            optax.softmax_cross_entropy(logits=logits, labels=labels)
            + logsumexp_coeff
            * jnp.square(jax.nn.logsumexp(logits, axis=1)))
      else:
        per_cell = optax.sigmoid_binary_cross_entropy(
            logits=logits, labels=labels)
      infonce = jnp.mean(per_cell)

      action_loss = jnp.asarray(0.0, dtype=logits.dtype)
      action_accuracy = jnp.asarray(0.0, dtype=logits.dtype)
      action_margin = jnp.asarray(0.0, dtype=logits.dtype)
      action_score_std = jnp.asarray(0.0, dtype=logits.dtype)
      if use_action_contrast:
        # Counterfactual matrix: hold (s, achieved_goal(s')) fixed and vary
        # only the replay action; the action that produced s' is positive.
        n = min(action_contrast_batch_size, transitions.action.shape[0])
        state = transitions.observation[:n, :obs_dim]
        next_state = transitions.next_observation[:n, :obs_dim]
        if end_index == -1:
          next_goal = next_state[:, start_index:]
        else:
          next_goal = next_state[:, start_index:end_index]
        anchor_obs = jnp.concatenate([state, next_goal], axis=-1)
        candidate = transitions.action[:n]

        def scores_for_anchor(one_obs):
          repeated_obs = jnp.repeat(one_obs[None, :], n, axis=0)
          return _paired(params, repeated_obs, candidate)

        action_logits = jax.vmap(scores_for_anchor)(anchor_obs)
        action_labels = jnp.eye(n)
        action_loss = jnp.mean(optax.softmax_cross_entropy(
            logits=action_logits / action_contrast_temperature,
            labels=action_labels))
        action_accuracy = jnp.mean(
            (jnp.argmax(action_logits, axis=1) == jnp.arange(n)).astype(
                jnp.float32))
        positive = jnp.diag(action_logits)
        negative = (
            jnp.sum(action_logits * (1.0 - action_labels), axis=1)
            / jnp.maximum(n - 1, 1))
        action_margin = jnp.mean(positive - negative)
        action_score_std = jnp.mean(jnp.std(action_logits, axis=1))

      total = infonce + action_contrast_weight * action_loss
      metrics = {
          'critic_loss': infonce,
          'infonce': infonce,
          'logsumexp': jnp.mean(
              jnp.square(jax.nn.logsumexp(logits, axis=1))),
          'binary_accuracy': jnp.mean((logits > 0) == labels),
          'categorical_accuracy': jnp.mean(
              (jnp.argmax(logits, axis=1) == jnp.arange(batch_size)).astype(
                  jnp.float32)),
          'logits_pos': jnp.mean(jnp.diag(logits)),
          'logits_neg': (
              jnp.sum(logits * (1.0 - labels))
              / jnp.maximum(jnp.sum(1.0 - labels), 1.0)),
          'acdcc/action_contrast_loss': action_loss,
          'acdcc/action_contrast_accuracy': action_accuracy,
          'acdcc/action_contrast_margin': action_margin,
          'acdcc/action_score_std': action_score_std,
          'acdcc/action_contrast_weight': jnp.asarray(
              action_contrast_weight, dtype=jnp.float32),
      }
      return total, metrics

    def dyn_loss_fn(p_b, p_dyn, transitions):
      hidden = decomp.apply_b_shared(
          p_b, transitions.observation, transitions.action)
      prediction = decomp.apply_h_dyn(p_dyn, hidden)
      next_state = transitions.next_observation[:, :obs_dim]
      target = next_state[:, stable_idx]
      return jnp.mean(jnp.square(prediction - target))

    def q_loss_fn(
        q_params, target_q_params, policy_params, alpha, transitions, key):
      next_dist = policy.apply(
          policy_params, transitions.next_observation)
      next_action = sample_fn(next_dist, key)
      next_log_prob = log_prob_fn(next_dist, next_action)
      next_q = q_network.apply(
          target_q_params, transitions.next_observation, next_action)
      next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
      target = jax.lax.stop_gradient(
          transitions.reward * reward_scaling
          + transitions.discount * q_discount * next_v)
      q_pred = q_network.apply(
          q_params, transitions.observation, transitions.action)
      q_error = q_pred - target[:, None]
      td_loss = 0.5 * jnp.mean(jnp.square(q_error))
      twin_abs = jnp.mean(jnp.abs(q_pred[:, 0] - q_pred[:, 1]))
      twin_normalized = twin_abs / jnp.maximum(
          jnp.mean(jnp.abs(q_pred)), 1.0)
      reached = her.reached_from_reward(
          transitions.reward, step_penalty_reward)
      next_state = transitions.next_observation[:, :obs_dim]
      relabeled_goal = transitions.observation[:, obs_dim:]
      if end_index == -1:
        achieved_next = next_state[:, start_index:]
      else:
        achieved_next = next_state[:, start_index:end_index]
      goal_distance = jnp.linalg.norm(
          achieved_next - relabeled_goal, axis=-1)
      distance_sorted = jnp.sort(goal_distance)
      distance_n = distance_sorted.shape[0]
      q_sorted = jnp.sort(jnp.reshape(q_pred, (-1,)))
      target_sorted = jnp.sort(jnp.reshape(target, (-1,)))
      td_sorted = jnp.sort(jnp.reshape(jnp.abs(q_error), (-1,)))
      q_n = q_sorted.shape[0]
      target_n = target_sorted.shape[0]
      td_n = td_sorted.shape[0]
      return td_loss, {
          'q_mean': jnp.mean(q_pred),
          'q_std': jnp.std(q_pred),
          'q_min': jnp.min(q_pred),
          'q_max': jnp.max(q_pred),
          'q_p01': q_sorted[int(0.01 * (q_n - 1))],
          'q_p99': q_sorted[int(0.99 * (q_n - 1))],
          'q_target_mean': jnp.mean(target),
          'q_target_std': jnp.std(target),
          'q_target_min': jnp.min(target),
          'q_target_max': jnp.max(target),
          'q_target_p01': target_sorted[int(0.01 * (target_n - 1))],
          'q_target_p99': target_sorted[int(0.99 * (target_n - 1))],
          'td_error_abs': jnp.mean(jnp.abs(q_error)),
          'td_error_max': jnp.max(jnp.abs(q_error)),
          'td_error_p95': td_sorted[int(0.95 * (td_n - 1))],
          'twin_disagreement_abs': twin_abs,
          'twin_disagreement_normalized': twin_normalized,
          'her_success_rate': jnp.mean(reached.astype(jnp.float32)),
          'her_goal_distance_mean': jnp.mean(goal_distance),
          'her_goal_distance_min': jnp.min(goal_distance),
          'her_goal_distance_max': jnp.max(goal_distance),
          'her_goal_distance_p50': distance_sorted[
              int(0.50 * (distance_n - 1))],
          'her_goal_distance_p95': distance_sorted[
              int(0.95 * (distance_n - 1))],
          'reward_mean': jnp.mean(transitions.reward),
          'discount_mean': jnp.mean(transitions.discount),
      }

    def actor_loss_fn(
        policy_params, dcc_params, q_params, alpha, gate, transitions, key):
      state = transitions.observation[:, :obs_dim]
      goal = transitions.observation[:, obs_dim:]
      if random_goals == 0.0:
        actor_state, actor_goal = state, goal
      elif random_goals == 0.5:
        actor_state = jnp.concatenate([state, state], axis=0)
        actor_goal = jnp.concatenate(
            [goal, jnp.roll(goal, 1, axis=0)], axis=0)
      else:
        actor_state = state
        actor_goal = jnp.roll(goal, 1, axis=0)
      actor_obs = jnp.concatenate([actor_state, actor_goal], axis=-1)

      key_action, key_reference = jax.random.split(key)
      dist = policy.apply(policy_params, actor_obs)
      action = sample_fn(dist, key_action)
      log_prob = log_prob_fn(dist, action)
      dcc_score = _paired(dcc_params, actor_obs, action)

      entropy_term = alpha * log_prob if use_action_entropy else 0.0
      q_pi = jnp.zeros_like(dcc_score)
      q_z = jnp.zeros_like(dcc_score)
      beta_effective = jnp.asarray(0.0, dtype=jnp.float32)

      if use_q:
        q_values = q_network.apply(q_params, actor_obs, action)
        q_pi = jnp.min(q_values, axis=-1)

      if separate_actor:
        loss = entropy_term - q_pi
      else:
        loss = entropy_term - dcc_score
        if fused_actor:
          reference_keys = jax.random.split(
              key_reference, candidate_actions)
          reference_actions = jax.vmap(
              lambda k: sample_fn(dist, k))(reference_keys)
          reference_q = jax.vmap(
              lambda a: jnp.min(
                  q_network.apply(q_params, actor_obs, a), axis=-1))(
                      reference_actions)
          center = jax.lax.stop_gradient(
              jnp.mean(reference_q, axis=0))
          scale = jax.lax.stop_gradient(
              jnp.maximum(jnp.std(reference_q, axis=0), normalization_eps))
          q_z = jnp.clip(
              (q_pi - center) / scale,
              -correction_clip, correction_clip)
          beta_effective = beta_max * jax.lax.stop_gradient(gate)
          loss = loss - beta_effective * q_z

      return jnp.mean(loss), {
          'actor_loss': jnp.mean(loss),
          'entropy_mean': jnp.mean(-log_prob),
          'dcc_sac/dcc_actor_score_mean': jnp.mean(dcc_score),
          'dcc_sac/q_pi_mean': jnp.mean(q_pi),
          'dcc_sac/q_normalized_mean': jnp.mean(q_z),
          'dcc_sac/q_normalized_std': jnp.std(q_z),
          'dcc_sac/q_gate': gate,
          'dcc_sac/beta_effective': beta_effective,
          'dcc_sac/q_correction_mean': jnp.mean(
              beta_effective * q_z),
          'dcc_sac/action_saturation_fraction': jnp.mean(
              (jnp.abs(action) > 0.95).astype(jnp.float32)),
      }

    def alpha_loss_fn(log_alpha, policy_params, transitions, key):
      dist = policy.apply(policy_params, transitions.observation)
      action = sample_fn(dist, key)
      log_prob = log_prob_fn(dist, action)
      alpha = jnp.exp(log_alpha)
      return jnp.mean(
          alpha * jax.lax.stop_gradient(
              -log_prob - target_entropy))

    def _tree_l2_norm(tree):
      squared = [
          jnp.sum(jnp.square(leaf))
          for leaf in jax.tree_util.tree_leaves(tree)]
      return jnp.sqrt(jnp.sum(jnp.stack(squared)))

    def _updated(opt, params, opt_state, grad):
      updates, new_opt_state = opt.update(grad, opt_state)
      return optax.apply_updates(params, updates), new_opt_state

    def update_step(
        state: DCCSACTrainingState,
        transitions: types.Transition,
    ) -> Tuple[DCCSACTrainingState, Dict[str, jnp.ndarray]]:
      key, k_q, k_actor, k_alpha = jax.random.split(state.key, 4)
      old_dcc = _dcc_params(state)
      alpha = jnp.exp(state.alpha_params)

      (dcc_total, dcc_metrics), dcc_grads = jax.value_and_grad(
          dcc_loss_fn, has_aux=True)(old_dcc, transitions)
      dyn_loss, (b_dyn_grad, h_dyn_grad) = jax.value_and_grad(
          dyn_loss_fn, argnums=(0, 1))(
              state.b_shared_params, state.h_dyn_params, transitions)

      b_grad = jax.tree_util.tree_map(
          lambda contrastive_grad, dyn_grad: (
              contrastive_grad + dyn_weight * dyn_grad),
          dcc_grads['b_shared'], b_dyn_grad)
      h_dyn_grad = jax.tree_util.tree_map(
          lambda grad: dyn_weight * grad, h_dyn_grad)

      new_b, b_opt = _updated(
          self._b_shared_opt, state.b_shared_params,
          state.b_shared_opt_state, b_grad)
      new_h_phi, h_phi_opt = _updated(
          self._h_phi_opt, state.h_phi_params,
          state.h_phi_opt_state, dcc_grads['h_phi'])
      new_h_dyn, h_dyn_opt = _updated(
          self._h_dyn_opt, state.h_dyn_params,
          state.h_dyn_opt_state, h_dyn_grad)
      new_task, task_opt = _updated(
          self._phi_task_opt, state.phi_task_params,
          state.phi_task_opt_state, dcc_grads['phi_task'])
      new_psi, psi_opt = _updated(
          self._psi_opt, state.psi_params,
          state.psi_opt_state, dcc_grads['psi'])
      new_dcc = {
          'b_shared': new_b,
          'h_phi': new_h_phi,
          'phi_task': new_task,
          'psi': new_psi,
      }

      if use_q:
        (q_loss, q_metrics), q_grad = jax.value_and_grad(
            q_loss_fn, has_aux=True)(
                state.q_params, state.target_q_params,
                state.policy_params, alpha, transitions, k_q)
        q_grad = jax.tree_util.tree_map(
            lambda grad: q_loss_weight * grad, q_grad)
        new_q, q_opt = _updated(
            self._q_opt, state.q_params, state.q_opt_state, q_grad)
        new_target_q = jax.tree_util.tree_map(
            lambda target, online: (
                (1.0 - q_tau) * target + q_tau * online),
            state.target_q_params, new_q)

        new_count = state.update_count + 1
        td_current = q_metrics['td_error_abs']
        twin_current = q_metrics['twin_disagreement_normalized']
        td_ema = jnp.where(
            state.update_count == 0,
            td_current,
            ema_decay * state.td_error_ema
            + (1.0 - ema_decay) * td_current)
        twin_ema = jnp.where(
            state.update_count == 0,
            twin_current,
            ema_decay * state.twin_disagreement_ema
            + (1.0 - ema_decay) * twin_current)
        finite = jnp.logical_and(
            jnp.isfinite(td_ema), jnp.isfinite(twin_ema))
        stable = jnp.logical_and(
            finite,
            jnp.logical_and(
                td_ema <= td_threshold,
                twin_ema <= twin_threshold))
        warm = new_count >= warmup_updates
        ramp = jnp.clip(
            (new_count.astype(jnp.float32) - warmup_updates)
            / float(ramp_updates),
            0.0, 1.0)
        stable_flag = jnp.logical_and(stable, warm).astype(jnp.float32)
        gate = ramp * stable_flag
        q_grad_norm = _tree_l2_norm(q_grad)
      else:
        q_loss = jnp.asarray(0.0, dtype=jnp.float32)
        q_metrics = {
            'q_mean': jnp.asarray(0.0),
            'q_std': jnp.asarray(0.0),
            'q_min': jnp.asarray(0.0),
            'q_max': jnp.asarray(0.0),
            'q_p01': jnp.asarray(0.0),
            'q_p99': jnp.asarray(0.0),
            'q_target_mean': jnp.asarray(0.0),
            'q_target_std': jnp.asarray(0.0),
            'q_target_min': jnp.asarray(0.0),
            'q_target_max': jnp.asarray(0.0),
            'q_target_p01': jnp.asarray(0.0),
            'q_target_p99': jnp.asarray(0.0),
            'td_error_abs': jnp.asarray(0.0),
            'td_error_max': jnp.asarray(0.0),
            'td_error_p95': jnp.asarray(0.0),
            'twin_disagreement_abs': jnp.asarray(0.0),
            'twin_disagreement_normalized': jnp.asarray(0.0),
            'her_success_rate': jnp.asarray(0.0),
            'her_goal_distance_mean': jnp.asarray(0.0),
            'her_goal_distance_min': jnp.asarray(0.0),
            'her_goal_distance_max': jnp.asarray(0.0),
            'her_goal_distance_p50': jnp.asarray(0.0),
            'her_goal_distance_p95': jnp.asarray(0.0),
            'reward_mean': jnp.mean(transitions.reward),
            'discount_mean': jnp.mean(transitions.discount),
        }
        new_q = state.q_params
        new_target_q = state.target_q_params
        q_opt = state.q_opt_state
        new_count = state.update_count + 1
        td_ema = state.td_error_ema
        twin_ema = state.twin_disagreement_ema
        stable_flag = jnp.asarray(0.0, dtype=jnp.float32)
        ramp = jnp.asarray(0.0, dtype=jnp.float32)
        gate = jnp.asarray(0.0, dtype=jnp.float32)
        q_grad_norm = jnp.asarray(0.0, dtype=jnp.float32)

      (actor_loss, actor_metrics), actor_grad = jax.value_and_grad(
          actor_loss_fn, has_aux=True)(
              state.policy_params, new_dcc, state.q_params,
              alpha, gate, transitions, k_actor)
      new_policy, policy_opt = _updated(
          self._actor_opt, state.policy_params,
          state.policy_opt_state, actor_grad)

      alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
          state.alpha_params, new_policy, transitions, k_alpha)
      new_alpha, alpha_opt = _updated(
          self._alpha_opt, state.alpha_params,
          state.alpha_opt_state, alpha_grad)

      new_state = DCCSACTrainingState(
          policy_params=new_policy,
          policy_opt_state=policy_opt,
          b_shared_params=new_b,
          b_shared_opt_state=b_opt,
          h_phi_params=new_h_phi,
          h_phi_opt_state=h_phi_opt,
          h_dyn_params=new_h_dyn,
          h_dyn_opt_state=h_dyn_opt,
          phi_task_params=new_task,
          phi_task_opt_state=task_opt,
          psi_params=new_psi,
          psi_opt_state=psi_opt,
          q_params=new_q,
          target_q_params=new_target_q,
          q_opt_state=q_opt,
          alpha_params=new_alpha,
          alpha_opt_state=alpha_opt,
          td_error_ema=td_ema,
          twin_disagreement_ema=twin_ema,
          update_count=new_count,
          key=key,
      )

      metrics = {
          **dcc_metrics,
          'decomp/L_dyn': dyn_loss,
          'dcc_sac/q_loss': q_loss,
          **{f'dcc_sac/{name}': value
             for name, value in q_metrics.items()},
          'dcc_sac/td_error_ema': td_ema,
          'dcc_sac/twin_disagreement_ema': twin_ema,
          'dcc_sac/q_stable': stable_flag,
          'dcc_sac/q_gate_ramp': ramp,
          'dcc_sac/q_gate': gate,
          'dcc_sac/q_grad_norm': q_grad_norm,
          'dcc_sac/dcc_grad_norm': _tree_l2_norm(dcc_grads),
          'dcc_sac/actor_grad_norm': _tree_l2_norm(actor_grad),
          'dcc_sac/alpha_grad_abs': jnp.abs(alpha_grad),
          'dcc_sac/shared_parameter_count': jnp.asarray(0.0),
          'dcc_sac/total_critic_loss': (
              dcc_total + dyn_weight * dyn_loss
              + q_loss_weight * q_loss),
          **actor_metrics,
          'alpha_loss': alpha_loss,
          'alpha': jnp.exp(new_alpha),
          'log_alpha': new_alpha,
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

    self._last_diagnostic_metrics = {}
    self._diagnostic_counter += 1
    if (
        self._diagnostic_fn is not None
        and self._diagnostic_counter % self._diagnostic_interval == 0):
      diagnostic_key = jax.random.fold_in(
          self._state.key, self._diagnostic_counter)
      diagnostic_metrics = self._diagnostic_fn(
          self._state.b_shared_params,
          self._state.h_phi_params,
          self._state.phi_task_params,
          self._state.psi_params,
          self._state.policy_params,
          self._state.q_params,
          transitions,
          diagnostic_key)
      self._last_diagnostic_metrics = {
          name: float(value) for name, value in diagnostic_metrics.items()}
      metrics = {**metrics, **diagnostic_metrics}

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
    # Return the DCC bundle under the standard critic variable name so the
    # existing decomposed representation-metric shim remains valid.  Scalar
    # Q parameters are exposed separately through sac_q_params.
    critic = {
        'b_shared': self._state.b_shared_params,
        'h_phi': self._state.h_phi_params,
        'phi_task': self._state.phi_task_params,
        'psi': self._state.psi_params,
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
  def last_diagnostic_metrics(self):
    """Metrics emitted on this step, empty between diagnostic events."""
    return getattr(self, '_last_diagnostic_metrics', {})

  @property
  def q_params(self):
    # The runner's generic representation-metric path assumes the monolithic
    # contrastive q-network layout.  Returning None prevents it from applying
    # that incompatible extractor to the independent scalar Q tree.
    return None

  @property
  def sac_q_params(self):
    return self._state.q_params if self._use_q else None

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
