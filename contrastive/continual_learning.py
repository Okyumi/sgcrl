"""Continual Contrastive RL learner.

Implements the continual goal-conditioned contrastive RL algorithm:
  - Base phase: standard contrastive RL on task 1.
  - Continual phase: CKA-style actor (θ' = θ_base + Σ α_j v_j + v_k)
    with persistent (never-reset) dual-encoder critic (φ, ψ).

Only v_k and β_k (and optionally α_scale) receive gradients during the
continual phase; θ_base is frozen.  The critic is fine-tuned on each task.
"""
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks
from contrastive.networks import is_actor_head_path
from contrastive.knowledge_pool import (
    KnowledgePool,
    compose_policy_params,
    _pytree_zeros_like,
    _flatten_pytree,
    # New CKA pool API (Fix A + B).
    CKAState,
    init_cka_state,
    reinit_for_new_task as cka_reinit_for_new_task,
    compute_contribution as cka_compute_contribution,
    compose_from_trainable as cka_compose_from_trainable,
    append_vector_host as cka_append_vector_host,
)
import jax
import jax.numpy as jnp
import optax
import reverb
from jax.scipy.special import logsumexp
import numpy as np
from default import make_default_logger


# ---------------------------------------------------------------------------
# Training state for the continual learner
# ---------------------------------------------------------------------------

class ContinualTrainingState(NamedTuple):
  """Training state for the continual contrastive RL learner.

  Two parallel state shapes are supported:

  * **Non-CKA modes** (``actor_mode in {'reset','persistent'}``):
    ``policy_base_params + v_k`` is the effective policy, ``beta_k``
    and ``alpha_scale`` are unused, ``actor_cka_state`` is ``None``.
    The pool is empty and the inner loop's pool-contribution is zero.
  * **CKA mode** (``actor_mode='cka'``):
    ``actor_cka_state`` carries the frozen ``base_params``, the pool of
    past-task knowledge vectors, and the per-task learnable
    ``alpha_logits`` and ``alpha_scale``. The inner loop's actor
    optimiser tracks the trainable bundle
    ``{'v_k', 'alpha_logits', 'alpha_scale'}`` jointly via
    ``actor_cka_opt_state``. The legacy ``policy_base_params``,
    ``v_k``, ``beta_k``, ``alpha_scale`` fields are unused in this mode
    (kept zero-shaped for pytree compatibility).

  The same logic applies on the critic side via ``critic_cka_state`` /
  ``critic_cka_opt_state``.
  """
  # Critic (persistent across tasks; used for non-CKA critic modes)
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  q_optimizer_state: optax.OptState

  # Actor – base (frozen after task 1, non-CKA path)
  policy_base_params: networks_lib.Params

  # Actor – current task vector v_k (optimised, non-CKA path)
  v_k: networks_lib.Params
  v_k_optimizer_state: optax.OptState

  # Legacy slots retained for state-shape compatibility; unused in CKA mode.
  beta_k: jnp.ndarray
  beta_k_optimizer_state: optax.OptState
  alpha_scale: jnp.ndarray
  alpha_scale_optimizer_state: optax.OptState

  # Entropy temperature (adaptive)
  alpha_params: Optional[jnp.ndarray]          # log_alpha
  alpha_optimizer_state: Optional[optax.OptState]

  # RNG key
  key: networks_lib.PRNGKey

  # New CKA slots (Fix A + B). When set, the inner loop sources all
  # CKA-related state from here and ignores the legacy fields above.
  actor_cka_state: Optional[CKAState] = None
  actor_cka_opt_state: Optional[optax.OptState] = None
  critic_cka_state: Optional[CKAState] = None
  critic_cka_opt_state: Optional[optax.OptState] = None


# ---------------------------------------------------------------------------
# Continual Learner
# ---------------------------------------------------------------------------

class ContinualContrastiveLearner(acme.Learner):
  """Continual contrastive RL learner with CKA-style actor adaptation."""

  _state: ContinualTrainingState

  def __init__(
      self,
      networks,
      rng,
      q_optimizer,
      vk_optimizer,
      beta_optimizer,
      alpha_scale_optimizer,
      iterator,
      counter,
      logger,
      obs_to_goal,
      config,
      continual_config,
      # --- continual state passed in from orchestrator ---
      task_id: int = 0,
      theta_base: Optional[networks_lib.Params] = None,
      pool: Optional[KnowledgePool] = None,
      prev_q_params: Optional[networks_lib.Params] = None,
      prev_target_q_params: Optional[networks_lib.Params] = None,
      prev_q_optimizer_state: Optional[optax.OptState] = None,
      critic_mode: str = 'persistent',
      actor_mode: str = 'cka',
      adapt_heads_only: bool = True,
      encoder_from_base: bool = True,
      # --- critic CKA state (only used when critic_mode='cka') ---
      q_base: Optional[networks_lib.Params] = None,
      critic_pool: Optional[KnowledgePool] = None,
      # --- previous-replay negative bank ---
      neg_bank_mode: str = 'off',       # 'off', 'vanilla', 'hard_weighted'
      neg_bank_n_per_step: int = 0,     # M bank negatives per anchor
      neg_bank_weight: float = 1.0,     # logit down-weighting (0-1]
      neg_bank_hard_ratio: int = 4,     # candidate pool multiplier for hard mining
  ):
    self._task_id = task_id
    self._actor_mode = actor_mode
    self._critic_mode = critic_mode
    self._adapt_heads_only = adapt_heads_only
    self._encoder_from_base = encoder_from_base
    # Whether the inner JIT loop should differentiate the actor/critic
    # losses through the CKA pool blend. Triggered at task >= 1 in CKA
    # modes; at task 0 we always run the plain non-CKA path because the
    # pool is empty and the base is being trained from scratch.
    self._actor_cka_path = (actor_mode == 'cka' and task_id > 0)
    self._critic_cka_path = (critic_mode == 'cka' and task_id > 0)
    self._config = config
    self._continual_config = continual_config
    self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
    self._obs_dim = config.obs_dim
    # Negative bank settings (see contrastive/negative_bank.py)
    self._neg_bank_mode = neg_bank_mode
    self._neg_bank_n_per_step = neg_bank_n_per_step
    self._neg_bank_weight = float(neg_bank_weight)
    self._neg_bank_hard_ratio = int(neg_bank_hard_ratio)
    assert neg_bank_mode in ('off', 'vanilla', 'hard_weighted'), (
        f'unknown neg_bank_mode={neg_bank_mode!r}')

    adaptive_entropy = config.entropy_coefficient is None

    # Pool
    self._pool = pool if pool is not None else KnowledgePool(
        k_max=continual_config.k_max)

    # Critic pool (for critic_mode='cka')
    self._q_base = q_base
    self._critic_pool = critic_pool if critic_pool is not None else KnowledgePool(
        k_max=continual_config.k_max)

    # ---- build head mask for gradient masking --------------------------------
    # When adapt_heads_only=True (or encoder_from_base=True with full adapt),
    # we zero out gradients for the body (encoder) parameters of v_k so that
    # only the actor head (NormalTanhDistribution) receives updates.
    # The mask is 1.0 for head leaves, 0.0 for body leaves.
    #
    # Haiku actor pytree keys:
    #   'mlp/~/linear_0', 'mlp/~/linear_1' → body (shared encoder)
    #   keys containing 'normal_tanh_distribution' → head (mean + log_std)
    #
    # Gradient masking: controls which parts of v_k receive gradients.
    #
    # In CKA-RL, the shared encoder (body) is fine-tuned on every task
    # via normal backprop.  Only the head uses the base+vectors+alpha
    # decomposition.  To match this:
    #   - v_k always receives FULL gradients (body + head).
    #   - After the task, only the HEAD portion of v_k is stored in the
    #     pool.  The BODY portion is folded into theta_base so the
    #     encoder keeps evolving.
    #
    # encoder_from_base=True freezes the body by masking body gradients.
    # adapt_heads_only controls what goes into the pool (see post-task
    # extraction in run_continual_contrastive.py), NOT gradient masking.
    if encoder_from_base and task_id > 0:
      self._mask_body_grads = True  # freeze encoder at base task values
    else:
      self._mask_body_grads = False  # body receives gradients (CKA-RL default)

    # ---- build loss functions ---------------------------------------------
    # We close over `self._pool` for the compose step inside JIT-ed functions
    # via the pool_vectors snapshot passed at each step.

    mask_body_grads = self._mask_body_grads  # close over as Python bool

    # Bank is activated only when mode != 'off', M > 0, AND we're past task 0.
    # At task 0 there are no previous tasks, so the bank is empty by
    # construction.  Keeping neg_bank_active=False at task 0 avoids JIT
    # recompilation at the task 0 → task 1 boundary.
    neg_bank_active = (self._neg_bank_mode != 'off'
                       and self._neg_bank_n_per_step > 0
                       and task_id > 0)
    neg_bank_mode_local = self._neg_bank_mode
    neg_bank_M = self._neg_bank_n_per_step
    neg_bank_weight_local = self._neg_bank_weight
    neg_bank_hard_ratio_local = self._neg_bank_hard_ratio

    actor_cka_path = self._actor_cka_path
    critic_cka_path = self._critic_cka_path

    def _make_update_step(networks, config, adaptive_entropy, obs_to_goal):
      """Factory that returns a (possibly jitted) update function."""

      # --- entropy temperature loss ----------------------------------------
      if adaptive_entropy:
        log_alpha_init = jnp.asarray(0.0, dtype=jnp.float32)
        _alpha_optimizer = optax.adam(learning_rate=3e-4)

      def alpha_loss_fn(log_alpha, combined_policy_params, transitions, key):
        dist_params = networks.policy_network.apply(
            combined_policy_params, transitions.observation)
        action = networks.sample(dist_params, key)
        log_prob = networks.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - config.target_entropy))

      # --- critic loss (InfoNCE – same as original) -------------------------
      def critic_loss_fn(q_params, combined_policy_params, target_q_params,
                         transitions, key, bank_goals=None):
        """Critic loss with optional previous-replay negative bank.

        When bank_goals is None (or neg_bank_active is False), this reduces
        to the standard in-batch InfoNCE loss.  Otherwise, bank goals are
        appended as extra negatives: the logits matrix grows from [B, B]
        to [B, B+M'] where M' depends on the bank mode.
        """
        batch_size = transitions.observation.shape[0]
        I = jnp.eye(batch_size)

        if config.use_td:
          s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)
          next_s, _ = jnp.split(transitions.next_observation, [config.obs_dim], axis=1)
          if config.add_mc_to_td:
            next_fraction = (1 - config.discount) / ((1 - config.discount) + 1)
            num_next = int(batch_size * next_fraction)
            new_g = jnp.concatenate([obs_to_goal(next_s[:num_next]), g[num_next:]], axis=0)
          else:
            new_g = obs_to_goal(next_s)
          obs = jnp.concatenate([s, new_g], axis=1)
          transitions = transitions._replace(observation=obs)

        logits, sa_repr, g_repr = networks.q_network.apply(
            q_params, transitions.observation, transitions.action)

        # ---- Negative bank: extra out-of-batch negatives --------------------
        # Only supported for the non-TD, non-twin CPC path.  When the TD or
        # twin_q paths are enabled, bank negatives are silently skipped.
        bank_logits_out = None
        if (neg_bank_active and bank_goals is not None
            and not config.use_td and len(logits.shape) == 2):
          # Encode bank goals via ψ.  We do this by running the g_encoder
          # portion of q_network.  The q_network's repr_fn takes
          # (obs=state||goal, action) and returns (sa_repr, g_repr).  We
          # pass dummy state/action and only use the g_repr on the BANK
          # portion of the observation.
          # Simpler: directly build fake observations [state_dummy, bank_goal]
          # and use the repr_fn's g-path.  But the cleanest approach is to
          # construct a dummy batch with the bank goals and extract g_repr.
          dummy_state = jnp.zeros((bank_goals.shape[0], config.obs_dim))
          bank_obs = jnp.concatenate([dummy_state, bank_goals], axis=1)
          dummy_action = jnp.zeros((bank_goals.shape[0],
                                    transitions.action.shape[-1]))
          _, _, bank_g_repr = networks.q_network.apply(
              q_params, bank_obs, dummy_action)
          # bank_g_repr shape: [bank_goals.shape[0], repr_dim]
          # stop gradient through bank g_repr? No — we want the critic to
          # learn from these negatives.  Keep gradient flowing.

          # Compute [B, bank_goals.shape[0]] bank-logits using same energy
          # as the main critic (inner_product or l2).
          if config.energy_fn == 'l2':
            bank_logits = -jnp.sqrt(
                jnp.sum((sa_repr[:, None, :] - bank_g_repr[None, :, :]) ** 2,
                        axis=-1) + 1e-6)
          else:
            bank_logits = jnp.einsum('ik,jk->ij', sa_repr, bank_g_repr)

          if neg_bank_mode_local == 'hard_weighted':
            # Per-anchor top-M by score (hard negatives)
            _, topk_idx = jax.lax.top_k(bank_logits, neg_bank_M)  # [B, M]
            # Gather per-anchor logits: [B, M]
            batch_idx = jnp.arange(batch_size)[:, None]
            bank_logits_selected = bank_logits[batch_idx, topk_idx]
            # Down-weight via a scalar multiplier on the logits.  Scaling
            # logits by w<1 reduces how "confident" the softmax is about
            # these negatives, limiting their gradient contribution.
            bank_logits_out = bank_logits_selected * neg_bank_weight_local
          else:  # 'vanilla'
            # Use all bank candidates as shared negatives (no per-anchor
            # selection).  Shape: [B, bank_goals.shape[0]]
            bank_logits_out = bank_logits * neg_bank_weight_local

        if config.use_td:
          assert len(logits.shape) == 3
          s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)
          del s
          next_s = transitions.next_observation[:, :config.obs_dim]
          goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)
          g = g[goal_indices]
          transitions = transitions._replace(
              next_observation=jnp.concatenate([next_s, g], axis=1))
          next_dist = networks.policy_network.apply(
              combined_policy_params, transitions.next_observation)
          next_action = networks.sample(next_dist, key)
          next_q, _, _ = networks.q_network.apply(
              target_q_params, transitions.next_observation, next_action)
          next_q = jax.nn.sigmoid(next_q)
          next_v = jnp.min(next_q, axis=-1)
          next_v = jax.lax.stop_gradient(next_v)
          next_v = jnp.diag(next_v)
          w = next_v / (1 - next_v)
          w = jnp.clip(w, 0, 20.0)
          pos_logits = jax.vmap(jnp.diag, -1, -1)(logits)
          loss_pos = optax.sigmoid_binary_cross_entropy(logits=pos_logits, labels=1)
          neg_logits = logits[jnp.arange(batch_size), goal_indices]
          loss_neg1 = w[:, None] * optax.sigmoid_binary_cross_entropy(logits=neg_logits, labels=1)
          loss_neg2 = optax.sigmoid_binary_cross_entropy(logits=neg_logits, labels=0)
          if config.add_mc_to_td:
            loss = (1 + (1 - config.discount)) * loss_pos + config.discount * loss_neg1 + 2 * loss_neg2
          else:
            loss = (1 - config.discount) * loss_pos + config.discount * loss_neg1 + loss_neg2
          logits = jnp.mean(logits, axis=-1)
        else:
          # Build extended logits and labels if bank negatives are active.
          # Extended labels: [I_B, zeros(B, M)] — positives only at diagonal,
          # bank negatives are all labelled negative.
          if bank_logits_out is not None:
            extended_logits = jnp.concatenate([logits, bank_logits_out], axis=1)
            extended_I = jnp.concatenate([I, jnp.zeros_like(bank_logits_out)],
                                         axis=1)
          else:
            extended_logits = logits
            extended_I = I

          def loss_fn(_logits, _labels):
            if config.use_cpc:
              return (optax.softmax_cross_entropy(logits=_logits, labels=_labels)
                      + config.logsumexp_penalty * jax.nn.logsumexp(_logits, axis=1)**2)
            else:
              return optax.sigmoid_binary_cross_entropy(logits=_logits, labels=_labels)
          if len(logits.shape) == 3:
            # Twin-Q path: bank negatives not supported; use original logits.
            loss = jax.vmap(lambda _l: loss_fn(_l, I), in_axes=2,
                            out_axes=-1)(logits)
            loss = jnp.mean(loss, axis=-1)
            logits = jnp.mean(logits, axis=-1)
          else:
            loss = loss_fn(extended_logits, extended_I)

        loss = jnp.mean(loss)
        correct = (jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1))
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
        if len(logits.shape) == 3:
          lse = jax.nn.logsumexp(logits[:, :, 0], axis=1)**2
        else:
          lse = jax.nn.logsumexp(logits, axis=1)**2

        metrics = {
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logsumexp': lse.mean(),
        }
        if bank_logits_out is not None:
          # Mean bank-negative logit (pre-weight division to recover raw score).
          bank_raw = (bank_logits_out / jnp.maximum(neg_bank_weight_local, 1e-8))
          metrics['bank/logits_mean'] = jnp.mean(bank_raw)
          metrics['bank/logits_max'] = jnp.mean(jnp.max(bank_raw, axis=1))
          # Extended-set categorical accuracy: how often does the true
          # positive beat ALL negatives (in-batch + bank)?
          ext_correct = (jnp.argmax(extended_logits, axis=1)
                         == jnp.argmax(I, axis=1))
          metrics['bank/extended_categorical_accuracy'] = jnp.mean(ext_correct)
        return loss, metrics

      # --- actor loss -------------------------------------------------------
      def actor_loss_fn(combined_policy_params, q_params, alpha,
                        transitions, key):
        obs = transitions.observation
        state = obs[:, :config.obs_dim]
        goal = obs[:, config.obs_dim:]

        if config.random_goals == 0.0:
          new_state, new_goal = state, goal
        elif config.random_goals == 0.5:
          new_state = jnp.concatenate([state, state], axis=0)
          new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
        else:
          new_state = state
          new_goal = jnp.roll(goal, 1, axis=0)

        new_obs = jnp.concatenate([new_state, new_goal], axis=1)
        dist_params = networks.policy_network.apply(combined_policy_params, new_obs)
        action = networks.sample(dist_params, key)
        log_prob = networks.log_prob(dist_params, action)

        # Critic score: inner product φ(s,a)^T ψ(g), matching SGCRL.
        q_action, _, _ = networks.q_network.apply(q_params, new_obs, action)
        if len(q_action.shape) == 3:  # twin_q
          q_action = jnp.min(q_action, axis=-1)
        actor_loss = -jnp.diag(q_action)  # maximize Q

        if config.use_action_entropy:
          actor_loss -= alpha * (-log_prob)  # maximize entropy

        metrics = {'entropy_mean': jnp.mean(-log_prob)}
        return jnp.mean(actor_loss), metrics

      # --- combined update step ---------------------------------------------
      alpha_grad = jax.value_and_grad(alpha_loss_fn)
      critic_grad = jax.value_and_grad(critic_loss_fn, has_aux=True)

      def update_step(state, data):
        """Single SGD update.

        Args:
          state: ContinualTrainingState
          data: tuple of (transitions[, bank_goals]). The pool blend is
            sourced entirely from ``state.actor_cka_state`` /
            ``state.critic_cka_state`` when the corresponding CKA path is
            active, so it is no longer passed in as data.
        """
        if neg_bank_active:
          transitions, bank_goals = data
        else:
          transitions = data
          bank_goals = None
        key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)

        # -- compose effective policy params --------------------------------
        # In CKA mode we differentiate through the pool blend; in non-CKA
        # modes we use the legacy policy_base_params + v_k path.
        if actor_cka_path:
          actor_trainable = {
              'v_k': state.actor_cka_state.v_k,
              'alpha_logits': state.actor_cka_state.alpha_logits,
              'alpha_scale': state.actor_cka_state.alpha_scale,
          }
          combined_policy = cka_compose_from_trainable(
              state.actor_cka_state, actor_trainable)
        else:
          combined_policy = jax.tree.map(
              lambda base, vk: base + vk,
              state.policy_base_params, state.v_k,
          )

        # In CKA mode for the critic, the effective q_params come from
        # composing q_base + sum_j alpha_j w_j + w_k. Otherwise q_params
        # are used directly.
        if critic_cka_path:
          critic_trainable = {
              'v_k': state.critic_cka_state.v_k,
              'alpha_logits': state.critic_cka_state.alpha_logits,
              'alpha_scale': state.critic_cka_state.alpha_scale,
          }
          composed_q_params = cka_compose_from_trainable(
              state.critic_cka_state, critic_trainable)
          composed_target_q_params = state.target_q_params  # follows q
        else:
          composed_q_params = state.q_params
          composed_target_q_params = state.target_q_params

        # -- entropy coefficient --------------------------------------------
        if adaptive_entropy:
          _alpha_loss, _alpha_grads = alpha_grad(
              state.alpha_params, combined_policy, transitions, key_alpha)
          alpha = jnp.exp(state.alpha_params)
        else:
          alpha = config.entropy_coefficient
          _alpha_loss = 0.0

        # -- critic update --------------------------------------------------
        # In non-CKA critic mode, we differentiate critic_loss w.r.t.
        # q_params directly.
        # In CKA critic mode, we differentiate w.r.t. the trainable bundle
        # (v_k, alpha_logits, alpha_scale) via cka_compose_from_trainable.
        if critic_cka_path:
          def critic_loss_cka(critic_trainable, transitions, key,
                              bank_goals):
            composed = cka_compose_from_trainable(
                state.critic_cka_state, critic_trainable)
            return critic_loss_fn(composed, combined_policy,
                                  state.target_q_params,
                                  transitions, key, bank_goals)
          critic_grad_cka = jax.value_and_grad(critic_loss_cka,
                                               has_aux=True)
          (c_loss, c_metrics), c_bundle_grads = critic_grad_cka(
              critic_trainable, transitions, key_critic, bank_goals)
          c_updates, c_cka_opt = q_optimizer.update(
              c_bundle_grads, state.critic_cka_opt_state)
          new_critic_trainable = optax.apply_updates(
              critic_trainable, c_updates)
          new_critic_cka_state = state.critic_cka_state.replace(
              v_k=new_critic_trainable['v_k'],
              alpha_logits=new_critic_trainable['alpha_logits'],
              alpha_scale=new_critic_trainable['alpha_scale'],
          )
          # Effective q for target update.
          new_composed_q = cka_compose_from_trainable(
              new_critic_cka_state, new_critic_trainable)
          new_target = jax.tree.map(
              lambda x, y: x * (1 - config.tau) + y * config.tau,
              state.target_q_params, new_composed_q)
          q_params = state.q_params  # legacy; unused in CKA path
          q_opt_state = state.q_optimizer_state  # legacy; unused
        else:
          (c_loss, c_metrics), c_grads = critic_grad(
              state.q_params, combined_policy, state.target_q_params,
              transitions, key_critic, bank_goals)
          c_updates, q_opt_state = q_optimizer.update(
              c_grads, state.q_optimizer_state)
          q_params = optax.apply_updates(state.q_params, c_updates)
          new_target = jax.tree.map(
              lambda x, y: x * (1 - config.tau) + y * config.tau,
              state.target_q_params, q_params)
          new_critic_cka_state = state.critic_cka_state
          c_cka_opt = state.critic_cka_opt_state

        # -- actor update --------------------------------------------------
        # In non-CKA actor mode: differentiate actor_loss w.r.t. v_k
        # (through combined_policy = base + v_k). Body grads optionally
        # masked.
        # In CKA actor mode: differentiate actor_loss w.r.t. the trainable
        # bundle (v_k, alpha_logits, alpha_scale). The actor optimiser
        # updates all three jointly.
        if actor_cka_path:
          def actor_loss_cka(actor_trainable, q_params_for_actor, alpha,
                             transitions, key):
            composed = cka_compose_from_trainable(
                state.actor_cka_state, actor_trainable)
            return actor_loss_fn(composed, q_params_for_actor, alpha,
                                 transitions, key)
          actor_grad_cka = jax.value_and_grad(actor_loss_cka,
                                              has_aux=True)
          q_for_actor = composed_q_params if critic_cka_path else state.q_params
          (a_loss, a_metrics), a_bundle_grads = actor_grad_cka(
              actor_trainable, q_for_actor, alpha, transitions, key_actor)
          # Body-grad masking on the v_k portion only (alpha_logits and
          # alpha_scale are unaffected).
          if mask_body_grads:
            def _mask_leaf(path, g):
              path_str = '/'.join(str(p) for p in path)
              return g if is_actor_head_path(path_str) else jnp.zeros_like(g)
            a_bundle_grads = {
                **a_bundle_grads,
                'v_k': jax.tree_util.tree_map_with_path(
                    _mask_leaf, a_bundle_grads['v_k']),
            }
          a_updates, a_cka_opt = vk_optimizer.update(
              a_bundle_grads, state.actor_cka_opt_state)
          new_actor_trainable = optax.apply_updates(
              actor_trainable, a_updates)
          new_actor_cka_state = state.actor_cka_state.replace(
              v_k=new_actor_trainable['v_k'],
              alpha_logits=new_actor_trainable['alpha_logits'],
              alpha_scale=new_actor_trainable['alpha_scale'],
          )
          v_k_new = state.v_k          # legacy slot, unchanged
          vk_opt_state = state.v_k_optimizer_state
        else:
          actor_loss_and_grad = jax.value_and_grad(actor_loss_fn,
                                                   has_aux=True)
          (a_loss, a_metrics), a_grads_combined = actor_loss_and_grad(
              combined_policy, state.q_params, alpha,
              transitions, key_actor)
          if mask_body_grads:
            def _mask_leaf(path, g):
              path_str = '/'.join(str(p) for p in path)
              return g if is_actor_head_path(path_str) else jnp.zeros_like(g)
            a_grads_combined = jax.tree_util.tree_map_with_path(
                _mask_leaf, a_grads_combined)
          vk_updates, vk_opt_state = vk_optimizer.update(
              a_grads_combined, state.v_k_optimizer_state)
          v_k_new = optax.apply_updates(state.v_k, vk_updates)
          new_actor_cka_state = state.actor_cka_state
          a_cka_opt = state.actor_cka_opt_state

        # -- legacy beta_k / alpha_scale: untouched by inner loop ------------
        beta_k_new = state.beta_k
        beta_opt_state = state.beta_k_optimizer_state
        alpha_scale_new = state.alpha_scale
        alpha_scale_opt_state = state.alpha_scale_optimizer_state

        metrics = dict(c_metrics)
        metrics.update({
            'critic_loss': c_loss,
            'actor_loss': a_loss,
        })
        metrics.update(a_metrics)

        # CKA diagnostics: visible alpha distribution + scale.
        if actor_cka_path:
          a_logits = new_actor_cka_state.alpha_logits
          a_scale = new_actor_cka_state.alpha_scale
          a_pool_mask = new_actor_cka_state.pool.mask
          a_logits_masked = jnp.where(a_pool_mask,
                                      a_logits * a_scale, -jnp.inf)
          any_active = jnp.any(a_pool_mask)
          a_softmax = jax.nn.softmax(
              jnp.where(any_active, a_logits_masked,
                        jnp.zeros_like(a_logits_masked)),
              axis=0)
          a_softmax = jnp.where(any_active, a_softmax,
                                jnp.zeros_like(a_softmax))
          metrics['actor_alpha_max'] = jnp.max(a_softmax)
          metrics['actor_alpha_entropy'] = -jnp.sum(
              a_softmax * jnp.log(a_softmax + 1e-12))
          metrics['actor_alpha_scale'] = a_scale
        if critic_cka_path:
          c_logits = new_critic_cka_state.alpha_logits
          c_scale = new_critic_cka_state.alpha_scale
          c_pool_mask = new_critic_cka_state.pool.mask
          c_logits_masked = jnp.where(c_pool_mask,
                                      c_logits * c_scale, -jnp.inf)
          any_active_c = jnp.any(c_pool_mask)
          c_softmax = jax.nn.softmax(
              jnp.where(any_active_c, c_logits_masked,
                        jnp.zeros_like(c_logits_masked)),
              axis=0)
          c_softmax = jnp.where(any_active_c, c_softmax,
                                jnp.zeros_like(c_softmax))
          metrics['critic_alpha_max'] = jnp.max(c_softmax)
          metrics['critic_alpha_entropy'] = -jnp.sum(
              c_softmax * jnp.log(c_softmax + 1e-12))
          metrics['critic_alpha_scale'] = c_scale

        new_state = ContinualTrainingState(
            q_params=q_params,
            target_q_params=new_target,
            q_optimizer_state=q_opt_state,
            policy_base_params=state.policy_base_params,
            v_k=v_k_new,
            v_k_optimizer_state=vk_opt_state,
            beta_k=beta_k_new,
            beta_k_optimizer_state=beta_opt_state,
            alpha_scale=alpha_scale_new,
            alpha_scale_optimizer_state=alpha_scale_opt_state,
            alpha_params=state.alpha_params,
            alpha_optimizer_state=state.alpha_optimizer_state,
            key=key,
            actor_cka_state=new_actor_cka_state,
            actor_cka_opt_state=a_cka_opt,
            critic_cka_state=new_critic_cka_state,
            critic_cka_opt_state=c_cka_opt,
        )

        if adaptive_entropy:
          _alpha_update, _alpha_opt = _alpha_optimizer.update(
              _alpha_grads, state.alpha_optimizer_state)
          new_alpha_params = optax.apply_updates(state.alpha_params, _alpha_update)
          metrics['alpha_loss'] = _alpha_loss
          metrics['alpha'] = jnp.exp(new_alpha_params)
          new_state = new_state._replace(
              alpha_optimizer_state=_alpha_opt,
              alpha_params=new_alpha_params)

        return new_state, metrics

      return update_step, (log_alpha_init if adaptive_entropy else None,
                           _alpha_optimizer if adaptive_entropy else None)

    # ---- build the update step & initial state ----------------------------
    update_step, (log_alpha_init, alpha_opt) = _make_update_step(
        networks, config, adaptive_entropy, obs_to_goal)

    # Wrap with lax.scan for num_sgd_steps_per_step, matching the original
    # SGCRL learner.  We can't use process_multiple_batches directly because
    # pool_contribution is a param-shaped pytree (not batch-indexed), so the
    # reshape it applies would break.  Instead we write a thin wrapper that
    # scans over mini-batches of transitions while broadcasting pool_c.
    num_sgd = config.num_sgd_steps_per_step

    def _scan_update(state, data):
      """Run num_sgd_steps_per_step updates via lax.scan.

      data = transitions [or (transitions, bank_goals)]
        transitions: leaves have shape [batch_size * num_sgd, ...]
        bank_goals (optional): [M_pool, goal_dim] — shared across all SGD
          steps in this scan (the bank doesn't change within one learner
          step, so we reuse the same sample).

      Pool contribution is no longer passed in; it is recomputed inside
      every JIT step from ``state.actor_cka_state`` /
      ``state.critic_cka_state`` so that ``alpha_logits`` and
      ``alpha_scale`` receive in-loop gradient updates.
      """
      if neg_bank_active:
        transitions, bank_goals = data
      else:
        transitions = data
        bank_goals = None
      # Reshape transitions: [B*N, ...] -> [N, B, ...]
      batched_transitions = jax.tree.map(
          lambda a: jnp.reshape(a, (num_sgd, -1, *a.shape[1:])),
          transitions)

      def scan_body(carry, mini_batch):
        if neg_bank_active:
          return update_step(carry, (mini_batch, bank_goals))
        return update_step(carry, mini_batch)

      state, metrics = jax.lax.scan(
          scan_body, state, batched_transitions, length=num_sgd)
      # Average metrics across SGD steps.
      metrics = jax.tree.map(jnp.mean, metrics)
      return state, metrics

    if config.jit:
      self._update_step = jax.jit(_scan_update)
    else:
      self._update_step = _scan_update

    # ---- initialise state -------------------------------------------------
    key_policy, key_q, rng = jax.random.split(rng, 3)

    # Pool capacity for CKA states (fixed, JIT-stable shape).
    capacity = continual_config.k_max + 1

    if theta_base is None:
      # Fresh init (base task, actor_mode='reset', or actor_mode='persistent'
      # on task 0).  The actor is initialised from scratch.
      policy_params = networks.policy_network.init(key_policy)
      theta_base = policy_params

      # Critic init: respect critic_mode even when actor is fresh.
      if prev_q_params is not None and critic_mode == 'persistent':
        q_params = prev_q_params
      elif critic_mode == 'cka' and self._q_base is not None and task_id > 0:
        # Actor is reset but critic uses CKA decomposition. We compose the
        # *initial* q_params from the new CKAState (handled below); for
        # bookkeeping we set q_params = q_base here and let the CKA
        # composition take over inside the inner loop.
        q_params = self._q_base
      else:
        q_params = networks.q_network.init(key_q)
    else:
      # Continual phase: actor_mode in {'persistent','cka'} with task_id > 0.
      policy_params = theta_base  # not used directly; composed via v_k / CKA

      if critic_mode == 'persistent':
        # Carry forward critic from previous task (never reset)
        assert prev_q_params is not None
        q_params = prev_q_params
      elif critic_mode == 'reset':
        # Reinitialize critic from scratch each task
        q_params = networks.q_network.init(key_q)
      elif critic_mode == 'cka':
        # CKA-style critic: q' = q_base + Σ α_j w_j + w_k. q_base is
        # frozen from task 0. The composed q_params are sourced from
        # ``critic_cka_state`` inside the inner loop; the legacy
        # ``q_params`` slot just carries q_base for reference.
        assert self._q_base is not None, (
            'critic_mode=cka requires q_base (frozen from task 0)')
        q_params = self._q_base
      else:
        raise ValueError(f'Unknown critic_mode: {critic_mode}')

    # ---- legacy non-CKA actor slots (kept for non-CKA modes) ---------------
    v_k = _pytree_zeros_like(theta_base)
    beta_k = jnp.zeros(0)
    alpha_scale = jnp.ones(1)
    vk_opt_state = vk_optimizer.init(v_k)
    beta_opt_state = beta_optimizer.init(beta_k)
    alpha_scale_opt_state = alpha_scale_optimizer.init(alpha_scale)

    # ---- new CKA actor state (Fix A + B) ----------------------------------
    actor_cka_state = None
    actor_cka_opt_state = None
    if self._actor_cka_path:
      key_logits, rng = jax.random.split(rng)
      # Build empty pool, then append each legacy pool vector via the
      # host-side append (vectors are already shaped like theta_base since
      # they were extracted from prior-task v_k).
      actor_cka_state = init_cka_state(theta_base, capacity=capacity)
      for v_j in self._pool.get_vectors():
        actor_cka_state = actor_cka_state.replace(
            pool=cka_append_vector_host(
                actor_cka_state.pool, v_j, k_max=continual_config.k_max))
      actor_cka_state = cka_reinit_for_new_task(
          actor_cka_state, theta_base, key_logits,
          alpha_logits_init_std=continual_config.beta_init_std,
          alpha_scale_init=1.0,
      )
      actor_bundle = {
          'v_k': actor_cka_state.v_k,
          'alpha_logits': actor_cka_state.alpha_logits,
          'alpha_scale': actor_cka_state.alpha_scale,
      }
      actor_cka_opt_state = vk_optimizer.init(actor_bundle)

    # ---- new CKA critic state (Fix A + B) ---------------------------------
    critic_cka_state = None
    critic_cka_opt_state = None
    if self._critic_cka_path:
      assert self._q_base is not None, (
          'critic_mode=cka requires q_base from task 0')
      key_clogits, rng = jax.random.split(rng)
      critic_cka_state = init_cka_state(self._q_base, capacity=capacity)
      for w_j in self._critic_pool.get_vectors():
        critic_cka_state = critic_cka_state.replace(
            pool=cka_append_vector_host(
                critic_cka_state.pool, w_j, k_max=continual_config.k_max))
      critic_cka_state = cka_reinit_for_new_task(
          critic_cka_state, self._q_base, key_clogits,
          alpha_logits_init_std=continual_config.beta_init_std,
          alpha_scale_init=1.0,
      )
      critic_bundle = {
          'v_k': critic_cka_state.v_k,
          'alpha_logits': critic_cka_state.alpha_logits,
          'alpha_scale': critic_cka_state.alpha_scale,
      }
      critic_cka_opt_state = q_optimizer.init(critic_bundle)

    # ---- legacy critic optimizer / target ---------------------------------
    # Track whether critic was freshly initialised / recomposed.
    # When True, legacy optimizer state and target Q must be reinitialised.
    critic_was_freshly_init = (
        task_id == 0
        or critic_mode in ('reset', 'cka')
        or prev_q_optimizer_state is None
    )

    if critic_was_freshly_init:
      q_opt_state = q_optimizer.init(q_params)
    else:
      q_opt_state = prev_q_optimizer_state

    if critic_was_freshly_init:
      target_q = q_params
    else:
      target_q = prev_target_q_params if prev_target_q_params is not None else q_params

    # In critic CKA path, the legacy ``q_params`` slot carries q_base; the
    # target Q is a separate (mutable across tasks) snapshot that the inner
    # JIT loop polyak-updates against the *composed* critic. We initialise
    # it to q_base (which equals the composed value when alpha is masked
    # and v_k = 0).
    if self._critic_cka_path:
      target_q = q_params

    # Entropy
    if adaptive_entropy:
      alpha_params_init = log_alpha_init
      alpha_opt_state_init = alpha_opt.init(alpha_params_init)
    else:
      alpha_params_init = None
      alpha_opt_state_init = None

    self._state = ContinualTrainingState(
        q_params=q_params,
        target_q_params=target_q,
        q_optimizer_state=q_opt_state,
        policy_base_params=theta_base,
        v_k=v_k,
        v_k_optimizer_state=vk_opt_state,
        beta_k=beta_k,
        beta_k_optimizer_state=beta_opt_state,
        alpha_scale=alpha_scale,
        alpha_scale_optimizer_state=alpha_scale_opt_state,
        alpha_params=alpha_params_init,
        alpha_optimizer_state=alpha_opt_state_init,
        key=rng,
        actor_cka_state=actor_cka_state,
        actor_cka_opt_state=actor_cka_opt_state,
        critic_cka_state=critic_cka_state,
        critic_cka_opt_state=critic_cka_opt_state,
    )

    self._networks = networks
    self._q_optimizer = q_optimizer
    self._vk_optimizer = vk_optimizer
    self._beta_optimizer = beta_optimizer
    self._alpha_scale_optimizer = alpha_scale_optimizer

    # Logging / counting
    self._counter = counter or counting.Counter()
    self._logger = logger or make_default_logger(
        'learner', asynchronous=True,
        serialize_fn=utils.fetch_devicearray, time_delta=10.0)
    self._iterator = iterator
    self._timestamp = None

  # ---- main step ----------------------------------------------------------
  #
  # Note: prior versions of this learner ran a host-side
  # ``_compute_pool_contribution`` per learner step plus a separate non-JIT
  # ``_update_beta_and_alpha_scale`` pass that updated ``beta_k`` /
  # ``alpha_scale`` once per learner step (cadence imbalance ~64x vs the
  # inner JIT loop, see Bugs 1–2 in docs/audit_apr26_cka_sgcrl.md). Both are
  # gone now: in CKA mode the trainable bundle is jointly optimised inside
  # the JIT body via ``cka_compose_from_trainable``.

  def set_bank_goals(self, bank_goals):
    """Set bank goals for subsequent step() calls.

    Args:
      bank_goals: jnp.ndarray of shape [M_pool, goal_dim].  Must have the
        same shape for all calls (JIT recompiles otherwise).  Only used
        when the learner's neg_bank_active is True (task > 0 and mode !=
        'off').  At task 0 this is a no-op.
    """
    self._current_bank_goals = bank_goals

  @property
  def neg_bank_active(self) -> bool:
    """True when this learner uses bank negatives in its update step."""
    return (self._neg_bank_mode != 'off'
            and self._neg_bank_n_per_step > 0
            and self._task_id > 0)

  def step(self):
    with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
      sample = next(self._iterator)
      transitions = types.Transition(*sample.data)

      # Cache the last batch for external use (e.g., rl_metrics)
      self._last_transitions = transitions

      # Single call: lax.scan handles the num_sgd_steps_per_step inner loop.
      # Pool contribution is sourced from state.actor_cka_state /
      # state.critic_cka_state inside the JIT body, so it is no longer
      # passed in.
      if self.neg_bank_active:
        bank_goals = getattr(self, '_current_bank_goals', None)
        assert bank_goals is not None, (
            'neg_bank_active=True but no bank_goals set. '
            'Call learner.set_bank_goals(goals) before step().')
        self._state, metrics = self._update_step(
            self._state, (transitions, bank_goals))
      else:
        self._state, metrics = self._update_step(self._state, transitions)

    # Timing
    timestamp = time.time()
    elapsed = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    counts = self._counter.increment(steps=1, walltime=elapsed)
    if elapsed > 0:
      metrics['steps_per_second'] = self._num_sgd_steps_per_step / elapsed
    else:
      metrics['steps_per_second'] = 0.0

    # Cache last metrics for external logging (e.g., W&B with global step)
    self._last_metrics = {**metrics, **counts}
    self._logger.write(self._last_metrics)

  @property
  def last_metrics(self):
    """Last metrics dict from the most recent step() call."""
    return getattr(self, '_last_metrics', {})

  @property
  def last_transitions(self):
    """Last preprocessed batch of transitions from the most recent step()."""
    return getattr(self, '_last_transitions', None)

  # ---- periodic actor reset (task 0 only) ---------------------------------

  def reset_actor(self, rng_key):
    """Reinitialise the actor with fresh random weights.

    Used during task 0 to escape seed-dependent initialisation traps
    (low weight norms, high dormancy).  The critic is left untouched.

    Steps:
      1. Generate fresh policy params from policy_network.init()
      2. Set policy_base_params = fresh params
      3. Reset v_k to zeros
      4. Reset v_k optimizer state
      5. Reset beta_k and alpha_scale (these are empty at task 0 anyway)

    This is only safe at task 0 because there is no pool, no base to
    preserve, and no cross-task state to corrupt.
    """
    assert self._task_id == 0, (
        'reset_actor is only safe during task 0 (base task).')

    new_policy_params = self._networks.policy_network.init(rng_key)
    new_v_k = _pytree_zeros_like(new_policy_params)
    new_vk_opt = self._vk_optimizer.init(new_v_k)

    self._state = self._state._replace(
        policy_base_params=new_policy_params,
        v_k=new_v_k,
        v_k_optimizer_state=new_vk_opt,
    )
    print(f'  [actor_reset] Actor reinitialised with fresh random weights.',
          flush=True)

  # ---- variable source (for actors) ---------------------------------------

  def get_variables(self, names):
    """Return the combined policy θ' (and current critic) for the actor."""
    if self._actor_cka_path and self._state.actor_cka_state is not None:
      from contrastive.knowledge_pool import compose as cka_compose
      combined = cka_compose(self._state.actor_cka_state)
    else:
      combined = jax.tree.map(
          lambda base, vk: base + vk,
          self._state.policy_base_params,
          self._state.v_k,
      )
    if self._critic_cka_path and self._state.critic_cka_state is not None:
      from contrastive.knowledge_pool import compose as cka_compose
      composed_q = cka_compose(self._state.critic_cka_state)
    else:
      composed_q = self._state.q_params
    variables = {
        'policy': combined,
        'critic': composed_q,
    }
    return [variables[name] for name in names]

  # ---- save / restore for checkpointing -----------------------------------

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state

  # ---- accessors for the orchestrator -------------------------------------

  @property
  def theta_base(self):
    """Frozen base parameters of the actor.

    In CKA mode this is sourced from ``actor_cka_state.base_params``;
    otherwise from the legacy ``policy_base_params`` slot.
    """
    if self._actor_cka_path and self._state.actor_cka_state is not None:
      return self._state.actor_cka_state.base_params
    return self._state.policy_base_params

  @property
  def q_params(self):
    """Effective critic parameters.

    In critic CKA mode the inner loop trains a trainable bundle and the
    composed q params are obtained on the fly. We cache the most recent
    composition lazily so that the orchestrator can checkpoint it.
    """
    if self._critic_cka_path and self._state.critic_cka_state is not None:
      from contrastive.knowledge_pool import compose as cka_compose
      return cka_compose(self._state.critic_cka_state)
    return self._state.q_params

  @property
  def target_q_params(self):
    return self._state.target_q_params

  @property
  def q_optimizer_state(self):
    """Optimizer state for the critic.

    In critic CKA mode the relevant optimizer is
    ``critic_cka_opt_state`` (which trains the bundle
    ``{v_k, alpha_logits, alpha_scale}``); we return that so checkpoints
    capture it.
    """
    if self._critic_cka_path and self._state.critic_cka_opt_state is not None:
      return self._state.critic_cka_opt_state
    return self._state.q_optimizer_state

  @property
  def v_k(self):
    """Per-task knowledge vector for the actor.

    Sourced from ``actor_cka_state.v_k`` in CKA mode.
    """
    if self._actor_cka_path and self._state.actor_cka_state is not None:
      return self._state.actor_cka_state.v_k
    return self._state.v_k

  @property
  def pool(self):
    return self._pool

  @property
  def critic_pool(self):
    return self._critic_pool

  @property
  def w_k_critic(self):
    """Per-task critic knowledge vector w_k.

    Only valid after training when ``critic_mode='cka'``. Sourced from
    ``critic_cka_state.v_k`` post-training.
    """
    assert self._critic_mode == 'cka' and self._q_base is not None, (
        'w_k_critic only available for critic_mode=cka')
    if self._critic_cka_path and self._state.critic_cka_state is not None:
      return self._state.critic_cka_state.v_k
    # Fallback: legacy decomposition q - q_base - 0
    return jax.tree.map(
        lambda q, b: q - b,
        self._state.q_params, self._q_base)
