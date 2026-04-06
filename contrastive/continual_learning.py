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
from contrastive.knowledge_pool import (
    KnowledgePool,
    compose_policy_params,
    _pytree_zeros_like,
    _flatten_pytree,
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
  """Training state for the continual contrastive RL learner."""
  # Critic (persistent across tasks)
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  q_optimizer_state: optax.OptState

  # Actor – base (frozen after task 1)
  policy_base_params: networks_lib.Params

  # Actor – current task vector v_k (optimised)
  v_k: networks_lib.Params
  v_k_optimizer_state: optax.OptState

  # Actor – CKA blending weights β_k (optimised); α_k = softmax(β_k)
  beta_k: jnp.ndarray
  beta_k_optimizer_state: optax.OptState

  # Optional α_scale (learnable scalar)
  alpha_scale: jnp.ndarray
  alpha_scale_optimizer_state: optax.OptState

  # Entropy temperature (adaptive)
  alpha_params: Optional[jnp.ndarray]          # log_alpha
  alpha_optimizer_state: Optional[optax.OptState]

  # RNG key
  key: networks_lib.PRNGKey

  # Frozen pool vectors (as a list serialised to a tuple of pytrees)
  # NOTE: we keep pool_vectors outside this NamedTuple because NamedTuples
  # must hold fixed-structure elements and the pool length changes.  We store
  # a placeholder here; the learner class holds the actual pool.


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
      adapt_heads_only: bool = True,
      encoder_from_base: bool = True,
      # --- critic CKA state (only used when critic_mode='cka') ---
      q_base: Optional[networks_lib.Params] = None,
      critic_pool: Optional[KnowledgePool] = None,
  ):
    self._task_id = task_id
    self._critic_mode = critic_mode
    self._adapt_heads_only = adapt_heads_only
    self._encoder_from_base = encoder_from_base
    self._config = config
    self._continual_config = continual_config
    self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
    self._obs_dim = config.obs_dim

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
                         transitions, key):
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

        logits, _, _ = networks.q_network.apply(
            q_params, transitions.observation, transitions.action)

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
          def loss_fn(_logits):
            if config.use_cpc:
              return (optax.softmax_cross_entropy(logits=_logits, labels=I)
                      + config.logsumexp_penalty * jax.nn.logsumexp(_logits, axis=1)**2)
            else:
              return optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I)
          if len(logits.shape) == 3:
            loss = jax.vmap(loss_fn, in_axes=2, out_axes=-1)(logits)
            loss = jnp.mean(loss, axis=-1)
            logits = jnp.mean(logits, axis=-1)
          else:
            loss = loss_fn(logits)

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
          data: tuple of (transitions, pool_contribution)
            transitions: batch of transitions
            pool_contribution: pytree – pre-computed Σ α_j v_j (using current
              α derived from state.beta_k outside JIT).
        """
        transitions, pool_contribution = data
        key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)

        # -- compose effective policy params --------------------------------
        combined_policy = jax.tree_map(
            lambda base, pool_c, vk: base + pool_c + vk,
            state.policy_base_params,
            pool_contribution,
            state.v_k,
        )

        # -- entropy coefficient --------------------------------------------
        if adaptive_entropy:
          _alpha_loss, _alpha_grads = alpha_grad(
              state.alpha_params, combined_policy, transitions, key_alpha)
          alpha = jnp.exp(state.alpha_params)
        else:
          alpha = config.entropy_coefficient
          _alpha_loss = 0.0

        # -- critic update --------------------------------------------------
        (c_loss, c_metrics), c_grads = critic_grad(
            state.q_params, combined_policy, state.target_q_params,
            transitions, key_critic)
        c_updates, q_opt_state = q_optimizer.update(c_grads, state.q_optimizer_state)
        q_params = optax.apply_updates(state.q_params, c_updates)
        new_target = jax.tree_map(
            lambda x, y: x * (1 - config.tau) + y * config.tau,
            state.target_q_params, q_params)

        # -- actor update (gradients w.r.t. v_k) ----------------------------
        # We need grad of actor_loss w.r.t. combined_policy, then chain-rule
        # to v_k.  Since combined_policy = base + pool_c + v_k, the grad
        # w.r.t. v_k equals the grad w.r.t. combined_policy (base and pool_c
        # are treated as constants).
        actor_loss_and_grad = jax.value_and_grad(actor_loss_fn, has_aux=True)
        (a_loss, a_metrics), a_grads_combined = actor_loss_and_grad(
            combined_policy, state.q_params, alpha, transitions, key_actor)
        # a_grads_combined has the same pytree structure as combined_policy
        # and equals ∂L/∂v_k because v_k enters additively.
        #
        # Head-only adaptation: zero out body gradients so v_k only modifies
        # the actor output head (NormalTanhDistribution layers).
        if mask_body_grads:
          def _mask_leaf(path, g):
            # Haiku key path is a tuple of DictKey / GetAttrKey objects.
            # Head leaves have 'normal_tanh_distribution' in their path.
            path_str = '/'.join(str(p) for p in path)
            is_head = 'normal_tanh_distribution' in path_str
            return g if is_head else jnp.zeros_like(g)
          a_grads_combined = jax.tree_util.tree_map_with_path(
              _mask_leaf, a_grads_combined)
        vk_updates, vk_opt_state = vk_optimizer.update(
            a_grads_combined, state.v_k_optimizer_state)
        v_k_new = optax.apply_updates(state.v_k, vk_updates)

        # -- β_k update (not needed for task 0 / base phase) ----------------
        # β_k affects α = softmax(β_k * α_scale) which modulates pool_contribution.
        # Since pool_contribution is pre-computed outside JIT (for variable-length
        # pool), β_k gradients are computed by the orchestrator's outer loop.
        # Here we just pass through.
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

      data = (transitions, pool_contribution)
        transitions: leaves have shape [batch_size * num_sgd, ...]
        pool_contribution: param-shaped pytree (no batch dim)
      """
      transitions, pool_c = data
      # Reshape transitions: [B*N, ...] -> [N, B, ...]
      batched_transitions = jax.tree_map(
          lambda a: jnp.reshape(a, (num_sgd, -1, *a.shape[1:])),
          transitions)

      def scan_body(carry, mini_batch):
        # mini_batch is transitions for one SGD step; pool_c is closed over.
        return update_step(carry, (mini_batch, pool_c))

      state, metrics = jax.lax.scan(
          scan_body, state, batched_transitions, length=num_sgd)
      # Average metrics across SGD steps (same as process_multiple_batches)
      metrics = jax.tree_map(jnp.mean, metrics)
      return state, metrics

    if config.jit:
      self._update_step = jax.jit(_scan_update)
    else:
      self._update_step = _scan_update

    # ---- initialise state -------------------------------------------------
    key_policy, key_q, rng = jax.random.split(rng, 3)

    if theta_base is None:
      # Fresh init (base task or reset_actor mode)
      policy_params = networks.policy_network.init(key_policy)
      q_params = networks.q_network.init(key_q)
      theta_base = policy_params
    else:
      # Continual phase
      policy_params = theta_base  # not used directly; composed via v_k

      if critic_mode == 'persistent':
        # Carry forward critic from previous task (never reset)
        assert prev_q_params is not None
        q_params = prev_q_params
      elif critic_mode == 'reset':
        # Reinitialize critic from scratch each task
        q_params = networks.q_network.init(key_q)
      elif critic_mode == 'cka':
        # CKA-style critic: q' = q_base + Σ α_j w_j + w_k
        # q_base is frozen from task 0.  We compose the initial q_params
        # by adding the pool contribution and a fresh w_k (zeros).
        # The inner loop trains q_params normally; after training we
        # extract w_k = q_params - q_base - pool_c.
        assert self._q_base is not None, (
            'critic_mode=cka requires q_base (frozen from task 0)')
        # Compute pool contribution for critic
        critic_pool_c = self._compute_critic_pool_contribution()
        # w_k starts at zero → composed q_params = q_base + pool_c + 0
        q_params = jax.tree_map(
            lambda b, pc: b + pc,
            self._q_base, critic_pool_c)
        # Store pool_c for w_k extraction after training
        self._critic_pool_c_at_init = critic_pool_c
      else:
        raise ValueError(f'Unknown critic_mode: {critic_mode}')

    # v_k: initialised to zeros (same structure as policy params)
    v_k = _pytree_zeros_like(theta_base)

    # β_k: for task 0, empty; for task k, length = |pool|
    n_pool = len(self._pool)
    if n_pool > 0:
      key_beta, rng = jax.random.split(rng)
      beta_k = jax.random.normal(key_beta, (n_pool,)) * continual_config.beta_init_std
    else:
      beta_k = jnp.zeros(0)

    alpha_scale = jnp.ones(1)

    # Optimisers
    vk_opt_state = vk_optimizer.init(v_k)
    beta_opt_state = beta_optimizer.init(beta_k)
    alpha_scale_opt_state = alpha_scale_optimizer.init(alpha_scale)

    if critic_mode in ('reset', 'cka') and task_id > 0:
      # reset: fresh init; cka: q_params recomposed, old opt state invalid
      q_opt_state = q_optimizer.init(q_params)
    elif prev_q_optimizer_state is not None and task_id > 0:
      q_opt_state = prev_q_optimizer_state
    else:
      q_opt_state = q_optimizer.init(q_params)

    if critic_mode in ('reset', 'cka') and task_id > 0:
      # For both reset and cka, target starts from the (re)composed q_params
      target_q = q_params
    else:
      target_q = prev_target_q_params if prev_target_q_params is not None else q_params

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

  # ---- pool-contribution helper (outside JIT for variable-length pool) ----

  def _compute_pool_contribution(self):
    """Compute Σ α_j v_j using current β_k and pool vectors."""
    pool_vecs = self._pool.get_vectors()
    if not pool_vecs:
      return _pytree_zeros_like(self._state.policy_base_params)

    beta = self._state.beta_k
    alpha_scale = self._state.alpha_scale
    alpha = jax.nn.softmax(beta * alpha_scale[0])

    contribution = _pytree_zeros_like(pool_vecs[0])
    for j, v_j in enumerate(pool_vecs):
      contribution = jax.tree_map(
          lambda c, v: c + float(alpha[j]) * v,
          contribution, v_j)
    return contribution

  # ---- critic pool contribution (outside JIT) ------------------------------

  def _compute_critic_pool_contribution(self):
    """Compute Σ α_j w_j for critic pool using uniform weights.

    For the critic CKA, we use uniform blending (no learnable β for critic)
    to keep the implementation simple: pool_c = mean(w_j) if pool non-empty.
    """
    critic_vecs = self._critic_pool.get_vectors()
    if not critic_vecs:
      # Need a zeros-like for critic params.  Use q_base if available,
      # otherwise fall back to current q_params.
      ref = self._q_base if self._q_base is not None else self._state.q_params
      return _pytree_zeros_like(ref)
    # Uniform average of critic knowledge vectors
    n = len(critic_vecs)
    result = _pytree_zeros_like(critic_vecs[0])
    for w_j in critic_vecs:
      result = jax.tree_map(lambda r, w: r + w / n, result, w_j)
    return result

  # ---- β_k gradient step (outside JIT) ------------------------------------

  def _update_beta_and_alpha_scale(self, transitions):
    """Compute and apply gradients for β_k and α_scale.

    Because the pool length varies, we compute these gradients outside JAX JIT
    using a small helper that evaluates the actor loss as a function of β_k.
    For simplicity in the initial implementation, we use finite differences
    or a non-JIT grad.  A more efficient approach would fix the pool size with
    padding, but this is cleaner for correctness.
    """
    pool_vecs = self._pool.get_vectors()
    if not pool_vecs or len(self._state.beta_k) == 0:
      return  # nothing to update for base task or empty pool

    # Define actor loss as function of beta_k (and alpha_scale)
    def beta_loss_fn(beta_k, alpha_scale_val):
      alpha = jax.nn.softmax(beta_k * alpha_scale_val)
      # Compose pool contribution
      contribution = _pytree_zeros_like(pool_vecs[0])
      for j, v_j in enumerate(pool_vecs):
        contribution = jax.tree_map(
            lambda c, v: c + alpha[j] * v,
            contribution, v_j)
      # Compose policy
      combined = jax.tree_map(
          lambda base, pc, vk: base + pc + vk,
          self._state.policy_base_params,
          contribution,
          self._state.v_k,
      )
      # Evaluate actor loss using inner product (matching SGCRL)
      obs = transitions.observation
      state = obs[:, :self._config.obs_dim]
      goal = obs[:, self._config.obs_dim:]
      new_obs = jnp.concatenate([state, goal], axis=1)
      key = jax.random.PRNGKey(0)  # deterministic for gradient
      dist_params = self._networks.policy_network.apply(combined, new_obs)
      action = self._networks.sample(dist_params, key)
      q_action, _, _ = self._networks.q_network.apply(
          self._state.q_params, new_obs, action)
      if len(q_action.shape) == 3:
        q_action = jnp.min(q_action, axis=-1)
      return -jnp.mean(jnp.diag(q_action))  # minimize neg Q = maximize Q

    # Compute gradients
    grad_fn = jax.grad(beta_loss_fn, argnums=(0, 1))
    beta_grad, alpha_scale_grad_scalar = grad_fn(
        self._state.beta_k, self._state.alpha_scale[0])

    # Apply updates for beta_k
    beta_updates, beta_opt = self._beta_optimizer.update(
        beta_grad, self._state.beta_k_optimizer_state)
    new_beta = optax.apply_updates(self._state.beta_k, beta_updates)

    # alpha_scale is shape (1,) but grad is scalar – reshape to match
    alpha_scale_grad = jnp.reshape(alpha_scale_grad_scalar, (1,))
    as_updates, as_opt = self._alpha_scale_optimizer.update(
        alpha_scale_grad, self._state.alpha_scale_optimizer_state)
    new_alpha_scale = optax.apply_updates(self._state.alpha_scale, as_updates)

    self._state = self._state._replace(
        beta_k=new_beta,
        beta_k_optimizer_state=beta_opt,
        alpha_scale=new_alpha_scale,
        alpha_scale_optimizer_state=as_opt,
    )

  # ---- main step ----------------------------------------------------------

  def step(self):
    with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
      sample = next(self._iterator)
      transitions = types.Transition(*sample.data)

      # Compute pool contribution (outside JIT for variable-length pool)
      pool_c = self._compute_pool_contribution()

      # Single call: lax.scan handles the num_sgd_steps_per_step inner loop
      self._state, metrics = self._update_step(
          self._state, (transitions, pool_c))

      # Update β_k and α_scale (outside JIT, once per learner step)
      if self._task_id > 0 and len(self._pool) > 0:
        # Use last mini-batch for β gradient
        batch_size = self._config.batch_size
        n_steps = self._config.num_sgd_steps_per_step
        last_start = (n_steps - 1) * batch_size
        last_transitions = jax.tree_map(
            lambda t: t[last_start:last_start + batch_size], transitions)
        self._update_beta_and_alpha_scale(last_transitions)

    # Timing
    timestamp = time.time()
    elapsed = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    counts = self._counter.increment(steps=1, walltime=elapsed)
    if elapsed > 0:
      metrics['steps_per_second'] = self._num_sgd_steps_per_step / elapsed
    else:
      metrics['steps_per_second'] = 0.0

    # Add continual-specific metrics
    if len(self._state.beta_k) > 0:
      alpha = jax.nn.softmax(
          self._state.beta_k * self._state.alpha_scale[0])
      metrics['alpha_weights'] = float(jnp.max(alpha))
      metrics['alpha_scale'] = float(self._state.alpha_scale[0])

    self._logger.write({**metrics, **counts})

  # ---- variable source (for actors) ---------------------------------------

  def get_variables(self, names):
    """Return the combined policy θ' for the actor."""
    pool_c = self._compute_pool_contribution()
    combined = jax.tree_map(
        lambda base, pc, vk: base + pc + vk,
        self._state.policy_base_params,
        pool_c,
        self._state.v_k,
    )
    variables = {
        'policy': combined,
        'critic': self._state.q_params,
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
    return self._state.policy_base_params

  @property
  def q_params(self):
    return self._state.q_params

  @property
  def target_q_params(self):
    return self._state.target_q_params

  @property
  def q_optimizer_state(self):
    return self._state.q_optimizer_state

  @property
  def v_k(self):
    return self._state.v_k

  @property
  def pool(self):
    return self._pool

  @property
  def critic_pool(self):
    return self._critic_pool

  @property
  def w_k_critic(self):
    """Extract critic knowledge vector: w_k = q_params - q_base - pool_c.

    Only valid after training when critic_mode='cka'.
    """
    assert self._critic_mode == 'cka' and self._q_base is not None, (
        'w_k_critic only available for critic_mode=cka')
    pool_c = getattr(self, '_critic_pool_c_at_init',
                     self._compute_critic_pool_contribution())
    return jax.tree_map(
        lambda q, b, pc: q - b - pc,
        self._state.q_params, self._q_base, pool_c)
