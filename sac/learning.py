"""Continual goal-conditioned SAC learner (baseline for continual CRL).

#Modified Core SAC math is adapted from JAXGCRL
(https://github.com/MichalBortkiewicz/JaxGCRL), whose SAC agent is in turn
``brax.training.agents.sac`` (see ``sac/reference/jaxgcrl_sac.py`` /
``sac/reference/jaxgcrl_sac_networks.py`` for the reference).  The three
SAC losses, per-step update order, pre-update-α semantics, and soft target
Q update below are a one-to-one port of brax's
``brax.training.agents.sac.losses.make_losses`` composed in JAXGCRL's
``update_step``:

  α-loss:  L_α = E[ α · sg(−log π(a|s) − H_target) ]
  critic:  L_Q = 0.5 · E[ (Q(s,a) − (r · scale + disc · γ · v'))² ]
           v'  = min(Q̄₁, Q̄₂)(s', ã') − α · log π(ã'|s'),  ã' ∼ π(·|s')
  actor:   L_π = E[ α · log π(a|s) − min(Q₁, Q₂)(s,a) ]

The following pieces are *added* on top of JAXGCRL's SAC and are the only
algorithmic deviations from the reference — each is marked with
``#Modified`` below:

  1. HER rewards / terminal discount come from the driver's
     ``flatten_fn`` (reverb pipeline) rather than brax's vectorised
     ``flatten_batch``.  Reward is distance-based:
         r_t = 1[ ||achieved_goal(s_{t+1}) - g_relabeled|| < τ ],
     and ``discount_t = (1-r_t) · γ_env`` zeros the bootstrap on
     goal-reaching transitions.  This matches JAXGCRL's
     ``flatten_batch`` up to the choice of s_t vs s_{t+1} for the
     achieved-goal argument — see :mod:`sac.her` for the justification.
  2. Actor parameters are decomposed CKA-RL style:
         θ_eff = θ_base + Σ_j α_j · v_j + v_k,
     where α = softmax(β_k · α_scale) and only (v_k, β_k, α_scale) are
     trained inside a task.  Only the COMBINED θ_eff reaches the SAC
     losses, so those losses are mathematically unchanged.
  3. Outer-loop β_k / α_scale update uses −min(Q₁, Q₂) as its scalar
     objective (same Q the SAC actor minimises), so the continual
     machinery never sees the contrastive diagonal score.
  4. Optional ``critic_mode`` switch replicates persistent /
     reset / CKA critic handling for a fair comparison with CRL.

Everything else — replay buffer, HER, observation spec, evaluation, W&B
logging — is inherited from the existing continual-CRL pipeline.
"""
import time
from typing import NamedTuple, Optional

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils as acme_utils
from acme.utils import counting
import jax
import jax.numpy as jnp
import optax

from contrastive.knowledge_pool import KnowledgePool, _pytree_zeros_like
from default import make_default_logger
from sac import her


class ContinualSACTrainingState(NamedTuple):
  """Training state for the continual SAC learner."""
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  q_optimizer_state: optax.OptState
  policy_base_params: networks_lib.Params
  v_k: networks_lib.Params
  v_k_optimizer_state: optax.OptState
  beta_k: jnp.ndarray
  beta_k_optimizer_state: optax.OptState
  alpha_scale: jnp.ndarray
  alpha_scale_optimizer_state: optax.OptState
  alpha_params: Optional[jnp.ndarray]           # log_alpha
  alpha_optimizer_state: Optional[optax.OptState]
  key: networks_lib.PRNGKey


class ContinualSACLearner(acme.Learner):
  """Continual SAC+HER learner with CKA-style actor adaptation."""

  _state: ContinualSACTrainingState

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
      config,
      continual_config,
      task_id: int = 0,
      theta_base: Optional[networks_lib.Params] = None,
      pool: Optional[KnowledgePool] = None,
      prev_q_params: Optional[networks_lib.Params] = None,
      prev_target_q_params: Optional[networks_lib.Params] = None,
      prev_q_optimizer_state: Optional[optax.OptState] = None,
      critic_mode: str = 'persistent',
      adapt_heads_only: bool = True,
      encoder_from_base: bool = False,
      q_base: Optional[networks_lib.Params] = None,
      critic_pool: Optional[KnowledgePool] = None,
      step_penalty_reward: bool = True,
  ):
    self._task_id = task_id
    self._critic_mode = critic_mode
    self._adapt_heads_only = adapt_heads_only
    self._encoder_from_base = encoder_from_base
    self._config = config
    self._continual_config = continual_config
    self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
    self._obs_dim = config.obs_dim
    self._step_penalty_reward = bool(step_penalty_reward)

    adaptive_entropy = config.entropy_coefficient is None
    self._adaptive_entropy = adaptive_entropy

    # Match brax/JAXGCRL defaults: reward_scaling=1.0.  Read from
    # ContrastiveConfig.reward_scale (which defaults to 1) so ablations can
    # override it without touching the learner.
    reward_scaling = float(getattr(config, 'reward_scale', 1.0))
    step_penalty_reward = self._step_penalty_reward

    self._pool = pool if pool is not None else KnowledgePool(
        k_max=continual_config.k_max)
    self._q_base = q_base
    self._critic_pool = critic_pool if critic_pool is not None else KnowledgePool(
        k_max=continual_config.k_max)

    # Body-gradient masking (optional; matches CRL semantics).
    self._mask_body_grads = bool(encoder_from_base and task_id > 0)
    mask_body_grads = self._mask_body_grads

    if adaptive_entropy:
      log_alpha_init = jnp.asarray(0.0, dtype=jnp.float32)
      alpha_optimizer_inner = optax.adam(learning_rate=3e-4)
    else:
      log_alpha_init = None
      alpha_optimizer_inner = None
    self._alpha_optimizer_inner = alpha_optimizer_inner

    # ======================================================================
    # Loss functions
    # ======================================================================

    # ======================================================================
    # SAC loss functions — ported from
    #   brax.training.agents.sac.losses.make_losses
    # as used by JAXGCRL (jaxgcrl_sac.py :: update_step, alpha/critic/actor
    # _update).  The only structural change is that the "policy params"
    # supplied to each loss is the COMBINED θ_eff assembled outside the
    # loss (see update_step) — the losses themselves never see the CKA
    # decomposition, so the gradient w.r.t. θ_eff equals the gradient
    # w.r.t. v_k (everything else is a constant).
    # ======================================================================

    def alpha_loss_fn(log_alpha, combined_policy_params, transitions, key):
      # brax: alpha = exp(log_alpha); loss = α · sg(−log π − H_target).
      dist_params = networks.policy_network.apply(
          combined_policy_params, transitions.observation)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)
      alpha = jnp.exp(log_alpha)
      return jnp.mean(
          alpha * jax.lax.stop_gradient(-log_prob - config.target_entropy))

    def critic_loss_fn(q_params, combined_policy_params, target_q_params,
                       alpha, transitions, key):
      """Twin-Q SAC critic loss — mirrors brax/JAXGCRL exactly.

      The only #Modified bits are where reward/discount originate: the
      driver's ``flatten_fn`` (reverb pipeline, adapted from the CRL
      infrastructure) emits
        - transitions.reward   = 1[ ||ag(s_{t+1}) - g_relabeled|| < τ ]
        - transitions.discount = (1 − reward) · γ_env  (0 at goal)
      which plays the role of ``transitions.reward * reward_scaling`` and
      ``transitions.discount`` in brax's ``sac_losses.critic_loss``.  The
      math of the TD target itself is unchanged.
      """
      # next action ã' ∼ π(·|s'), log π(ã'|s')
      next_dist = networks.policy_network.apply(
          combined_policy_params, transitions.next_observation)
      next_action = networks.sample(next_dist, key)
      next_log_prob = networks.log_prob(next_dist, next_action)

      # next_q: twin target Q at (s', ã') — shape [B, 2].
      next_q = networks.q_network.apply(
          target_q_params, transitions.next_observation, next_action)
      # v(s') = min(Q̄₁, Q̄₂)(s', ã') − α · log π(ã'|s')  (brax convention)
      next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob  # [B]

      # brax: target = sg(r · reward_scaling + disc · γ · next_v)
      target = jax.lax.stop_gradient(
          transitions.reward * reward_scaling
          + transitions.discount * config.discount * next_v)

      q_pred = networks.q_network.apply(
          q_params, transitions.observation, transitions.action)
      # brax: q_loss = 0.5 · mean((Q(s,a) − target)²).  Broadcast target
      # against the n_critics=2 heads.
      q_error = q_pred - target[:, None]
      loss = 0.5 * jnp.mean(jnp.square(q_error))
      metrics = {
          'q_mean': jnp.mean(q_pred),
          'q_std': jnp.std(q_pred),
          'target_mean': jnp.mean(target),
          'td_error_abs': jnp.mean(jnp.abs(q_error)),
          'reward_mean': jnp.mean(transitions.reward),
          'her_success_rate': jnp.mean(
              her.reached_from_reward(
                  transitions.reward, step_penalty_reward).astype(jnp.float32)),
          'discount_mean': jnp.mean(transitions.discount),
      }
      return loss, metrics

    def actor_loss_fn(combined_policy_params, q_params, alpha,
                      transitions, key):
      # brax: L_π = E[α · log π(a|s) − min(Q₁, Q₂)(s, a)],  a ∼ π(·|s).
      dist_params = networks.policy_network.apply(
          combined_policy_params, transitions.observation)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)
      q_values = networks.q_network.apply(
          q_params, transitions.observation, action)
      q_min = jnp.min(q_values, axis=-1)  # [B]
      actor_loss = alpha * log_prob - q_min
      metrics = {
          'entropy_mean': jnp.mean(-log_prob),
          'q_pi_mean': jnp.mean(q_min),
      }
      return jnp.mean(actor_loss), metrics

    alpha_grad = jax.value_and_grad(alpha_loss_fn)
    critic_grad = jax.value_and_grad(critic_loss_fn, has_aux=True)
    actor_grad = jax.value_and_grad(actor_loss_fn, has_aux=True)

    def update_step(state, data):
      """One SAC gradient step — mirrors JAXGCRL's ``update_step``.

      Order (α-loss → critic → actor → target Q soft update) and
      pre-update-α / pre-update-Q semantics match
      ``jaxgcrl_sac.py :: update_step`` exactly.  The only #Modified
      step is the assembly of ``combined_policy`` from (θ_base, pool,
      v_k) before any loss is evaluated — the losses themselves see a
      plain policy parameter tree, as in brax.
      """
      transitions, pool_contribution = data
      key, k_alpha, k_critic, k_actor = jax.random.split(state.key, 4)

      # #Modified CKA composition: θ_eff = θ_base + Σ_j α_j v_j + v_k.
      # Everything in (θ_base, pool_contribution) is a constant from the
      # loss's perspective, so ∂L/∂θ_eff = ∂L/∂v_k.
      combined_policy = jax.tree_map(
          lambda b, pc, vk: b + pc + vk,
          state.policy_base_params, pool_contribution, state.v_k)

      # (1) α gradient — uses current (pre-update) policy and α.
      if adaptive_entropy:
        a_loss_val, a_grads = alpha_grad(
            state.alpha_params, combined_policy, transitions, k_alpha)
        # brax: α passed to critic/actor is exp(OLD log_alpha).
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = config.entropy_coefficient
        a_loss_val = 0.0

      # (2) Critic update — uses OLD policy (combined) and OLD target Q.
      (c_loss, c_metrics), c_grads = critic_grad(
          state.q_params, combined_policy, state.target_q_params,
          alpha, transitions, k_critic)
      c_updates, q_opt_state = q_optimizer.update(
          c_grads, state.q_optimizer_state)
      q_params = optax.apply_updates(state.q_params, c_updates)
      # (4) Target Q soft update (brax: τ=0.005 by default).  Done here so
      # the actor step below still sees the PRE-critic-update Q.
      new_target = jax.tree_map(
          lambda x, y: x * (1 - config.tau) + y * config.tau,
          state.target_q_params, q_params)

      # (3) Actor update — brax uses OLD q_params (state.q_params), not the
      # just-updated q_params.  Gradient flows through combined_policy,
      # which is v_k-additive, so updating v_k is equivalent to updating
      # policy_params in brax (see #Modified note above).
      (act_loss, act_metrics), act_grads_combined = actor_grad(
          combined_policy, state.q_params, alpha, transitions, k_actor)
      if mask_body_grads:
        def _mask_leaf(path, g):
          path_str = '/'.join(str(p) for p in path)
          is_head = 'Normal' in path_str
          return g if is_head else jnp.zeros_like(g)
        act_grads_combined = jax.tree_util.tree_map_with_path(
            _mask_leaf, act_grads_combined)
      vk_updates, vk_opt = vk_optimizer.update(
          act_grads_combined, state.v_k_optimizer_state)
      v_k_new = optax.apply_updates(state.v_k, vk_updates)

      metrics = {'critic_loss': c_loss, 'actor_loss': act_loss,
                 **c_metrics, **act_metrics}

      new_state = ContinualSACTrainingState(
          q_params=q_params,
          target_q_params=new_target,
          q_optimizer_state=q_opt_state,
          policy_base_params=state.policy_base_params,
          v_k=v_k_new,
          v_k_optimizer_state=vk_opt,
          beta_k=state.beta_k,
          beta_k_optimizer_state=state.beta_k_optimizer_state,
          alpha_scale=state.alpha_scale,
          alpha_scale_optimizer_state=state.alpha_scale_optimizer_state,
          alpha_params=state.alpha_params,
          alpha_optimizer_state=state.alpha_optimizer_state,
          key=key,
      )

      if adaptive_entropy:
        alpha_up, alpha_opt = alpha_optimizer_inner.update(
            a_grads, state.alpha_optimizer_state)
        new_alpha_params = optax.apply_updates(state.alpha_params, alpha_up)
        metrics['alpha_loss'] = a_loss_val
        metrics['alpha'] = jnp.exp(new_alpha_params)
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_opt,
            alpha_params=new_alpha_params)

      return new_state, metrics

    num_sgd = config.num_sgd_steps_per_step

    def _scan_update(state, data):
      transitions, pool_c = data
      batched = jax.tree_map(
          lambda a: jnp.reshape(a, (num_sgd, -1, *a.shape[1:])),
          transitions)

      def scan_body(carry, mini_batch):
        return update_step(carry, (mini_batch, pool_c))

      state, metrics = jax.lax.scan(
          scan_body, state, batched, length=num_sgd)
      metrics = jax.tree_map(jnp.mean, metrics)
      return state, metrics

    self._update_step = (
        jax.jit(_scan_update) if config.jit else _scan_update)

    # ======================================================================
    # Initial training state
    # ======================================================================
    key_policy, key_q, rng = jax.random.split(rng, 3)

    if theta_base is None:
      policy_params = networks.policy_network.init(key_policy)
      theta_base = policy_params

      if prev_q_params is not None and critic_mode == 'persistent':
        q_params = prev_q_params
      elif critic_mode == 'cka' and self._q_base is not None:
        critic_pool_c = self._compute_critic_pool_contribution()
        q_params = jax.tree_map(
            lambda b, pc: b + pc, self._q_base, critic_pool_c)
        self._critic_pool_c_at_init = critic_pool_c
      else:
        q_params = networks.q_network.init(key_q)
    else:
      policy_params = theta_base  # not used directly (composed via v_k)
      if critic_mode == 'persistent':
        assert prev_q_params is not None
        q_params = prev_q_params
      elif critic_mode == 'reset':
        q_params = networks.q_network.init(key_q)
      elif critic_mode == 'cka':
        assert self._q_base is not None, (
            'critic_mode=cka requires q_base (frozen from task 0)')
        critic_pool_c = self._compute_critic_pool_contribution()
        q_params = jax.tree_map(
            lambda b, pc: b + pc, self._q_base, critic_pool_c)
        self._critic_pool_c_at_init = critic_pool_c
      else:
        raise ValueError(f'Unknown critic_mode: {critic_mode}')

    v_k = _pytree_zeros_like(theta_base)

    n_pool = len(self._pool)
    if n_pool > 0:
      key_beta, rng = jax.random.split(rng)
      beta_k = (jax.random.normal(key_beta, (n_pool,))
                * continual_config.beta_init_std)
    else:
      beta_k = jnp.zeros(0)
    alpha_scale = jnp.ones(1)

    vk_opt_state = vk_optimizer.init(v_k)
    beta_opt_state = beta_optimizer.init(beta_k)
    alpha_scale_opt_state = alpha_scale_optimizer.init(alpha_scale)

    critic_fresh = (
        task_id == 0
        or critic_mode in ('reset', 'cka')
        or prev_q_optimizer_state is None)

    if critic_fresh:
      q_opt_state = q_optimizer.init(q_params)
      target_q = q_params
    else:
      q_opt_state = prev_q_optimizer_state
      target_q = (prev_target_q_params
                  if prev_target_q_params is not None else q_params)

    if adaptive_entropy:
      alpha_params_init = log_alpha_init
      alpha_opt_state_init = alpha_optimizer_inner.init(alpha_params_init)
    else:
      alpha_params_init = None
      alpha_opt_state_init = None

    self._state = ContinualSACTrainingState(
        q_params=q_params, target_q_params=target_q,
        q_optimizer_state=q_opt_state,
        policy_base_params=theta_base,
        v_k=v_k, v_k_optimizer_state=vk_opt_state,
        beta_k=beta_k, beta_k_optimizer_state=beta_opt_state,
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

    self._counter = counter or counting.Counter()
    self._logger = logger or make_default_logger(
        'learner', asynchronous=True, serialize_fn=acme_utils.fetch_devicearray,
        time_delta=10.0)
    self._iterator = iterator
    self._timestamp = None

  # ----------------------------------------------------------------------
  # Pool-contribution helpers (outside JIT because pool length varies).
  # ----------------------------------------------------------------------

  def _compute_pool_contribution(self):
    pool_vecs = self._pool.get_vectors()
    if not pool_vecs:
      return _pytree_zeros_like(self._state.policy_base_params)
    beta = self._state.beta_k
    alpha_scale = self._state.alpha_scale
    alpha = jax.nn.softmax(beta * alpha_scale[0])
    contribution = _pytree_zeros_like(pool_vecs[0])
    for j, v_j in enumerate(pool_vecs):
      contribution = jax.tree_map(
          lambda c, v: c + float(alpha[j]) * v, contribution, v_j)
    return contribution

  def _compute_critic_pool_contribution(self):
    vecs = self._critic_pool.get_vectors()
    if not vecs:
      ref = self._q_base if self._q_base is not None else getattr(
          getattr(self, '_state', None), 'q_params', None)
      return _pytree_zeros_like(ref)
    n = len(vecs)
    res = _pytree_zeros_like(vecs[0])
    for w_j in vecs:
      res = jax.tree_map(lambda r, w: r + w / n, res, w_j)
    return res

  # ----------------------------------------------------------------------
  # β_k / α_scale outer-loop gradient step.
  # ----------------------------------------------------------------------
  def _update_beta_and_alpha_scale(self, transitions):
    pool_vecs = self._pool.get_vectors()
    if not pool_vecs or len(self._state.beta_k) == 0:
      return

    def beta_loss_fn(beta_k, alpha_scale_val):
      alpha = jax.nn.softmax(beta_k * alpha_scale_val)
      contribution = _pytree_zeros_like(pool_vecs[0])
      for j, v_j in enumerate(pool_vecs):
        contribution = jax.tree_map(
            lambda c, v: c + alpha[j] * v, contribution, v_j)
      combined = jax.tree_map(
          lambda b, pc, vk: b + pc + vk,
          self._state.policy_base_params, contribution, self._state.v_k)
      key = jax.random.PRNGKey(0)
      dist = self._networks.policy_network.apply(
          combined, transitions.observation)
      action = self._networks.sample(dist, key)
      # #Modified Objective for β_k is now −min(Q₁, Q₂) rather than the
      # contrastive diagonal score.
      q = self._networks.q_network.apply(
          self._state.q_params, transitions.observation, action)
      return -jnp.mean(jnp.min(q, axis=-1))

    grad_fn = jax.grad(beta_loss_fn, argnums=(0, 1))
    beta_grad, as_grad = grad_fn(
        self._state.beta_k, self._state.alpha_scale[0])
    beta_up, beta_opt = self._beta_optimizer.update(
        beta_grad, self._state.beta_k_optimizer_state)
    new_beta = optax.apply_updates(self._state.beta_k, beta_up)
    as_up, as_opt = self._alpha_scale_optimizer.update(
        jnp.reshape(as_grad, (1,)),
        self._state.alpha_scale_optimizer_state)
    new_as = optax.apply_updates(self._state.alpha_scale, as_up)
    self._state = self._state._replace(
        beta_k=new_beta, beta_k_optimizer_state=beta_opt,
        alpha_scale=new_as, alpha_scale_optimizer_state=as_opt)

  # ----------------------------------------------------------------------
  # Learner step
  # ----------------------------------------------------------------------
  def step(self):
    with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
      sample = next(self._iterator)
      transitions = types.Transition(*sample.data)
      self._last_transitions = transitions
      pool_c = self._compute_pool_contribution()
      self._state, metrics = self._update_step(
          self._state, (transitions, pool_c))
      if self._task_id > 0 and len(self._pool) > 0:
        bs = self._config.batch_size
        n = self._config.num_sgd_steps_per_step
        last_start = (n - 1) * bs
        last_t = jax.tree_map(
            lambda t: t[last_start:last_start + bs], transitions)
        self._update_beta_and_alpha_scale(last_t)

    t = time.time()
    elapsed = t - self._timestamp if self._timestamp else 0
    self._timestamp = t
    counts = self._counter.increment(steps=1, walltime=elapsed)
    metrics['steps_per_second'] = (
        self._num_sgd_steps_per_step / elapsed if elapsed > 0 else 0.0)
    if len(self._state.beta_k) > 0:
      alpha = jax.nn.softmax(
          self._state.beta_k * self._state.alpha_scale[0])
      metrics['alpha_weights_max'] = float(jnp.max(alpha))
      metrics['alpha_scale'] = float(self._state.alpha_scale[0])

    self._last_metrics = {**metrics, **counts}
    self._logger.write(self._last_metrics)

  @property
  def last_metrics(self):
    return getattr(self, '_last_metrics', {})

  @property
  def last_transitions(self):
    return getattr(self, '_last_transitions', None)

  # ----------------------------------------------------------------------
  # Optional actor reset (task 0 only; same contract as CRL).
  # ----------------------------------------------------------------------
  def reset_actor(self, rng_key):
    assert self._task_id == 0, (
        'reset_actor is only safe during task 0 (base task).')
    new_policy_params = self._networks.policy_network.init(rng_key)
    new_v_k = _pytree_zeros_like(new_policy_params)
    self._state = self._state._replace(
        policy_base_params=new_policy_params,
        v_k=new_v_k,
        v_k_optimizer_state=self._vk_optimizer.init(new_v_k))
    print('  [actor_reset] Actor reinitialised with fresh random weights.',
          flush=True)

  # ----------------------------------------------------------------------
  # Variable source / checkpoint accessors — identical to CRL so the rest
  # of the driver (orchestrator, evaluator, checkpoint I/O) is unchanged.
  # ----------------------------------------------------------------------
  def get_variables(self, names):
    pool_c = self._compute_pool_contribution()
    combined = jax.tree_map(
        lambda b, pc, vk: b + pc + vk,
        self._state.policy_base_params, pool_c, self._state.v_k)
    table = {'policy': combined, 'critic': self._state.q_params}
    return [table[n] for n in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state

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
    """Extract critic knowledge vector w_k = q_params − q_base − pool_c."""
    assert self._critic_mode == 'cka' and self._q_base is not None, (
        'w_k_critic only available for critic_mode=cka')
    pool_c = getattr(self, '_critic_pool_c_at_init',
                     self._compute_critic_pool_contribution())
    return jax.tree_map(
        lambda q, b, pc: q - b - pc,
        self._state.q_params, self._q_base, pool_c)
