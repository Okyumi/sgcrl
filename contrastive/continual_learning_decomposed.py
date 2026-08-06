"""Continual contrastive RL learner with decomposed critic (proposal 1).

Sibling to ``ContinualContrastiveLearner``. Forked deliberately so the
existing CKA / persistent / reset paths are not perturbed: every line of
the existing learner is preserved; this file is what the orchestrator
reaches for when ``critic_mode='decomposed'``.

Algorithm (matches ``docs/2026-05-08_plan_proposal1_dyn_aux.md``):

  phi(s, a)  = h_phi(b_shared([s; a])) + phi_task([s; a])
  psi(g)     = psi(g)
  score      = phi(s, a) @ psi(g).T            # SGCRL inner product (default)
             or -|| phi(s, a) - psi(g) ||_2    # if config.energy_fn == 'l2'

  L_InfoNCE  = standard InfoNCE on the (B, B) score matrix, in the same
               form as the existing critic: sigmoid-BCE (use_cpc=False,
               default) or softmax-CE+logsumexp (use_cpc=True)
  L_dyn      = || h_dyn(b_shared(s, a)) - select_stable(s') ||_2^2

Trainable groups (each has its own optimiser state):

  b_shared     : sees L_InfoNCE + dyn_aux_weight * L_dyn
  h_phi        : sees L_InfoNCE
  h_dyn        : sees dyn_aux_weight * L_dyn
  phi_task     : sees L_InfoNCE                  (reset every task)
  psi          : sees L_InfoNCE
  actor params : sees actor objective only       (reset every task)

Continual handoff at task k > 0:

  b_shared, h_phi, h_dyn, psi, q_optimizer states for them   -> carry over
  phi_task params + opt state                                 -> reinit
  actor params + opt state                                    -> reinit (reset)

The actor objective matches the existing SGCRL learner: optionally roll
goals via ``config.random_goals`` (0.0 / 0.5 / 1.0), then maximise the
critic's diagonal score under the composed phi, plus an entropy bonus
when ``config.use_action_entropy=True``.

Excluded from this learner intentionally (out of scope for proposal 1):
  - TD critic path (``config.use_td=False`` is required)
  - Twin Q
  - Negative bank
  - Image observations
  - Actor CKA decomposition (use ``actor_mode='reset'``)

The orchestrator should fall back to ``ContinualContrastiveLearner`` for
any cell that needs the above features.
"""
from __future__ import annotations

import time
from typing import Any, Dict, NamedTuple, Optional, Tuple

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb

from contrastive import config as contrastive_config
from contrastive import decomposed_networks
from contrastive import shortcut_diagnostics
from contrastive import state_mask as sm
from default import make_default_logger


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

class DecomposedTrainingState(NamedTuple):
  """All parameters and optimiser states for the decomposed-critic learner.

  Five critic parameter groups (b_shared, h_phi, h_dyn, phi_task, psi)
  plus the actor (policy_params) and the adaptive entropy temperature.
  """
  # Actor (reset every task)
  policy_params: networks_lib.Params
  policy_opt_state: optax.OptState

  # Critic — shared body with two heads
  b_shared_params: networks_lib.Params
  b_shared_opt_state: optax.OptState
  h_phi_params: networks_lib.Params
  h_phi_opt_state: optax.OptState
  h_dyn_params: networks_lib.Params
  h_dyn_opt_state: optax.OptState

  # Critic — task-specific (reset every task)
  phi_task_params: networks_lib.Params
  phi_task_opt_state: optax.OptState

  # Critic — goal encoder
  psi_params: networks_lib.Params
  psi_opt_state: optax.OptState

  # Adaptive entropy
  alpha_params: Optional[jnp.ndarray]            # log_alpha
  alpha_optimizer_state: Optional[optax.OptState]

  # RNG
  key: networks_lib.PRNGKey


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------

class ContinualDecomposedLearner(acme.Learner):
  """Continual contrastive learner with the decomposed critic.

  Same outer interface as ``ContinualContrastiveLearner`` so the
  orchestrator's checkpoint / step / get_variables calls continue to
  work, just with the new state shape.
  """

  _state: DecomposedTrainingState

  def __init__(
      self,
      *,
      decomp_nets: decomposed_networks.DecomposedCriticNetworks,
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
      # State carried in from previous task (decomposed components only).
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
          'critic_mode=decomposed does not support config.use_td=True. '
          'Set use_td=False or use a different critic_mode.')
    if getattr(config, 'twin_q', False):
      raise ValueError('critic_mode=decomposed does not support twin_q.')

    self._iterator = iterator
    self._counter = counter or counting.Counter()
    self._logger = logger or make_default_logger('learner')
    self._config = config
    self._continual_cfg = continual_config
    self._decomp_nets = decomp_nets
    self._policy_network = policy_network
    self._sample_fn = sample_fn
    self._log_prob_fn = log_prob_fn
    self._task_id = task_id
    self._timestamp: Optional[float] = None
    self._last_metrics: Dict[str, float] = {}

    # SGCRL convention: adaptive entropy is on iff entropy_coefficient is
    # None, mirroring continual_learning.py:170.
    self._adaptive_entropy = (config.entropy_coefficient is None)
    if not self._adaptive_entropy:
      raise ValueError(
          'critic_mode=decomposed requires adaptive entropy '
          '(config.entropy_coefficient=None). The project default is '
          'adaptive with target_entropy=-2.0; do not set '
          'entropy_coefficient if you want to keep that behaviour.')
    self._dyn_aux_weight = float(
        getattr(continual_config, 'dyn_aux_weight', 1.0))

    # Per-component optimisers. Same learning rate as the legacy critic
    # to keep the comparison fair in the first cell. Adam betas left at
    # optax defaults.
    lr = config.learning_rate
    self._b_shared_opt = optax.adam(lr)
    self._h_phi_opt = optax.adam(lr)
    self._h_dyn_opt = optax.adam(lr)
    self._phi_task_opt = optax.adam(lr)
    self._psi_opt = optax.adam(lr)
    self._actor_opt = optax.adam(lr)
    self._alpha_opt = optax.adam(lr) if self._adaptive_entropy else None

    # Initialise state.
    rng, *subkeys = jax.random.split(rng, 8)
    if (task_id == 0 or prev_b_shared_params is None
        or prev_h_phi_params is None or prev_h_dyn_params is None
        or prev_psi_params is None):
      b_shared_params = decomp_nets.init_b_shared(subkeys[0])
      h_phi_params = decomp_nets.init_h_phi(subkeys[1])
      h_dyn_params = decomp_nets.init_h_dyn(subkeys[2])
      psi_params = decomp_nets.init_psi(subkeys[3])
      b_shared_opt_state = self._b_shared_opt.init(b_shared_params)
      h_phi_opt_state = self._h_phi_opt.init(h_phi_params)
      h_dyn_opt_state = self._h_dyn_opt.init(h_dyn_params)
      psi_opt_state = self._psi_opt.init(psi_params)
    else:
      # Carry forward shared components.
      b_shared_params = prev_b_shared_params
      h_phi_params = prev_h_phi_params
      h_dyn_params = prev_h_dyn_params
      psi_params = prev_psi_params
      b_shared_opt_state = (
          prev_b_shared_opt_state if prev_b_shared_opt_state is not None
          else self._b_shared_opt.init(b_shared_params))
      h_phi_opt_state = (
          prev_h_phi_opt_state if prev_h_phi_opt_state is not None
          else self._h_phi_opt.init(h_phi_params))
      h_dyn_opt_state = (
          prev_h_dyn_opt_state if prev_h_dyn_opt_state is not None
          else self._h_dyn_opt.init(h_dyn_params))
      psi_opt_state = (
          prev_psi_opt_state if prev_psi_opt_state is not None
          else self._psi_opt.init(psi_params))

    # Task-specific encoder + actor reinitialised every task.
    phi_task_params = decomp_nets.init_phi_task(subkeys[4])
    phi_task_opt_state = self._phi_task_opt.init(phi_task_params)
    policy_params = policy_network.init(subkeys[5])
    policy_opt_state = self._actor_opt.init(policy_params)

    if self._adaptive_entropy:
      alpha_params = jnp.array(log_alpha_init, dtype=jnp.float32)
      alpha_opt_state = self._alpha_opt.init(alpha_params)
    else:
      alpha_params = None
      alpha_opt_state = None

    self._state = DecomposedTrainingState(
        policy_params=policy_params,
        policy_opt_state=policy_opt_state,
        b_shared_params=b_shared_params,
        b_shared_opt_state=b_shared_opt_state,
        h_phi_params=h_phi_params,
        h_phi_opt_state=h_phi_opt_state,
        h_dyn_params=h_dyn_params,
        h_dyn_opt_state=h_dyn_opt_state,
        phi_task_params=phi_task_params,
        phi_task_opt_state=phi_task_opt_state,
        psi_params=psi_params,
        psi_opt_state=psi_opt_state,
        alpha_params=alpha_params,
        alpha_optimizer_state=alpha_opt_state,
        key=subkeys[6],
    )

    self._update_step = self._make_update_step()

    # Optional task-5/task-8 shortcut diagnostics.  The default interval of
    # zero preserves the legacy DCC hot path exactly.
    self._diagnostic_interval = int(getattr(
        continual_config, 'shortcut_diagnostic_interval', 0))
    self._diagnostic_counter = 0
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
          q_network=None,
      )

  # ------------------------------------------------------------------
  # JIT-compiled update step
  # ------------------------------------------------------------------

  def _make_update_step(self):
    config = self._config
    decomp_nets = self._decomp_nets
    policy_network = self._policy_network
    sample_fn = self._sample_fn
    log_prob_fn = self._log_prob_fn
    adaptive_entropy = self._adaptive_entropy
    target_entropy = config.target_entropy
    logsumexp_penalty_coeff = float(getattr(config, 'logsumexp_penalty', 0.0))
    use_cpc = bool(getattr(config, 'use_cpc', False))
    use_action_entropy = bool(getattr(config, 'use_action_entropy', True))
    random_goals = float(getattr(config, 'random_goals', 0.5))
    dyn_w = self._dyn_aux_weight
    stable_idx = jnp.asarray(sm.STABLE_INDICES)

    b_shared_opt = self._b_shared_opt
    h_phi_opt = self._h_phi_opt
    h_dyn_opt = self._h_dyn_opt
    phi_task_opt = self._phi_task_opt
    psi_opt = self._psi_opt
    actor_opt = self._actor_opt
    alpha_opt = self._alpha_opt

    obs_dim = config.obs_dim

    def critic_loss_fn(critic_params_dict, transitions, key):
      """Combined critic loss: L_InfoNCE on the score + dyn_w * L_dyn.

      ``critic_params_dict`` carries the four critic groups jointly so
      that we can take a single ``value_and_grad``; the gradients are
      then applied component-wise. ``h_dyn`` lives outside this dict
      because it does not see the contrastive grad.
      """
      p_b = critic_params_dict['b_shared']
      p_phi = critic_params_dict['h_phi']
      p_task = critic_params_dict['phi_task']
      p_psi = critic_params_dict['psi']

      logits = decomp_nets.apply_score(p_b, p_phi, p_task, p_psi,
                                       transitions.observation,
                                       transitions.action)
      B = logits.shape[0]
      I = jnp.eye(B)

      # InfoNCE in the same form as the existing critic
      # (continual_learning.py: critic_loss_fn). use_cpc=False (default)
      # uses sigmoid-BCE per-cell with the identity as labels and folds
      # the logsumexp regulariser into the metrics. use_cpc=True uses
      # softmax-CE plus logsumexp_penalty * logsumexp(...)^2.
      if use_cpc:
        loss = (optax.softmax_cross_entropy(logits=logits, labels=I)
                + logsumexp_penalty_coeff * jax.nn.logsumexp(logits, axis=1) ** 2)
      else:
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=I)
      total = jnp.mean(loss)

      # Diagnostics matching the existing learner's keys.
      correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
      bin_acc = jnp.mean((logits > 0) == I)
      cat_acc = jnp.mean(correct.astype(jnp.float32))
      logits_pos = jnp.sum(logits * I) / jnp.sum(I)
      logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
      lse = jax.nn.logsumexp(logits, axis=1) ** 2
      aux = dict(
          critic_loss=total,
          infonce=total,
          logsumexp=jnp.mean(lse),
          binary_accuracy=bin_acc,
          categorical_accuracy=cat_acc,
          logits_pos=logits_pos,
          logits_neg=logits_neg,
      )
      return total, aux

    def dyn_loss_fn(p_b, p_dyn, transitions):
      """Masked-MSE forward-dynamics loss on b_shared + h_dyn only."""
      hidden = decomp_nets.apply_b_shared(p_b, transitions.observation,
                                          transitions.action)
      pred = decomp_nets.apply_h_dyn(p_dyn, hidden)  # (B, d_M)
      next_state = transitions.next_observation[:, :obs_dim]
      target = next_state[:, stable_idx]             # (B, d_M)
      mse = jnp.mean((pred - target) ** 2)
      return mse, dict(dyn_mse=mse)

    def actor_loss_fn(policy_params, b_shared_params, h_phi_params,
                      phi_task_params, psi_params, log_alpha, transitions,
                      key):
      """Actor loss: matches continual_learning.py:408-438.

      Optionally rolls goals via ``config.random_goals`` (0.0 / 0.5 / 1.0),
      computes the critic score under the composed phi, takes
      ``-jnp.diag(score)``, and adds the entropy bonus when
      ``config.use_action_entropy=True``. The critic params are NOT
      stop-gradient-wrapped here because policy_params are the only
      argnums=0 leaves seen by ``value_and_grad``; the existing learner
      relies on the same pattern.
      """
      obs = transitions.observation
      state = obs[:, :obs_dim]
      goal = obs[:, obs_dim:]

      if random_goals == 0.0:
        new_state, new_goal = state, goal
      elif random_goals == 0.5:
        new_state = jnp.concatenate([state, state], axis=0)
        new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
      else:
        new_state = state
        new_goal = jnp.roll(goal, 1, axis=0)

      new_obs = jnp.concatenate([new_state, new_goal], axis=1)
      dist_params = policy_network.apply(policy_params, new_obs)
      action = sample_fn(dist_params, key)
      log_prob = log_prob_fn(dist_params, action)

      # Critic score under the composed phi. argnums=0 of the outer
      # value_and_grad is policy_params, so critic params receive no
      # gradient signal here.
      score = decomp_nets.apply_score(
          b_shared_params, h_phi_params, phi_task_params, psi_params,
          new_obs, action)
      q_action = jnp.diag(score)            # matched (s,a)-to-own-g entry
      actor_loss = -q_action                # maximise Q

      if use_action_entropy:
        alpha = jnp.exp(log_alpha)
        # Match continual_learning.py:435 sign: -= alpha * (-log_prob).
        actor_loss -= alpha * (-log_prob)

      ent_aux = dict(entropy_mean=jnp.mean(-log_prob),
                     actor_loss=jnp.mean(actor_loss))
      return jnp.mean(actor_loss), ent_aux

    def alpha_loss_fn(log_alpha, policy_params, transitions, key):
      dist_params = policy_network.apply(policy_params, transitions.observation)
      action = sample_fn(dist_params, key)
      log_prob = log_prob_fn(dist_params, action)
      alpha = jnp.exp(log_alpha)
      return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - target_entropy))

    def update_step(state: DecomposedTrainingState,
                    transitions: types.Transition) -> Tuple[DecomposedTrainingState, Dict[str, jnp.ndarray]]:
      key = state.key
      key, k_critic, k_dyn, k_actor, k_alpha = jax.random.split(key, 5)

      # ---- 1. critic InfoNCE step ----------------------------------
      critic_dict = dict(
          b_shared=state.b_shared_params,
          h_phi=state.h_phi_params,
          phi_task=state.phi_task_params,
          psi=state.psi_params,
      )
      (c_loss_val, c_aux), c_grads = jax.value_and_grad(
          critic_loss_fn, has_aux=True)(critic_dict, transitions, k_critic)

      # ---- 2. dyn-aux step on b_shared + h_dyn ---------------------
      (d_loss_val, d_aux), (g_b_dyn, g_h_dyn) = jax.value_and_grad(
          dyn_loss_fn, argnums=(0, 1), has_aux=True)(
              state.b_shared_params, state.h_dyn_params, transitions)

      # ---- 3. compose b_shared gradient (NCE + mu * dyn) -----------
      g_b_shared = jax.tree_util.tree_map(
          lambda g_nce, g_dyn: g_nce + dyn_w * g_dyn,
          c_grads['b_shared'], g_b_dyn)
      g_h_dyn_scaled = jax.tree_util.tree_map(lambda g: dyn_w * g, g_h_dyn)

      b_upd, b_opt = b_shared_opt.update(g_b_shared, state.b_shared_opt_state)
      new_b_shared = optax.apply_updates(state.b_shared_params, b_upd)
      phi_upd, phi_opt = h_phi_opt.update(c_grads['h_phi'],
                                           state.h_phi_opt_state)
      new_h_phi = optax.apply_updates(state.h_phi_params, phi_upd)
      dyn_upd, dyn_opt = h_dyn_opt.update(g_h_dyn_scaled,
                                          state.h_dyn_opt_state)
      new_h_dyn = optax.apply_updates(state.h_dyn_params, dyn_upd)
      task_upd, task_opt = phi_task_opt.update(c_grads['phi_task'],
                                                state.phi_task_opt_state)
      new_phi_task = optax.apply_updates(state.phi_task_params, task_upd)
      psi_upd, psi_opt_new = psi_opt.update(c_grads['psi'],
                                             state.psi_opt_state)
      new_psi = optax.apply_updates(state.psi_params, psi_upd)

      # ---- 4. actor step against the just-updated critic -----------
      log_alpha = state.alpha_params
      (a_loss_val, a_aux), a_grad = jax.value_and_grad(
          actor_loss_fn, has_aux=True)(
              state.policy_params, new_b_shared, new_h_phi, new_phi_task,
              new_psi, log_alpha, transitions, k_actor)
      act_upd, act_opt = actor_opt.update(a_grad, state.policy_opt_state)
      new_policy = optax.apply_updates(state.policy_params, act_upd)

      # ---- 5. adaptive entropy step --------------------------------
      al_loss_val, al_grad = jax.value_and_grad(alpha_loss_fn)(
          log_alpha, new_policy, transitions, k_alpha)
      al_upd, al_opt = alpha_opt.update(al_grad, state.alpha_optimizer_state)
      new_alpha = optax.apply_updates(log_alpha, al_upd)

      new_state = DecomposedTrainingState(
          policy_params=new_policy,
          policy_opt_state=act_opt,
          b_shared_params=new_b_shared,
          b_shared_opt_state=b_opt,
          h_phi_params=new_h_phi,
          h_phi_opt_state=phi_opt,
          h_dyn_params=new_h_dyn,
          h_dyn_opt_state=dyn_opt,
          phi_task_params=new_phi_task,
          phi_task_opt_state=task_opt,
          psi_params=new_psi,
          psi_opt_state=psi_opt_new,
          alpha_params=new_alpha,
          alpha_optimizer_state=al_opt,
          key=key,
      )

      metrics = {
          'critic_loss': c_aux['critic_loss'],
          'infonce': c_aux['infonce'],
          'logsumexp': c_aux['logsumexp'],
          'binary_accuracy': c_aux['binary_accuracy'],
          'categorical_accuracy': c_aux['categorical_accuracy'],
          'logits_pos': c_aux['logits_pos'],
          'logits_neg': c_aux['logits_neg'],
          'decomp/L_dyn': d_aux['dyn_mse'],
          'actor_loss': a_aux['actor_loss'],
          'entropy_mean': a_aux['entropy_mean'],
          'alpha_loss': al_loss_val,
          'alpha': jnp.exp(new_alpha),
      }
      return new_state, metrics

    # The iterator hands us transitions with leaves of shape
    # [B * N, ...] where N = num_sgd_steps_per_step. Match the existing
    # learner: reshape to [N, B, ...] and scan ``update_step`` over the
    # leading axis.
    num_sgd = int(getattr(config, 'num_sgd_steps_per_step', 1))

    def scan_step(state, transitions):
      batched = jax.tree_util.tree_map(
          lambda a: jnp.reshape(a, (num_sgd, -1, *a.shape[1:])),
          transitions)

      def scan_body(carry, mini_batch):
        return update_step(carry, mini_batch)

      state, metrics = jax.lax.scan(scan_body, state, batched, length=num_sgd)
      mean_metrics = jax.tree_util.tree_map(jnp.mean, metrics)
      return state, mean_metrics

    return jax.jit(scan_step)

  # ------------------------------------------------------------------
  # acme.Learner interface
  # ------------------------------------------------------------------

  def step(self):
    sample = next(self._iterator)
    transitions = types.Transition(*sample.data)

    # Cache the last batch for external use (e.g., rl_metrics in
    # run_continual_contrastive.py:852). Mirrors the equivalent line
    # in ContinualContrastiveLearner.step().
    self._last_transitions = transitions

    self._state, metrics = self._update_step(self._state, transitions)

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
          None,
          transitions,
          diagnostic_key)
      metrics = {**metrics, **diagnostic_metrics}

    timestamp = time.time()
    elapsed = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp
    counts = self._counter.increment(steps=1, walltime=elapsed)
    if elapsed > 0:
      metrics = {**metrics, 'steps_per_second':
                 self._config.num_sgd_steps_per_step / elapsed}
    self._last_metrics = {**{k: float(v) for k, v in metrics.items()},
                          **counts}
    self._logger.write(self._last_metrics)

  def get_variables(self, names):
    """Return the current actor (and a synthetic q params dict for evaluator)."""
    actor_params = self._state.policy_params
    # Acme evaluators that read 'critic' get a tuple of the four critic
    # parameter groups; downstream code that needs the composed critic
    # should call ``compose_score`` from this module's networks.
    critic_bundle = {
        'b_shared': self._state.b_shared_params,
        'h_phi': self._state.h_phi_params,
        'h_dyn': self._state.h_dyn_params,
        'phi_task': self._state.phi_task_params,
        'psi': self._state.psi_params,
    }
    available = {'policy': actor_params, 'critic': critic_bundle}
    return [available[n] for n in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state

  # ------------------------------------------------------------------
  # Accessors mirroring ContinualContrastiveLearner so the orchestrator
  # can checkpoint and continue without branching.
  # ------------------------------------------------------------------

  @property
  def last_metrics(self):
    """Last metrics dict from the most recent step() call."""
    return getattr(self, '_last_metrics', {})

  @property
  def last_transitions(self):
    """Last preprocessed batch of transitions from the most recent step()."""
    return getattr(self, '_last_transitions', None)

  @property
  def q_params(self):
    """Compatibility shim: return ``None`` so callers (e.g., the runner's
    rl_metrics block) can detect that this learner uses the decomposed
    critic layout and skip code paths that expect a single monolithic
    ``q_params`` pytree with ``sa_encoder`` / ``g_encoder`` modules.

    Use the individual accessors (``b_shared_params``, ``h_phi_params``,
    ``phi_task_params``, ``psi_params``, ``h_dyn_params``) for the
    decomposed groups, or ``get_variables(['critic'])[0]`` for the
    bundle dict.
    """
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

  @property
  def last_metrics(self):
    return self._last_metrics
