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
  actor params : sees actor objective only       (reset or persistent)

Continual handoff at task k > 0:

  b_shared, h_phi, h_dyn, psi, q_optimizer states for them   -> carry over
  phi_task params + opt state                                 -> reinit
  actor params + opt state + entropy state                     -> actor-mode

The actor objective matches the existing SGCRL learner: optionally roll
goals via ``config.random_goals`` (0.0 / 0.5 / 1.0), then maximise the
critic's diagonal score under the composed phi, plus an entropy bonus
when ``config.use_action_entropy=True``.

Excluded from this learner intentionally (out of scope for proposal 1):
  - TD critic path (``config.use_td=False`` is required)
  - Twin Q
  - Negative bank
  - Image observations
  - Actor CKA decomposition (use ``actor_mode='reset'`` or
    ``actor_mode='persistent'``)

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
  # Actor (reset or carried according to actor_mode)
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

  # Optional task-local forward action-effect head (Bridge-DCC). Appending
  # defaulted fields preserves construction/unpickling of legacy DCC states.
  u_task_params: Optional[networks_lib.Params] = None
  u_task_opt_state: Optional[optax.OptState] = None
  control_q_scale_ema: Optional[jnp.ndarray] = None
  outcome_progress_mean_ema: Optional[jnp.ndarray] = None
  outcome_progress_var_ema: Optional[jnp.ndarray] = None
  success_buffer_observation: Optional[jnp.ndarray] = None
  success_buffer_action: Optional[jnp.ndarray] = None
  success_buffer_size: Optional[jnp.ndarray] = None
  success_buffer_index: Optional[jnp.ndarray] = None
  counterfactual_rank_updates: Optional[jnp.ndarray] = None


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
      actor_mode: str = 'reset',
      # Actor state carried only when actor_mode='persistent'.
      prev_policy_params: Optional[networks_lib.Params] = None,
      prev_policy_opt_state: Optional[optax.OptState] = None,
      prev_alpha_params: Optional[jnp.ndarray] = None,
      prev_alpha_opt_state: Optional[optax.OptState] = None,
      # Shared critic state carried in from the previous task.
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
    if actor_mode not in ('reset', 'persistent'):
      raise ValueError(
          'critic_mode=decomposed supports actor_mode in '
          "{'reset', 'persistent'} only.")

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
    self._actor_mode = actor_mode
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
    self._action_effect_enabled = bool(
        getattr(continual_config, 'action_effect_enabled', False))
    self._action_effect_loss_weight = float(
        getattr(continual_config, 'action_effect_loss_weight', 1.0))
    self._action_effect_discount = float(
        getattr(continual_config, 'action_effect_discount', 0.99))
    self._action_effect_temperature = float(
        getattr(continual_config, 'action_effect_temperature', 1.0))
    self._action_effect_actor_weight = float(
        getattr(continual_config, 'action_effect_actor_weight', 1.0))
    self._action_effect_normalization_eps = float(
        getattr(continual_config, 'action_effect_normalization_eps', 1e-3))
    self._action_effect_q_scale_ema_decay = float(getattr(
        continual_config, 'action_effect_q_scale_ema_decay', 0.99))
    self._action_effect_actor_mode = getattr(
        continual_config, 'action_effect_actor_mode', 'combined')
    self._action_effect_target_mode = getattr(
        continual_config, 'action_effect_target_mode', 'psi_one_step')
    self._outcome_success_actor_weight = float(getattr(
        continual_config, 'outcome_success_actor_weight', 1.0))
    self._outcome_progress_ema_decay = float(getattr(
        continual_config, 'outcome_progress_ema_decay', 0.99))
    self._outcome_progress_std_floor = float(getattr(
        continual_config, 'outcome_progress_std_floor', 0.01))
    self._success_bc_weight = float(getattr(
        continual_config, 'success_bc_weight', 0.0))
    self._success_buffer_capacity = int(getattr(
        continual_config, 'success_buffer_capacity', 4096))
    self._success_bc_batch_size = int(getattr(
        continual_config, 'success_bc_batch_size', 64))
    self._counterfactual_rank_temperature = float(getattr(
        continual_config, 'counterfactual_rank_pairwise_temperature', 1.0))
    self._counterfactual_rank_min_gap = float(getattr(
        continual_config, 'counterfactual_rank_min_outcome_gap', 0.002))
    self._counterfactual_rank_l2_weight = float(getattr(
        continual_config, 'counterfactual_rank_l2_weight', 1e-4))
    if self._action_effect_actor_mode not in ('combined', 'effect_only'):
      raise ValueError(
          'action_effect_actor_mode must be combined or effect_only.')
    if self._action_effect_target_mode not in (
        'psi_one_step', 'raw_horizon', 'counterfactual_rank'):
      raise ValueError(
          'action_effect_target_mode must be psi_one_step, raw_horizon, or '
          'counterfactual_rank.')
    if (self._action_effect_actor_mode == 'effect_only'
        and not self._action_effect_enabled):
      raise ValueError('effect_only actor mode requires action_effect_enabled.')
    if (self._action_effect_target_mode == 'raw_horizon'
        and not self._action_effect_enabled):
      raise ValueError('raw_horizon target mode requires action_effect_enabled.')
    if self._action_effect_target_mode == 'counterfactual_rank':
      if not self._action_effect_enabled:
        raise ValueError(
            'counterfactual_rank target mode requires action_effect_enabled.')
      if self._action_effect_actor_mode != 'effect_only':
        raise ValueError(
            'counterfactual_rank requires effect_only actor mode so the test '
            'does not reintroduce the miscalibrated DCC action landscape.')
      if self._counterfactual_rank_temperature <= 0:
        raise ValueError('counterfactual rank temperature must be positive.')
      if self._counterfactual_rank_min_gap < 0:
        raise ValueError('counterfactual rank minimum gap cannot be negative.')
    if self._success_bc_weight > 0:
      if self._action_effect_target_mode != 'raw_horizon':
        raise ValueError('success retention requires raw_horizon targets.')
      if self._success_buffer_capacity <= 0 or self._success_bc_batch_size <= 0:
        raise ValueError('Success-buffer capacity and batch size must be > 0.')
    if self._action_effect_enabled:
      if (self._action_effect_target_mode == 'psi_one_step'
          and (getattr(decomp_nets, 'combine_mode', 'add') != 'add'
          or getattr(decomp_nets, 'goal_encoder_mode', 'shared') != 'shared')):
        raise ValueError(
            'The action-effect head requires combine_mode=add and '
            'goal_encoder_mode=shared so its output and psi(g) share the '
            'same geometry.')
      if decomp_nets.init_u_task is None or decomp_nets.apply_u_task is None:
        raise ValueError('Action-effect head requested but network is absent.')

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
    self._u_task_opt = (
        optax.adam(lr) if self._action_effect_enabled else None)

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

    # The task-specific critic encoder is always reinitialised. The actor
    # follows actor_mode: reset starts from a fresh policy, while persistent
    # carries both policy parameters and Adam state without modification.
    phi_task_params = decomp_nets.init_phi_task(subkeys[4])
    phi_task_opt_state = self._phi_task_opt.init(phi_task_params)
    carry_actor = actor_mode == 'persistent' and task_id > 0
    if carry_actor:
      if prev_policy_params is None or prev_policy_opt_state is None:
        raise ValueError(
            'Persistent DCC actor requires previous policy parameters and '
            'optimizer state at task_id > 0. Start from task 0 or resume from '
            'a persistent-actor DCC checkpoint.')
      policy_params = prev_policy_params
      policy_opt_state = prev_policy_opt_state
    else:
      policy_params = policy_network.init(subkeys[5])
      policy_opt_state = self._actor_opt.init(policy_params)

    if self._adaptive_entropy:
      if carry_actor:
        if prev_alpha_params is None or prev_alpha_opt_state is None:
          raise ValueError(
              'Persistent DCC actor requires previous entropy parameters and '
              'optimizer state at task_id > 0.')
        alpha_params = prev_alpha_params
        alpha_opt_state = prev_alpha_opt_state
      else:
        alpha_params = jnp.array(log_alpha_init, dtype=jnp.float32)
        alpha_opt_state = self._alpha_opt.init(alpha_params)
    else:
      alpha_params = None
      alpha_opt_state = None

    if self._action_effect_enabled:
      # Fold into the legacy state key so enabling this head does not change
      # any of DCC's existing initialization keys. The head is task-local.
      u_task_params = decomp_nets.init_u_task(
          jax.random.fold_in(subkeys[6], 0xDAF))
      u_task_opt_state = self._u_task_opt.init(u_task_params)
      control_q_scale_ema = jnp.asarray(1.0, dtype=jnp.float32)
    else:
      u_task_params = None
      u_task_opt_state = None
      control_q_scale_ema = None

    if self._action_effect_target_mode == 'raw_horizon':
      outcome_progress_mean_ema = jnp.asarray(0.0, dtype=jnp.float32)
      outcome_progress_var_ema = jnp.asarray(
          self._outcome_progress_std_floor ** 2, dtype=jnp.float32)
    else:
      outcome_progress_mean_ema = None
      outcome_progress_var_ema = None
    if self._success_bc_weight > 0:
      success_buffer_observation = jnp.zeros(
          (self._success_buffer_capacity, 2 * config.obs_dim),
          dtype=jnp.float32)
      success_buffer_action = jnp.zeros(
          (self._success_buffer_capacity, decomp_nets.action_dim),
          dtype=jnp.float32)
      success_buffer_size = jnp.asarray(0, dtype=jnp.int32)
      success_buffer_index = jnp.asarray(0, dtype=jnp.int32)
    else:
      success_buffer_observation = None
      success_buffer_action = None
      success_buffer_size = None
      success_buffer_index = None
    counterfactual_rank_updates = (
        jnp.asarray(0, dtype=jnp.int32)
        if self._action_effect_target_mode == 'counterfactual_rank' else None)

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
        u_task_params=u_task_params,
        u_task_opt_state=u_task_opt_state,
        control_q_scale_ema=control_q_scale_ema,
        outcome_progress_mean_ema=outcome_progress_mean_ema,
        outcome_progress_var_ema=outcome_progress_var_ema,
        success_buffer_observation=success_buffer_observation,
        success_buffer_action=success_buffer_action,
        success_buffer_size=success_buffer_size,
        success_buffer_index=success_buffer_index,
        counterfactual_rank_updates=counterfactual_rank_updates,
    )

    self._update_step = self._make_update_step()
    self._counterfactual_rank_update = (
        self._make_counterfactual_rank_update()
        if self._action_effect_target_mode == 'counterfactual_rank' else None)

    # Optional task-5/task-8 shortcut diagnostics.  The default interval of
    # zero preserves the legacy DCC hot path exactly.
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
    action_effect_enabled = self._action_effect_enabled
    action_effect_loss_weight = self._action_effect_loss_weight
    action_effect_discount = self._action_effect_discount
    action_effect_temperature = self._action_effect_temperature
    action_effect_actor_weight = self._action_effect_actor_weight
    action_effect_eps = self._action_effect_normalization_eps
    q_scale_ema_decay = self._action_effect_q_scale_ema_decay
    action_effect_actor_mode = self._action_effect_actor_mode
    action_effect_target_mode = self._action_effect_target_mode
    outcome_progress_loss_weight = float(getattr(
        self._continual_cfg, 'outcome_progress_loss_weight', 1.0))
    outcome_success_loss_weight = float(getattr(
        self._continual_cfg, 'outcome_success_loss_weight', 1.0))
    outcome_success_actor_weight = self._outcome_success_actor_weight
    outcome_progress_ema_decay = self._outcome_progress_ema_decay
    outcome_progress_std_floor = self._outcome_progress_std_floor
    success_bc_weight = self._success_bc_weight
    success_buffer_capacity = self._success_buffer_capacity
    success_bc_batch_size = self._success_bc_batch_size
    iwr_enabled = bool(getattr(
        self._continual_cfg, 'interaction_weighted_relabeling', False))
    interaction_threshold = float(getattr(
        self._continual_cfg, 'interaction_threshold', 0.09))
    interaction_bandwidth = float(getattr(
        self._continual_cfg, 'interaction_bandwidth', 0.03))
    interaction_weight_floor = float(getattr(
        self._continual_cfg, 'interaction_weight_floor', 0.05))
    stable_idx = jnp.asarray(sm.STABLE_INDICES)

    b_shared_opt = self._b_shared_opt
    h_phi_opt = self._h_phi_opt
    h_dyn_opt = self._h_dyn_opt
    phi_task_opt = self._phi_task_opt
    psi_opt = self._psi_opt
    actor_opt = self._actor_opt
    alpha_opt = self._alpha_opt
    u_task_opt = self._u_task_opt

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

    def action_effect_loss_fn(p_u, p_psi, progress_mean, progress_var,
                              transitions):
      """Train either the legacy psi-effect head or raw H-step outcome head."""
      prediction = decomp_nets.apply_u_task(
          p_u, transitions.observation, transitions.action)
      if action_effect_target_mode == 'raw_horizon':
        raw_progress = transitions.extras['outcome_progress']
        success = transitions.extras['outcome_success']
        progress_std = jnp.sqrt(jnp.maximum(
            progress_var, outcome_progress_std_floor ** 2))
        progress_target = jax.lax.stop_gradient(
            (raw_progress - progress_mean) / progress_std)
        progress_prediction = prediction[:, 0]
        diagnostic_prediction = progress_prediction
        success_logit = prediction[:, 1]
        progress_loss = optax.huber_loss(
            progress_prediction, progress_target, delta=1.0)
        success_loss = optax.sigmoid_binary_cross_entropy(
            success_logit, success)
        per_transition_loss = (
            outcome_progress_loss_weight * progress_loss
            + outcome_success_loss_weight * success_loss)
        pred_centered = progress_prediction - jnp.mean(progress_prediction)
        target_centered = raw_progress - jnp.mean(raw_progress)
        progress_corr = jnp.sum(pred_centered * target_centered) / jnp.maximum(
            jnp.linalg.norm(pred_centered) * jnp.linalg.norm(target_centered),
            1e-8)
        shuffled_prediction = decomp_nets.apply_u_task(
            p_u, transitions.observation, jnp.roll(
                transitions.action, 1, axis=0))[:, 0]
        shuffled_centered = shuffled_prediction - jnp.mean(
            shuffled_prediction)
        prediction_centered = diagnostic_prediction - jnp.mean(
            diagnostic_prediction)
        action_shuffle_retention = jnp.sum(
            prediction_centered * shuffled_centered) / jnp.maximum(
                jnp.linalg.norm(prediction_centered)
                * jnp.linalg.norm(shuffled_centered), 1e-8)
        fixed_observation = jnp.repeat(
            transitions.observation[:1], transitions.action.shape[0], axis=0)
        fixed_action_scores = decomp_nets.apply_u_task(
            p_u, fixed_observation, transitions.action)[:, 0]
      else:
        progress_loss = jnp.zeros((prediction.shape[0],))
        success_loss = jnp.zeros((prediction.shape[0],))
        progress_corr = jnp.asarray(0.0)
      state = transitions.observation[:, :obs_dim]
      if action_effect_target_mode == 'psi_one_step':
        next_state = transitions.next_observation[:, :obs_dim]
        state_as_goal = jnp.concatenate([state, state], axis=1)
        next_state_as_goal = jnp.concatenate([next_state, next_state], axis=1)
        psi_state = decomp_nets.apply_psi(p_psi, state_as_goal)
        psi_next = decomp_nets.apply_psi(p_psi, next_state_as_goal)
        psi_state = psi_state / jnp.maximum(
            jnp.linalg.norm(psi_state, axis=1, keepdims=True), 1e-8)
        psi_next = psi_next / jnp.maximum(
            jnp.linalg.norm(psi_next, axis=1, keepdims=True), 1e-8)
        continuation = transitions.discount[:, None]
        target = jax.lax.stop_gradient(
            action_effect_discount * continuation * psi_next - psi_state)
        per_transition_loss = jnp.mean(
            optax.huber_loss(prediction, target, delta=1.0), axis=1)
        goal_repr = decomp_nets.apply_psi(p_psi, transitions.observation)
        goal_repr = goal_repr / jnp.maximum(
            jnp.linalg.norm(goal_repr, axis=1, keepdims=True), 1e-8)
        diagnostic_prediction = jnp.sum(prediction * goal_repr, axis=1)
        shuffled_effect = decomp_nets.apply_u_task(
            p_u, transitions.observation,
            jnp.roll(transitions.action, 1, axis=0))
        shuffled_prediction = jnp.sum(shuffled_effect * goal_repr, axis=1)
        fixed_observation = jnp.repeat(
            transitions.observation[:1], transitions.action.shape[0], axis=0)
        fixed_effect = decomp_nets.apply_u_task(
            p_u, fixed_observation, transitions.action)
        fixed_goal_repr = decomp_nets.apply_psi(p_psi, fixed_observation)
        fixed_goal_repr = fixed_goal_repr / jnp.maximum(
            jnp.linalg.norm(fixed_goal_repr, axis=1, keepdims=True), 1e-8)
        fixed_action_scores = jnp.sum(
            fixed_effect * fixed_goal_repr, axis=1)
        prediction_centered = diagnostic_prediction - jnp.mean(
            diagnostic_prediction)
        shuffled_centered = shuffled_prediction - jnp.mean(
            shuffled_prediction)
        action_shuffle_retention = jnp.sum(
            prediction_centered * shuffled_centered) / jnp.maximum(
                jnp.linalg.norm(prediction_centered)
                * jnp.linalg.norm(shuffled_centered), 1e-8)
      if iwr_enabled:
        interaction_distance = jnp.linalg.norm(
            state[:, :3] - state[:, 4:7], axis=1)
        standardized = (
            (interaction_distance - interaction_threshold)
            / interaction_bandwidth)
        interaction_weight = (
            interaction_weight_floor + jnp.exp(-0.5 * standardized ** 2))
        loss = jnp.sum(
            interaction_weight * per_transition_loss) / jnp.maximum(
                jnp.sum(interaction_weight), 1e-8)
      else:
        interaction_weight = jnp.ones_like(per_transition_loss)
        loss = jnp.mean(per_transition_loss)
      if action_effect_target_mode == 'psi_one_step':
        pred_norm = jnp.linalg.norm(prediction, axis=1)
        target_norm = jnp.linalg.norm(target, axis=1)
        cosine = jnp.sum(prediction * target, axis=1) / jnp.maximum(
            pred_norm * target_norm, 1e-8)
      else:
        pred_norm = jnp.abs(prediction[:, 0])
        target_norm = jnp.abs(raw_progress)
        cosine = jnp.zeros_like(pred_norm)
      return loss, {
          'action_effect/loss': loss,
          'action_effect/cosine': jnp.mean(cosine),
          'action_effect/pred_norm': jnp.mean(pred_norm),
          'action_effect/target_norm': jnp.mean(target_norm),
          'action_effect/interaction_weight_mean':
              jnp.mean(interaction_weight),
          'outcome/progress_loss': jnp.mean(progress_loss),
          'outcome/success_loss': jnp.mean(success_loss),
          'outcome/progress_target_mean': (
              jnp.mean(raw_progress)
              if action_effect_target_mode == 'raw_horizon' else 0.0),
          'outcome/progress_target_std': (
              jnp.std(raw_progress)
              if action_effect_target_mode == 'raw_horizon' else 0.0),
          'outcome/progress_prediction_std': (
              jnp.std(prediction[:, 0])
              if action_effect_target_mode == 'raw_horizon' else 0.0),
          'outcome/progress_pearson': progress_corr,
          'outcome/action_shuffle_delta': jnp.mean(jnp.abs(
              diagnostic_prediction - shuffled_prediction)),
          'outcome/action_shuffle_retention': action_shuffle_retention,
          'outcome/fixed_state_action_std': jnp.std(fixed_action_scores),
          'outcome/success_rate': (
              jnp.mean(success)
              if action_effect_target_mode == 'raw_horizon' else 0.0),
      }

    def actor_loss_fn(policy_params, b_shared_params, h_phi_params,
                      phi_task_params, psi_params, u_task_params,
                      control_q_scale_ema, success_buffer_observation,
                      success_buffer_action, success_buffer_size,
                      counterfactual_rank_updates, log_alpha, transitions,
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

      if action_effect_target_mode == 'counterfactual_rank':
        # The ranker is supervised on the environment's original task goal,
        # never on HER future-state goals. Keep the actor on that same domain.
        new_obs = transitions.extras['counterfactual_task_observation']
      else:
        if random_goals == 0.0:
          new_state, new_goal = state, goal
        elif random_goals == 0.5:
          new_state = jnp.concatenate([state, state], axis=0)
          new_goal = jnp.concatenate(
              [goal, jnp.roll(goal, 1, axis=0)], axis=0)
        else:
          new_state = state
          new_goal = jnp.roll(goal, 1, axis=0)
        new_obs = jnp.concatenate([new_state, new_goal], axis=1)
      key, action_key, bc_key = jax.random.split(key, 3)
      dist_params = policy_network.apply(policy_params, new_obs)
      action = sample_fn(dist_params, action_key)
      log_prob = log_prob_fn(dist_params, action)

      # Critic score under the composed phi. argnums=0 of the outer
      # value_and_grad is policy_params, so critic params receive no
      # gradient signal here.
      score = decomp_nets.apply_score(
          b_shared_params, h_phi_params, phi_task_params, psi_params,
          new_obs, action)
      q_action = jnp.diag(score)            # matched (s,a)-to-own-g entry
      if action_effect_enabled:
        effect = decomp_nets.apply_u_task(u_task_params, new_obs, action)
        if action_effect_target_mode == 'raw_horizon':
          advantage = (
              effect[:, 0]
              + outcome_success_actor_weight * jax.nn.sigmoid(effect[:, 1]))
        elif action_effect_target_mode == 'counterfactual_rank':
          # Do not let random head initialization steer the actor. The gate
          # opens only after at least one informative exact-state rank update.
          active = (counterfactual_rank_updates > 0).astype(effect.dtype)
          advantage = active * effect[:, 0]
        else:
          goal_repr = decomp_nets.apply_psi(psi_params, new_obs)
          goal_repr = goal_repr / jnp.maximum(
              jnp.linalg.norm(goal_repr, axis=1, keepdims=True), 1e-8)
          advantage_raw = jnp.sum(effect * goal_repr, axis=1)
          advantage = jnp.tanh(
              advantage_raw / max(action_effect_temperature, 1e-8))
        q_scale = jax.lax.stop_gradient(jnp.maximum(
            control_q_scale_ema, action_effect_eps))
        if action_effect_actor_mode == 'effect_only':
          control_score = action_effect_actor_weight * advantage
        else:
          control_score = (
              q_action / q_scale + action_effect_actor_weight * advantage)
      else:
        advantage = jnp.zeros_like(q_action)
        q_scale = jnp.asarray(1.0, dtype=q_action.dtype)
        control_score = q_action
      actor_loss = -control_score

      if use_action_entropy:
        alpha = jnp.exp(log_alpha)
        # Match continual_learning.py:435 sign: -= alpha * (-log_prob).
        actor_loss -= alpha * (-log_prob)

      if success_bc_weight > 0:
        safe_size = jnp.maximum(success_buffer_size, 1)
        bc_index = jax.random.randint(
            bc_key, (success_bc_batch_size,), 0, safe_size)
        bc_observation = success_buffer_observation[bc_index]
        bc_action = success_buffer_action[bc_index]
        bc_dist_params = policy_network.apply(policy_params, bc_observation)
        bc_loss = -jnp.mean(log_prob_fn(bc_dist_params, bc_action))
        bc_active = (success_buffer_size > 0).astype(jnp.float32)
        actor_loss = actor_loss + success_bc_weight * bc_active * bc_loss
      else:
        bc_loss = jnp.asarray(0.0)
        bc_active = jnp.asarray(0.0)

      ent_aux = dict(
          entropy_mean=jnp.mean(-log_prob),
          actor_loss=jnp.mean(actor_loss),
          action_effect_advantage=jnp.mean(advantage),
          action_effect_advantage_std=jnp.std(advantage),
          action_effect_dcc_scale=q_scale,
          action_effect_batch_q_abs=jnp.mean(jnp.abs(q_action)),
          action_effect_score_std=jnp.std(advantage),
          action_effect_head_to_dcc_ratio=(
              jnp.mean(jnp.abs(action_effect_actor_weight * advantage))
              / jnp.maximum(jnp.mean(jnp.abs(q_action / q_scale)), 1e-8)),
          control_score=jnp.mean(control_score),
          success_bc_loss=bc_loss,
          success_bc_active=bc_active)
      return jnp.mean(actor_loss), ent_aux

    def alpha_loss_fn(log_alpha, policy_params, transitions, key):
      dist_params = policy_network.apply(policy_params, transitions.observation)
      action = sample_fn(dist_params, key)
      log_prob = log_prob_fn(dist_params, action)
      alpha = jnp.exp(log_alpha)
      return jnp.mean(alpha * jax.lax.stop_gradient(-log_prob - target_entropy))

    def update_success_buffer(observation_buffer, action_buffer, size, index,
                              transitions):
      """Insert task-goal successes without changing the replay sampler."""
      retention_observation = transitions.extras[
          'outcome_retention_observation']
      retention_action = transitions.action
      success = transitions.extras['outcome_task_success'] > 0.5

      def insert_one(carry, values):
        obs_buffer, act_buffer, current_size, current_index = carry
        obs, action, keep = values

        def do_insert(bufs):
          o_buf, a_buf, n, cursor = bufs
          o_buf = o_buf.at[cursor].set(obs)
          a_buf = a_buf.at[cursor].set(action)
          return (o_buf, a_buf,
                  jnp.minimum(n + 1, success_buffer_capacity),
                  (cursor + 1) % success_buffer_capacity)

        new_carry = jax.lax.cond(
            keep, do_insert, lambda bufs: bufs,
            (obs_buffer, act_buffer, current_size, current_index))
        return new_carry, jnp.asarray(0, dtype=jnp.int32)

      final, _ = jax.lax.scan(
          insert_one,
          (observation_buffer, action_buffer, size, index),
          (retention_observation, retention_action, success))
      return final

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

      # ---- 4. optional local forward action-effect step ------------
      if (action_effect_enabled
          and action_effect_target_mode != 'counterfactual_rank'):
        if action_effect_target_mode == 'raw_horizon':
          batch_progress = transitions.extras['outcome_progress']
          batch_progress_mean = jnp.mean(batch_progress)
          batch_progress_var = jnp.var(batch_progress)
          new_progress_mean = (
              outcome_progress_ema_decay * state.outcome_progress_mean_ema
              + (1.0 - outcome_progress_ema_decay) * batch_progress_mean)
          new_progress_var = (
              outcome_progress_ema_decay * state.outcome_progress_var_ema
              + (1.0 - outcome_progress_ema_decay) * batch_progress_var)
        else:
          new_progress_mean = state.outcome_progress_mean_ema
          new_progress_var = state.outcome_progress_var_ema
        (u_loss_val, u_aux), u_grad = jax.value_and_grad(
            action_effect_loss_fn, has_aux=True)(
                state.u_task_params, new_psi,
                state.outcome_progress_mean_ema,
                state.outcome_progress_var_ema, transitions)
        u_grad = jax.tree_util.tree_map(
            lambda g: action_effect_loss_weight * g, u_grad)
        u_upd, u_opt_state = u_task_opt.update(
            u_grad, state.u_task_opt_state)
        new_u_task = optax.apply_updates(state.u_task_params, u_upd)
      else:
        u_loss_val = jnp.asarray(0.0)
        u_aux = {
            'action_effect/loss': u_loss_val,
            'action_effect/cosine': u_loss_val,
            'action_effect/pred_norm': u_loss_val,
            'action_effect/target_norm': u_loss_val,
            'action_effect/interaction_weight_mean': u_loss_val,
            'outcome/progress_loss': u_loss_val,
            'outcome/success_loss': u_loss_val,
            'outcome/progress_target_mean': u_loss_val,
            'outcome/progress_target_std': u_loss_val,
            'outcome/progress_prediction_std': u_loss_val,
            'outcome/progress_pearson': u_loss_val,
            'outcome/action_shuffle_delta': u_loss_val,
            'outcome/action_shuffle_retention': u_loss_val,
            'outcome/fixed_state_action_std': u_loss_val,
            'outcome/success_rate': u_loss_val,
        }
        new_u_task = state.u_task_params
        u_opt_state = state.u_task_opt_state
        new_progress_mean = state.outcome_progress_mean_ema
        new_progress_var = state.outcome_progress_var_ema

      if success_bc_weight > 0:
        (new_success_observation, new_success_action,
         new_success_size, new_success_index) = update_success_buffer(
             state.success_buffer_observation, state.success_buffer_action,
             state.success_buffer_size, state.success_buffer_index,
             transitions)
      else:
        new_success_observation = state.success_buffer_observation
        new_success_action = state.success_buffer_action
        new_success_size = state.success_buffer_size
        new_success_index = state.success_buffer_index

      # ---- 5. actor step against the just-updated critic -----------
      log_alpha = state.alpha_params
      (a_loss_val, a_aux), a_grad = jax.value_and_grad(
          actor_loss_fn, has_aux=True)(
              state.policy_params, new_b_shared, new_h_phi, new_phi_task,
              new_psi, new_u_task, state.control_q_scale_ema,
              new_success_observation, new_success_action, new_success_size,
              state.counterfactual_rank_updates, log_alpha, transitions,
              k_actor)
      act_upd, act_opt = actor_opt.update(a_grad, state.policy_opt_state)
      new_policy = optax.apply_updates(state.policy_params, act_upd)
      if action_effect_enabled:
        new_control_q_scale_ema = (
            q_scale_ema_decay * state.control_q_scale_ema
            + (1.0 - q_scale_ema_decay)
            * jax.lax.stop_gradient(a_aux['action_effect_batch_q_abs']))
      else:
        new_control_q_scale_ema = state.control_q_scale_ema

      # ---- 6. adaptive entropy step --------------------------------
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
          u_task_params=new_u_task,
          u_task_opt_state=u_opt_state,
          control_q_scale_ema=new_control_q_scale_ema,
          outcome_progress_mean_ema=new_progress_mean,
          outcome_progress_var_ema=new_progress_var,
          success_buffer_observation=new_success_observation,
          success_buffer_action=new_success_action,
          success_buffer_size=new_success_size,
          success_buffer_index=new_success_index,
          counterfactual_rank_updates=state.counterfactual_rank_updates,
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
      if action_effect_enabled:
        metrics.update({
            **u_aux,
            'action_effect/advantage_mean':
                a_aux['action_effect_advantage'],
            'action_effect/advantage_std':
                a_aux['action_effect_advantage_std'],
            'action_effect/dcc_scale': a_aux['action_effect_dcc_scale'],
            'action_effect/control_score': a_aux['control_score'],
            'action_effect/head_score_std':
                a_aux['action_effect_score_std'],
            'action_effect/head_to_dcc_ratio':
                a_aux['action_effect_head_to_dcc_ratio'],
            'retention/bc_loss': a_aux['success_bc_loss'],
            'retention/bc_active': a_aux['success_bc_active'],
            'retention/buffer_size': (
                new_success_size if success_bc_weight > 0 else 0.0),
            'counterfactual_rank/active': (
                (state.counterfactual_rank_updates > 0).astype(jnp.float32)
                if action_effect_target_mode == 'counterfactual_rank'
                else 0.0),
        })
      if iwr_enabled:
        sampled_distance = transitions.extras['iwr_interaction_distance']
        metrics.update({
            'iwr/selected_interaction_distance':
                jnp.mean(sampled_distance),
            'iwr/selected_near_bridge_fraction': jnp.mean(
                (jnp.abs(sampled_distance - interaction_threshold)
                 <= interaction_bandwidth).astype(jnp.float32)),
            'iwr/selected_sampling_weight': jnp.mean(
                transitions.extras['iwr_sampling_weight']),
        })
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

  def _make_counterfactual_rank_update(self):
    """Build the exact-state, within-anchor pairwise ranking update."""
    apply_u_task = self._decomp_nets.apply_u_task
    optimizer = self._u_task_opt
    temperature = self._counterfactual_rank_temperature
    min_gap = self._counterfactual_rank_min_gap
    l2_weight = self._counterfactual_rank_l2_weight

    def update(params, opt_state, observations, actions, outcomes,
               informative):
      num_anchors, num_candidates = outcomes.shape

      def loss_fn(current_params):
        flat_observations = observations.reshape(
            (num_anchors * num_candidates, observations.shape[-1]))
        flat_actions = actions.reshape(
            (num_anchors * num_candidates, actions.shape[-1]))
        scores = apply_u_task(
            current_params, flat_observations, flat_actions)[:, 0]
        scores = scores.reshape((num_anchors, num_candidates))

        outcome_delta = outcomes[:, :, None] - outcomes[:, None, :]
        score_delta = scores[:, :, None] - scores[:, None, :]
        upper = jnp.triu(
            jnp.ones((num_candidates, num_candidates), dtype=bool), k=1)
        valid = (
            informative[:, None, None]
            & upper[None, :, :]
            & (jnp.abs(outcome_delta) >= min_gap))
        valid_float = valid.astype(scores.dtype)
        pair_count = jnp.sum(valid_float)
        preference = jnp.sign(outcome_delta)
        pair_loss = jax.nn.softplus(
            -preference * score_delta / temperature)
        rank_loss = jnp.sum(valid_float * pair_loss) / jnp.maximum(
            pair_count, 1.0)

        leaves = jax.tree_util.tree_leaves(current_params)
        l2_numerator = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
        l2_denominator = sum(leaf.size for leaf in leaves)
        l2 = l2_numerator / max(l2_denominator, 1)
        total = rank_loss + l2_weight * l2

        correct = (score_delta * outcome_delta > 0).astype(scores.dtype)
        pairwise_accuracy = jnp.sum(valid_float * correct) / jnp.maximum(
            pair_count, 1.0)
        score_centered = scores - jnp.mean(scores, axis=1, keepdims=True)
        outcome_centered = outcomes - jnp.mean(
            outcomes, axis=1, keepdims=True)
        correlation = jnp.sum(
            score_centered * outcome_centered, axis=1) / jnp.maximum(
                jnp.linalg.norm(score_centered, axis=1)
                * jnp.linalg.norm(outcome_centered, axis=1), 1e-8)
        correlation = jnp.sum(
            informative.astype(scores.dtype) * correlation) / jnp.maximum(
                jnp.sum(informative.astype(scores.dtype)), 1.0)
        selected_outcome = jnp.take_along_axis(
            outcomes, jnp.argmax(scores, axis=1)[:, None], axis=1)[:, 0]
        top_regret = jnp.sum(
            informative.astype(scores.dtype)
            * (jnp.max(outcomes, axis=1) - selected_outcome)) / jnp.maximum(
                jnp.sum(informative.astype(scores.dtype)), 1.0)
        return total, {
            'counterfactual_rank/loss': rank_loss,
            'counterfactual_rank/l2': l2,
            'counterfactual_rank/pair_count': pair_count,
            'counterfactual_rank/pairwise_accuracy': pairwise_accuracy,
            'counterfactual_rank/score_vs_outcome_pearson': correlation,
            'counterfactual_rank/fixed_state_score_std': jnp.mean(
                jnp.std(scores, axis=1)),
            'counterfactual_rank/top_action_regret': top_regret,
            'counterfactual_rank/train_informative_anchor_fraction': jnp.mean(
                informative.astype(scores.dtype)),
        }

      (loss, metrics), gradients = jax.value_and_grad(
          loss_fn, has_aux=True)(params)
      pair_count = metrics['counterfactual_rank/pair_count']

      def apply_update(operand):
        current_params, current_opt_state, current_gradients = operand
        updates, new_opt_state = optimizer.update(
            current_gradients, current_opt_state, current_params)
        return optax.apply_updates(current_params, updates), new_opt_state

      new_params, new_opt_state = jax.lax.cond(
          pair_count > 0,
          apply_update,
          lambda operand: (operand[0], operand[1]),
          (params, opt_state, gradients))
      metrics = {
          **metrics,
          'counterfactual_rank/total_loss': loss,
          'counterfactual_rank/did_update': (pair_count > 0).astype(jnp.float32),
      }
      return new_params, new_opt_state, metrics

    return jax.jit(update)

  def train_counterfactual_ranker(self, batch, updates=1):
    """Train ``u_task`` on a fixed-shape exact-state ranking batch."""
    if self._counterfactual_rank_update is None:
      raise RuntimeError(
          'train_counterfactual_ranker requires counterfactual_rank mode.')
    if updates <= 0:
      raise ValueError('counterfactual rank updates must be positive.')
    observations = jnp.asarray(batch.observations)
    actions = jnp.asarray(batch.actions)
    outcomes = jnp.asarray(batch.outcomes)
    informative = jnp.asarray(batch.informative)
    metrics = None
    completed_updates = 0
    for _ in range(int(updates)):
      params, opt_state, metrics = self._counterfactual_rank_update(
          self._state.u_task_params, self._state.u_task_opt_state,
          observations, actions, outcomes, informative)
      did_update = int(float(metrics['counterfactual_rank/did_update']) > 0.5)
      completed_updates += did_update
      self._state = self._state._replace(
          u_task_params=params,
          u_task_opt_state=opt_state,
          counterfactual_rank_updates=(
              self._state.counterfactual_rank_updates + did_update))
    result = {name: float(value) for name, value in metrics.items()}
    result['counterfactual_rank/updates_this_event'] = float(completed_updates)
    result['counterfactual_rank/updates_total'] = float(
        self._state.counterfactual_rank_updates)
    return result

  def score_counterfactual_batch(self, batch):
    """Return raw task-local head scores for every anchor/action candidate."""
    observations = jnp.asarray(batch.observations)
    actions = jnp.asarray(batch.actions)
    shape = observations.shape[:2]
    scores = self._decomp_nets.apply_u_task(
        self._state.u_task_params,
        observations.reshape((-1, observations.shape[-1])),
        actions.reshape((-1, actions.shape[-1])))[:, 0]
    return np.asarray(scores.reshape(shape))

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
          None,
          transitions,
          diagnostic_key)
      self._last_diagnostic_metrics = {
          name: float(value) for name, value in diagnostic_metrics.items()}
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
    if self._action_effect_enabled:
      critic_bundle['u_task'] = self._state.u_task_params
    available = {'policy': actor_params, 'critic': critic_bundle}
    return [available[n] for n in names]

  def score_actions(self, observation, actions):
    """Score same-state candidates with the actor's actual objective."""
    actions = jnp.asarray(actions)
    obs = jnp.repeat(
        jnp.asarray(observation)[None, :], actions.shape[0], axis=0)
    q_action = self._decomp_nets.apply_paired_score(
        self._state.b_shared_params, self._state.h_phi_params,
        self._state.phi_task_params, self._state.psi_params, obs, actions)
    if not self._action_effect_enabled:
      return q_action
    effect = self._decomp_nets.apply_u_task(
        self._state.u_task_params, obs, actions)
    if self._action_effect_target_mode == 'raw_horizon':
      advantage = (
          effect[:, 0]
          + self._outcome_success_actor_weight * jax.nn.sigmoid(effect[:, 1]))
    elif self._action_effect_target_mode == 'counterfactual_rank':
      active = (self._state.counterfactual_rank_updates > 0).astype(effect.dtype)
      advantage = active * effect[:, 0]
    else:
      goal_repr = self._decomp_nets.apply_psi(self._state.psi_params, obs)
      goal_repr = goal_repr / jnp.maximum(
          jnp.linalg.norm(goal_repr, axis=1, keepdims=True), 1e-8)
      advantage = jnp.tanh(
          jnp.sum(effect * goal_repr, axis=1)
          / max(self._action_effect_temperature, 1e-8))
    if self._action_effect_actor_mode == 'effect_only':
      return self._action_effect_actor_weight * advantage
    q_scale = jnp.maximum(
        self._state.control_q_scale_ema,
        self._action_effect_normalization_eps)
    return q_action / q_scale + self._action_effect_actor_weight * advantage

  def save(self):
    return self._state

  def restore(self, state):
    # Checkpoints written before the counterfactual experiment have one fewer
    # trailing field. Preserve their exact parameters and append an inactive
    # counter only when such a legacy tuple is restored.
    if len(state) == len(DecomposedTrainingState._fields) - 1:
      state = DecomposedTrainingState(
          *state, counterfactual_rank_updates=None)
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
  def last_diagnostic_metrics(self):
    """Metrics emitted on this step, empty between diagnostic events."""
    return getattr(self, '_last_diagnostic_metrics', {})

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
  def u_task_params(self):
    return self._state.u_task_params

  @property
  def policy_params(self):
    return self._state.policy_params

  @property
  def policy_opt_state(self):
    return self._state.policy_opt_state

  @property
  def alpha_params(self):
    return self._state.alpha_params

  @property
  def alpha_opt_state(self):
    return self._state.alpha_optimizer_state

  @property
  def last_metrics(self):
    return self._last_metrics
