"""Networks for the decomposed-critic algorithm (proposal 1).

The contrastive critic factors into three Haiku transforms instead of
a single ``q_network``:

    sa_repr = h_phi(b_shared([s; a])) + phi_task([s; a])
    g_repr  = psi(g)
    score   = sa_repr @ g_repr.T              # SGCRL inner-product (default)
            or -|| sa_repr - g_repr ||_2     # if config.energy_fn == 'l2'

with a separate dynamics head trained on a masked next-state target:

    s'_M_pred = h_dyn(b_shared([s; a]))
    L_dyn     = || s'_M_pred - select_stable(s') ||_2^2

Trainable groups (each its own optax optimiser state):

    b_shared, h_phi, h_dyn, phi_task, psi

Continual-learning rule at task boundary k > 0:

    b_shared, h_phi, h_dyn, psi    -> carry forward (transfer)
    phi_task                       -> reinitialise (task-specific)

The actor is unaffected by this module; it is still produced by
``contrastive.networks.make_networks`` and reset every task per the
existing ``actor_mode='reset'`` default.

This module is a pure addition: ``contrastive/networks.py`` and the
existing CKA / persistent / reset critic paths are not touched.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from acme.jax import networks as networks_lib
from acme.jax import utils

from contrastive import networks as contrastive_networks
from contrastive import state_mask as sm

# Type aliases for clarity in signatures.
Params = Any
Apply = Callable[..., jnp.ndarray]


# ---------------------------------------------------------------------------
# Container for the decomposed critic.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DecomposedCriticNetworks:
  """Bundle of Haiku transforms + apply fns for the decomposed critic.

  Each ``init_*`` is a function ``key -> params``; each ``apply_*`` is
  a function ``(params, ...) -> output``. Caller (the learner) holds
  five independent parameter pytrees and five matching optimiser states.
  """

  init_b_shared: Callable[[jax.Array], Params]
  apply_b_shared: Apply  # (params, obs, action) -> hidden, shape (B, hidden_dim)

  init_h_phi: Callable[[jax.Array], Params]
  apply_h_phi: Apply  # (params, hidden) -> contrastive embedding (B, repr_dim)

  init_h_dyn: Callable[[jax.Array], Params]
  apply_h_dyn: Apply  # (params, hidden) -> masked next-state pred (B, d_M)

  init_phi_task: Callable[[jax.Array], Params]
  apply_phi_task: Apply  # (params, obs, action) -> embedding (B, repr_dim)

  init_psi: Callable[[jax.Array], Params]
  apply_psi: Apply  # (params, obs) -> goal embedding (B, repr_dim)

  # Shape metadata; useful for the learner.
  obs_dim: int
  state_dim: int
  d_M: int
  repr_dim: int
  hidden_dim: int
  action_dim: int

  # Optional task-local forward action-effect head used by Bridge-DCC.
  # It predicts a displacement in the goal-embedding geometry rather than
  # identifying which replay action generated a transition.
  init_u_task: Optional[Callable[[jax.Array], Params]] = None
  apply_u_task: Optional[Apply] = None  # (params, obs, action) -> (B, repr_dim)

  # Convenience apply fns that callers may use directly.
  apply_sa_repr: Optional[Callable] = None  # composed phi(s,a)
  apply_sa_components: Optional[Callable] = None  # raw shared/task branches
  apply_score: Optional[Callable] = None    # full (B, B) critic score
  apply_score_with_components: Optional[Callable] = None
  apply_paired_score: Optional[Callable] = None  # matched triples, shape (B,)


# ---------------------------------------------------------------------------
# Builder.
# ---------------------------------------------------------------------------

def make_decomposed_networks(
    spec,
    obs_dim: int,
    *,
    repr_dim: int = 64,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    use_residual: bool = True,
    network_width: int = 1024,
    critic_depth: int = 4,
    phi_task_width: int = 256,
    phi_task_depth: int = 4,
    energy_fn: str = 'inner_product',
    repr_norm: bool = False,
    combine_mode: str = 'add',
    goal_encoder_mode: str = 'shared',
    shared_repr_scale: float = 1.0,
    action_effect_hidden_dim: int = 256,
    action_effect_output_dim: Optional[int] = None,
    action_effect_include_goal: bool = False,
) -> DecomposedCriticNetworks:
  """Build the decomposed critic networks for the given spec.

  Architecture choices mirror ``make_networks``:

  - ``b_shared`` is a ``ResidualMLP`` body of width ``network_width``
    and depth ``critic_depth``, returning the hidden representation
    BEFORE the final output projection (we replace it with two heads).
  - ``h_phi`` is a single linear layer mapping the body's hidden
    output to ``repr_dim``.
  - ``h_dyn`` is a single linear layer mapping the same hidden output
    to ``d_M`` (number of stable indices).
  - ``phi_task`` is a smaller ``ResidualMLP`` (default width 256,
    depth 2) producing a ``repr_dim``-dim embedding directly from
    ``[s; a]``. It is reset every task.
  - ``psi`` is a ``ResidualMLP`` of the same shape as the contrastive
    critic body, mapping ``g`` to ``repr_dim``. Carried across tasks.

  Args:
    spec: env spec; same as ``make_networks``.
    obs_dim: state dim (the goal occupies ``[obs_dim:]`` of the obs).
    repr_dim: contrastive embedding dim. Default 64 (sgcrl baseline).
    hidden_layer_sizes: only used if ``use_residual=False`` (plain MLP).
    use_residual: build residual bodies (default; matches scaling-CRL
        paper). Set False to use plain ``hk.nets.MLP``.
    network_width: width of the shared body (default 1024).
    critic_depth: depth (in Dense layers) of the shared body. Must be
        a multiple of 4 when ``use_residual=True``.
    phi_task_width: width of the task-specific encoder.
    phi_task_depth: depth of the task-specific encoder.
    energy_fn: 'inner_product' (SGCRL default) or 'l2'. Must match
        ``config.energy_fn``.
    repr_norm: if True, L2-normalise sa and g embeddings before scoring,
        matching ``contrastive.networks.make_networks(repr_norm=...)``.
    combine_mode: 'add' (default; z_sa = alpha*h_phi(b_shared) + phi_task) or
        'concat' (z_sa = [h_phi(b_shared); phi_task]). When 'concat',
        the goal side passes psi(g) through a learnable Linear projection
        to the resulting 2*repr_dim space so the scoring function still
        operates in matched dimensions.
    goal_encoder_mode: 'shared' (default) or 'projected'. 'projected'
        adds the Linear projection on top of psi(g) regardless of
        combine_mode (so combine='add' + goal='projected' is a useful
        ablation that tests whether the extra goal head matters in the
        additive case).
    shared_repr_scale: fixed nonnegative alpha multiplying the shared
        contrastive branch. The default 1.0 is the original DCC.

  Returns:
    A ``DecomposedCriticNetworks`` instance with all init / apply
    functions populated.
  """
  ResidualMLP = contrastive_networks.ResidualMLP  # local alias
  num_dimensions = int(np.prod(spec.actions.shape))
  state_dim = obs_dim
  d_M = sm.num_stable_indices()
  hidden_dim = network_width if use_residual else hidden_layer_sizes[-1]

  if combine_mode not in ('add', 'concat'):
    raise ValueError(
        f'combine_mode must be add or concat, got {combine_mode!r}')
  if goal_encoder_mode not in ('shared', 'projected'):
    raise ValueError(
        f'goal_encoder_mode must be shared or projected, got '
        f'{goal_encoder_mode!r}')
  if not np.isfinite(shared_repr_scale) or shared_repr_scale < 0:
    raise ValueError(
        'shared_repr_scale must be finite and nonnegative, got '
        f'{shared_repr_scale!r}')
  shared_repr_scale = float(shared_repr_scale)
  z_sa_dim = repr_dim if combine_mode == 'add' else 2 * repr_dim
  use_psi_proj = (combine_mode == 'concat') or (
      goal_encoder_mode == 'projected')

  # ---- b_shared: hidden representation, no final projection ----------------
  def _b_shared_fn(obs, action):
    state = obs[:, :obs_dim]
    if use_residual:
      body = ResidualMLP(
          repr_dim, width=network_width, depth=critic_depth,
          name='b_shared')
      hidden = body(jnp.concatenate([state, action], axis=-1),
                    return_hidden=True)
    else:
      body = hk.nets.MLP(
          list(hidden_layer_sizes),
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu, activate_final=True,
          name='b_shared')
      hidden = body(jnp.concatenate([state, action], axis=-1))
    return hidden  # (B, hidden_dim)

  # ---- h_phi: linear projection to repr_dim --------------------------------
  def _h_phi_fn(hidden):
    head = hk.Linear(repr_dim, name='h_phi')
    return head(hidden)  # (B, repr_dim)

  # ---- h_dyn: linear projection to d_M -------------------------------------
  def _h_dyn_fn(hidden):
    head = hk.Linear(d_M, name='h_dyn')
    return head(hidden)  # (B, d_M)

  # ---- phi_task: small per-task encoder, [s; a] -> repr_dim ----------------
  # Same builder pattern as b_shared / psi: ResidualMLP when use_residual
  # is True (with phi_task_depth a multiple of 4), plain MLP otherwise.
  # Plan section 6 default is phi_task_depth=4 (one residual block) at
  # phi_task_width=256, which is ~16x smaller than the b_shared body
  # (1024-width x 4-block) per the project's run config.
  def _phi_task_fn(obs, action):
    state = obs[:, :obs_dim]
    if use_residual:
      body = ResidualMLP(
          repr_dim, width=phi_task_width, depth=phi_task_depth,
          name='phi_task')
      return body(jnp.concatenate([state, action], axis=-1))
    else:
      body = hk.nets.MLP(
          [phi_task_width, phi_task_width, repr_dim],
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu, activate_final=False,
          name='phi_task')
      return body(jnp.concatenate([state, action], axis=-1))

  # ---- psi: full encoder, g -> repr_dim ------------------------------------
  def _psi_fn(obs):
    goal = obs[:, obs_dim:]
    if use_residual:
      body = ResidualMLP(
          repr_dim, width=network_width, depth=critic_depth,
          name='psi')
      return body(goal)
    else:
      body = hk.nets.MLP(
          list(hidden_layer_sizes) + [repr_dim],
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu,
          name='psi')
      return body(goal)

  # ---- u_task: task-local action credit head -------------------------------
  def _u_task_fn(obs, action):
    state = obs[:, :obs_dim]
    head_input = obs if action_effect_include_goal else state
    output_dim = repr_dim if action_effect_output_dim is None \
        else int(action_effect_output_dim)
    return hk.nets.MLP(
        [action_effect_hidden_dim, action_effect_hidden_dim, output_dim],
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        activation=jax.nn.relu,
        activate_final=False,
        name='u_task')(
            jnp.concatenate([head_input, action], axis=-1))

  # ---- optional psi projector (concat-combine or 'projected' goal mode) ---
  def _psi_proj_fn(z):
    return hk.Linear(z_sa_dim, name='psi_proj')(z)

  b_shared = hk.without_apply_rng(hk.transform(_b_shared_fn))
  h_phi = hk.without_apply_rng(hk.transform(_h_phi_fn))
  h_dyn = hk.without_apply_rng(hk.transform(_h_dyn_fn))
  phi_task = hk.without_apply_rng(hk.transform(_phi_task_fn))
  psi = hk.without_apply_rng(hk.transform(_psi_fn))
  u_task = hk.without_apply_rng(hk.transform(_u_task_fn))
  psi_proj = (hk.without_apply_rng(hk.transform(_psi_proj_fn))
              if use_psi_proj else None)

  # Dummy inputs for init.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)
  # Shape probe for hidden_dim: trace b_shared once with a key to get
  # the actual hidden output shape, in case use_residual=False yields a
  # different size from `network_width`.
  _probe_params = b_shared.init(jax.random.PRNGKey(0), dummy_obs, dummy_action)
  _probe_hidden = b_shared.apply(_probe_params, dummy_obs, dummy_action)
  hidden_dim_actual = int(_probe_hidden.shape[-1])
  dummy_hidden = jnp.zeros((1, hidden_dim_actual))

  init_b_shared = lambda key: b_shared.init(key, dummy_obs, dummy_action)
  init_h_phi = lambda key: h_phi.init(key, dummy_hidden)
  init_h_dyn = lambda key: h_dyn.init(key, dummy_hidden)
  init_phi_task = lambda key: phi_task.init(key, dummy_obs, dummy_action)
  init_u_task = lambda key: u_task.init(key, dummy_obs, dummy_action)
  # Standalone projector init for callers that want psi_proj separately.
  # ``init_psi`` below already folds this into the combined bundle when
  # use_psi_proj is True; keep a None sentinel when the projector is off.
  init_psi_proj = (
      (lambda key: psi_proj.init(key, jnp.zeros((1, repr_dim))))
      if psi_proj is not None else None)

  def _init_psi_combined(key):
    """Single init that returns the goal-encoder param bundle.

    When `psi_proj` exists, the bundle is a dict
        {'psi': psi_params, 'psi_proj': psi_proj_params}
    so the rest of the learner can keep a single ``psi_params`` slot in
    its state. When `psi_proj` is None (the default 'shared' goal mode
    with combine='add'), the bundle is just the bare psi params for
    backward compatibility with existing checkpoints.
    """
    k1, k2 = jax.random.split(key)
    p_psi = psi.init(k1, dummy_obs)
    if psi_proj is None:
      return p_psi
    p_proj = psi_proj.init(k2, jnp.zeros((1, repr_dim)))
    return {'psi': p_psi, 'psi_proj': p_proj}
  init_psi = _init_psi_combined

  def apply_sa_components(params_b_shared, params_h_phi, params_phi_task,
                          obs, action):
    """Return the unscaled shared and task-specific embeddings."""
    hidden = b_shared.apply(params_b_shared, obs, action)
    sa_shared = h_phi.apply(params_h_phi, hidden)
    sa_task = phi_task.apply(params_phi_task, obs, action)
    return sa_shared, sa_task

  def apply_sa_repr(params_b_shared, params_h_phi, params_phi_task,
                    obs, action):
    """Composed contrastive embedding (additive or concatenated)."""
    sa_shared, sa_task = apply_sa_components(
        params_b_shared, params_h_phi, params_phi_task, obs, action)
    if combine_mode == 'add':
      return shared_repr_scale * sa_shared + sa_task
    # 'concat': dim becomes 2*repr_dim, which is z_sa_dim.
    return jnp.concatenate([shared_repr_scale * sa_shared, sa_task], axis=-1)

  def apply_psi(params_psi_bundle, obs, params_psi_proj=None):
    """Goal embedding.

    `params_psi_bundle` is either the bare psi params (backward compat,
    when psi_proj is disabled) or the dict
    ``{'psi': psi_params, 'psi_proj': psi_proj_params}``.
    `params_psi_proj` is an explicit override; callers that already have
    psi_proj params separately can pass it here and the bundle is
    treated as the bare psi params.
    """
    if isinstance(params_psi_bundle, dict) and 'psi' in params_psi_bundle:
      p_psi = params_psi_bundle['psi']
      p_proj = params_psi_bundle.get('psi_proj') if params_psi_proj is None \
          else params_psi_proj
    else:
      p_psi = params_psi_bundle
      p_proj = params_psi_proj
    g = psi.apply(p_psi, obs)
    if psi_proj is not None:
      if p_proj is None:
        raise ValueError(
            'goal projection is enabled but psi_proj params were not '
            'found in the psi bundle or passed explicitly to apply_psi')
      g = psi_proj.apply(p_proj, g)
    return g

  def apply_score(params_b_shared, params_h_phi, params_phi_task, params_psi,
                  obs, action, params_psi_proj=None):
    """Full (B, B) critic score matrix.

    Mirrors ``_combine_repr`` in ``contrastive.networks``: inner product
    when ``energy_fn='inner_product'`` (the SGCRL default), negative L2
    when ``energy_fn='l2'`` (1000-layer paper).

    `params_psi` may be either the bare psi params (when no projection
    is used) or the bundle dict described in ``apply_psi``.
    """
    sa = apply_sa_repr(params_b_shared, params_h_phi, params_phi_task,
                       obs, action)
    g = apply_psi(params_psi, obs, params_psi_proj)
    if repr_norm:
      sa = sa / jnp.linalg.norm(sa, axis=1, keepdims=True)
      g = g / jnp.linalg.norm(g, axis=1, keepdims=True)
    if energy_fn == 'l2':
      return -jnp.sqrt(
          jnp.sum((sa[:, None, :] - g[None, :, :]) ** 2, axis=-1) + 1e-6)
    # inner_product (SGCRL default)
    return jnp.einsum('ik,jk->ij', sa, g)

  def apply_score_with_components(
      params_b_shared, params_h_phi, params_phi_task, params_psi,
      obs, action, params_psi_proj=None):
    """Score matrix plus branch embeddings for cheap diagnostics."""
    sa_shared, sa_task = apply_sa_components(
        params_b_shared, params_h_phi, params_phi_task, obs, action)
    if combine_mode == 'add':
      sa = shared_repr_scale * sa_shared + sa_task
    else:
      sa = jnp.concatenate(
          [shared_repr_scale * sa_shared, sa_task], axis=-1)
    g = apply_psi(params_psi, obs, params_psi_proj)
    scored_sa, scored_g = sa, g
    if repr_norm:
      scored_sa = scored_sa / jnp.maximum(
          jnp.linalg.norm(scored_sa, axis=1, keepdims=True), 1e-8)
      scored_g = scored_g / jnp.maximum(
          jnp.linalg.norm(scored_g, axis=1, keepdims=True), 1e-8)
    if energy_fn == 'l2':
      logits = -jnp.sqrt(jnp.sum(
          (scored_sa[:, None, :] - scored_g[None, :, :]) ** 2,
          axis=-1) + 1e-6)
    else:
      logits = jnp.einsum('ik,jk->ij', scored_sa, scored_g)
    return logits, sa_shared, sa_task, g

  def apply_paired_score(
      params_b_shared, params_h_phi, params_phi_task, params_psi,
      obs, action, params_psi_proj=None):
    """Efficient scores for matched (state, action, goal) triples.

    Unlike apply_score, this function does not materialize a B-by-B matrix.
    It is therefore the canonical scoring path for actor, Bellman-correction,
    and fixed-state action-ranking computations.
    """
    sa = apply_sa_repr(
        params_b_shared, params_h_phi, params_phi_task, obs, action)
    g = apply_psi(params_psi, obs, params_psi_proj)
    if repr_norm:
      sa = sa / jnp.maximum(
          jnp.linalg.norm(sa, axis=1, keepdims=True), 1e-8)
      g = g / jnp.maximum(
          jnp.linalg.norm(g, axis=1, keepdims=True), 1e-8)
    if energy_fn == 'l2':
      return -jnp.sqrt(jnp.sum(jnp.square(sa - g), axis=-1) + 1e-6)
    return jnp.sum(sa * g, axis=-1)

  bundle = DecomposedCriticNetworks(
      init_b_shared=init_b_shared, apply_b_shared=b_shared.apply,
      init_h_phi=init_h_phi, apply_h_phi=h_phi.apply,
      init_h_dyn=init_h_dyn, apply_h_dyn=h_dyn.apply,
      init_phi_task=init_phi_task, apply_phi_task=phi_task.apply,
      init_psi=init_psi, apply_psi=apply_psi,
      init_u_task=init_u_task, apply_u_task=u_task.apply,
      obs_dim=obs_dim, state_dim=state_dim, d_M=d_M,
      repr_dim=repr_dim, hidden_dim=hidden_dim_actual,
      action_dim=num_dimensions,
      apply_sa_repr=apply_sa_repr,
      apply_sa_components=apply_sa_components,
      apply_score=apply_score,
      apply_score_with_components=apply_score_with_components,
      apply_paired_score=apply_paired_score,
  )
  # Attach extra fields outside the dataclass field list so older callers
  # do not have to change.  Access via `bundle.combine_mode` etc.
  bundle.combine_mode = combine_mode
  bundle.goal_encoder_mode = goal_encoder_mode
  bundle.z_sa_dim = z_sa_dim
  bundle.init_psi_proj = init_psi_proj
  bundle.apply_psi_proj = (psi_proj.apply if psi_proj is not None else None)
  bundle.use_psi_proj = use_psi_proj
  bundle.shared_repr_scale = shared_repr_scale
  return bundle
