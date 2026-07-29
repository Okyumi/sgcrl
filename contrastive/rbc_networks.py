"""Networks for Residual Bellman-Calibrated DCC (RBC-DCC).

RBC-DCC keeps the existing decomposed contrastive representation and adds two
task-local scalar Bellman heads:

  Q_i(s, a, g) = softplus(rho_i) * stop_gradient(f_C(s, a, g)) + b_i
                 + Delta_i(stop_gradient(z_shared), z_task,
                           stop_gradient(z_goal)).

The existing :mod:`contrastive.decomposed_networks` module is not modified.
This wrapper is used only when ``critic_mode='rbc_decomposed'``.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp

from contrastive import decomposed_networks

Params = Any


@dataclasses.dataclass
class RBCNetworks:
  """Pure network bundle for the contrastive base and Bellman residual."""

  decomposed: decomposed_networks.DecomposedCriticNetworks
  init_residual: Callable[[jax.Array], Params]
  apply_residual: Callable[[Params, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                           jnp.ndarray]
  init_calibration: Callable[[], Params]
  apply_components: Callable[..., tuple]
  apply_paired_score: Callable[..., jnp.ndarray]
  apply_hybrid_q: Callable[..., jnp.ndarray]


def inverse_softplus(value: float) -> float:
  """Return ``x`` such that ``softplus(x) == value``."""
  return float(jnp.log(jnp.expm1(jnp.asarray(value, dtype=jnp.float32))))


def polyak_update(target_params: Params, online_params: Params, tau: float):
  """Return ``(1 - tau) * target + tau * online`` for a parameter pytree."""
  if not 0.0 <= tau <= 1.0:
    raise ValueError(f'tau must lie in [0, 1], got {tau}.')
  return jax.tree_util.tree_map(
      lambda target, online: target * (1.0 - tau) + online * tau,
      target_params, online_params)


def make_rbc_networks(
    spec,
    obs_dim: int,
    *,
    repr_dim: int = 64,
    use_residual: bool = True,
    network_width: int = 1024,
    critic_depth: int = 4,
    phi_task_width: int = 256,
    phi_task_depth: int = 4,
    energy_fn: str = 'inner_product',
    repr_norm: bool = False,
    combine_mode: str = 'add',
    goal_encoder_mode: str = 'shared',
    bellman_hidden_dim: int = 256,
) -> RBCNetworks:
  """Build RBC-DCC networks without changing the legacy DCC factory."""
  if combine_mode != 'add':
    raise ValueError(
        "RBC-DCC v1 supports combine_mode='add' only; got "
        f'{combine_mode!r}.')
  if goal_encoder_mode != 'shared':
    raise ValueError(
        "RBC-DCC v1 supports goal_encoder_mode='shared' only; got "
        f'{goal_encoder_mode!r}.')
  if energy_fn not in ('inner_product', 'l2'):
    raise ValueError(f'Unsupported energy_fn={energy_fn!r}.')
  if bellman_hidden_dim <= 0:
    raise ValueError('bellman_hidden_dim must be positive.')

  decomp = decomposed_networks.make_decomposed_networks(
      spec,
      obs_dim=obs_dim,
      repr_dim=repr_dim,
      use_residual=use_residual,
      network_width=network_width,
      critic_depth=critic_depth,
      phi_task_width=phi_task_width,
      phi_task_depth=phi_task_depth,
      energy_fn=energy_fn,
      repr_norm=repr_norm,
      combine_mode=combine_mode,
      goal_encoder_mode=goal_encoder_mode,
  )

  def _residual_fn(z_shared, z_task, z_goal):
    x = jnp.concatenate([z_shared, z_task, z_goal], axis=-1)

    def _head(name):
      hidden = hk.Linear(bellman_hidden_dim, name=f'{name}_linear_0')(x)
      hidden = jax.nn.swish(hidden)
      hidden = hk.Linear(bellman_hidden_dim, name=f'{name}_linear_1')(hidden)
      hidden = jax.nn.swish(hidden)
      return hk.Linear(
          1,
          w_init=hk.initializers.Constant(0.0),
          b_init=hk.initializers.Constant(0.0),
          name=f'{name}_out',
      )(hidden)

    return jnp.concatenate([_head('delta_1'), _head('delta_2')], axis=-1)

  residual = hk.without_apply_rng(hk.transform(_residual_fn))
  dummy = jnp.zeros((1, repr_dim), dtype=jnp.float32)

  def init_residual(key):
    return residual.init(key, dummy, dummy, dummy)

  def init_calibration():
    return {
        'rho': jnp.full(
            (2,), inverse_softplus(1.0), dtype=jnp.float32),
        'bias': jnp.zeros((2,), dtype=jnp.float32),
    }

  def apply_components(p_b, p_h_phi, p_task, p_psi, obs, action):
    hidden = decomp.apply_b_shared(p_b, obs, action)
    z_shared = decomp.apply_h_phi(p_h_phi, hidden)
    z_task = decomp.apply_phi_task(p_task, obs, action)
    z_goal = decomp.apply_psi(p_psi, obs)
    return z_shared, z_task, z_goal

  def _paired_from_components(z_shared, z_task, z_goal):
    z_sa = z_shared + z_task
    if repr_norm:
      eps = 1e-8
      z_sa = z_sa / jnp.maximum(
          jnp.linalg.norm(z_sa, axis=-1, keepdims=True), eps)
      z_goal = z_goal / jnp.maximum(
          jnp.linalg.norm(z_goal, axis=-1, keepdims=True), eps)
    if energy_fn == 'l2':
      return -jnp.sqrt(
          jnp.sum(jnp.square(z_sa - z_goal), axis=-1) + 1e-6)
    return jnp.sum(z_sa * z_goal, axis=-1)

  def apply_paired_score(p_b, p_h_phi, p_task, p_psi, obs, action):
    return _paired_from_components(
        *apply_components(p_b, p_h_phi, p_task, p_psi, obs, action))

  def apply_hybrid_q(
      p_b,
      p_h_phi,
      p_task,
      p_psi,
      p_residual,
      p_calibration,
      obs,
      action,
  ):
    # Stop persistent *parameters*, not the score tensor. This blocks TD
    # gradients into transferred encoders while preserving dQ/da for the
    # actor. A blanket stop_gradient(base) would accidentally remove the
    # contrastive component from action selection.
    stop_tree = lambda tree: jax.tree_util.tree_map(
        jax.lax.stop_gradient, tree)
    p_b_const = stop_tree(p_b)
    p_h_phi_const = stop_tree(p_h_phi)
    p_psi_const = stop_tree(p_psi)
    p_task_const = stop_tree(p_task)

    z_shared, z_task, z_goal = apply_components(
        p_b_const, p_h_phi_const, p_task, p_psi_const, obs, action)
    z_task_base = decomp.apply_phi_task(p_task_const, obs, action)
    base = _paired_from_components(z_shared, z_task_base, z_goal)
    delta = residual.apply(
        p_residual,
        jax.lax.stop_gradient(z_shared),
        z_task,
        jax.lax.stop_gradient(z_goal),
    )
    slopes = jax.nn.softplus(p_calibration['rho'])
    return base[:, None] * slopes[None, :] + (
        p_calibration['bias'][None, :]) + delta

  return RBCNetworks(
      decomposed=decomp,
      init_residual=init_residual,
      apply_residual=residual.apply,
      init_calibration=init_calibration,
      apply_components=apply_components,
      apply_paired_score=apply_paired_score,
      apply_hybrid_q=apply_hybrid_q,
  )
