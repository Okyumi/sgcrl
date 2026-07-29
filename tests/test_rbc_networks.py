"""Runtime network checks; skipped outside the pinned training environment."""
import types

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = pytest.importorskip('jax.numpy')
pytest.importorskip('haiku')
pytest.importorskip('optax')
pytest.importorskip('acme')
dm_specs = pytest.importorskip('dm_env.specs')

from contrastive import rbc_networks


def _spec():
  return types.SimpleNamespace(
      observations=dm_specs.Array((22,), np.float32),
      actions=dm_specs.BoundedArray(
          (4,), np.float32, minimum=-1.0, maximum=1.0))


def _bundle_and_params():
  networks = rbc_networks.make_rbc_networks(
      _spec(), obs_dim=11, repr_dim=4, network_width=16, critic_depth=4,
      phi_task_width=8, phi_task_depth=4, bellman_hidden_dim=8)
  keys = jax.random.split(jax.random.PRNGKey(0), 6)
  decomp = networks.decomposed
  params = {
      'b': decomp.init_b_shared(keys[0]),
      'h': decomp.init_h_phi(keys[1]),
      'task': decomp.init_phi_task(keys[2]),
      'psi': decomp.init_psi(keys[3]),
      'residual': networks.init_residual(keys[4]),
      'calibration': networks.init_calibration(),
  }
  return networks, params


def _batch():
  obs = jax.random.normal(jax.random.PRNGKey(7), (5, 22))
  action = jax.random.uniform(
      jax.random.PRNGKey(8), (5, 4), minval=-1.0, maxval=1.0)
  return obs, action


def test_paired_scores_match_the_full_matrix_diagonal():
  networks, p = _bundle_and_params()
  obs, action = _batch()
  full = networks.decomposed.apply_score(
      p['b'], p['h'], p['task'], p['psi'], obs, action)
  paired = networks.apply_paired_score(
      p['b'], p['h'], p['task'], p['psi'], obs, action)
  np.testing.assert_allclose(paired, jnp.diag(full), rtol=1e-5, atol=1e-5)


def test_zero_residual_initially_reduces_to_twin_calibrated_base():
  networks, p = _bundle_and_params()
  obs, action = _batch()
  base = networks.apply_paired_score(
      p['b'], p['h'], p['task'], p['psi'], obs, action)
  q = networks.apply_hybrid_q(
      p['b'], p['h'], p['task'], p['psi'], p['residual'],
      p['calibration'], obs, action)
  assert q.shape == (5, 2)
  assert np.isfinite(np.asarray(q)).all()
  np.testing.assert_allclose(q[:, 0], base, rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(q[:, 1], base, rtol=1e-5, atol=1e-5)
  assert np.all(np.asarray(jax.nn.softplus(p['calibration']['rho'])) > 0)


def test_td_value_blocks_persistent_parameter_gradients():
  networks, p = _bundle_and_params()
  obs, action = _batch()

  def loss(b_params, h_params, psi_params):
    q = networks.apply_hybrid_q(
        b_params, h_params, p['task'], psi_params, p['residual'],
        p['calibration'], obs, action)
    return jnp.mean(q)

  grads = jax.grad(loss, argnums=(0, 1, 2))(p['b'], p['h'], p['psi'])
  for tree in grads:
    for leaf in jax.tree_util.tree_leaves(tree):
      np.testing.assert_allclose(leaf, 0.0, atol=0.0)


def test_stopping_encoder_parameters_does_not_remove_action_derivative():
  networks, p = _bundle_and_params()
  obs, action = _batch()

  def q_sum(candidate_action):
    return jnp.sum(networks.apply_hybrid_q(
        p['b'], p['h'], p['task'], p['psi'], p['residual'],
        p['calibration'], obs, candidate_action))

  action_grad = jax.grad(q_sum)(action)
  assert np.isfinite(np.asarray(action_grad)).all()
  assert float(jnp.linalg.norm(action_grad)) > 0.0


def test_polyak_update_moves_each_leaf_by_tau():
  target = {'x': jnp.asarray([0.0, 2.0])}
  online = {'x': jnp.asarray([2.0, 4.0])}
  updated = rbc_networks.polyak_update(target, online, 0.25)
  np.testing.assert_allclose(updated['x'], [0.5, 2.5])


def test_nonzero_residual_routes_td_gradient_to_task_encoder():
  networks, p = _bundle_and_params()
  obs, action = _batch()
  nonzero_residual = jax.tree_util.tree_map(
      lambda leaf: leaf + 0.1, p['residual'])

  def loss(task_params):
    q = networks.apply_hybrid_q(
        p['b'], p['h'], task_params, p['psi'], nonzero_residual,
        p['calibration'], obs, action)
    return jnp.mean(q)

  task_grad = jax.grad(loss)(p['task'])
  norm = sum(
      float(jnp.sum(jnp.abs(leaf)))
      for leaf in jax.tree_util.tree_leaves(task_grad))
  assert norm > 0.0
