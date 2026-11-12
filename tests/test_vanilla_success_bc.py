"""CPU numerical checks; runnable without Acme, MuJoCo, pytest or a GPU."""
from pathlib import Path
import sys
import ast
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from contrastive import vanilla_success_bc as bc
from experiment_configs_vanilla_crl_gate_bc import config_for_seed


class VanillaSuccessBCTest(unittest.TestCase):
  def setUp(self):
    self.key = jax.random.PRNGKey(42)
    self.obs = jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 10
    self.action = jnp.full((6, 1), 0.5)

  def test_success_filter_original_goals_and_ring_wrap(self):
    buffer = bc.empty_buffer(6, 4, 1)
    data = SimpleNamespace(action=self.action, extras={
        'outcome_task_success': jnp.array([0., 1., 0., 1., 1., 0.]),
        'outcome_retention_observation': self.obs})
    insert = jax.jit(lambda b: bc.insert(b, data))
    buffer = insert(buffer)
    self.assertEqual(int(buffer.size), 3)
    np.testing.assert_allclose(buffer.observation[:3], self.obs[jnp.array([1, 3, 4])])
    buffer = insert(insert(buffer))
    self.assertEqual(int(buffer.size), 6)
    self.assertEqual(int(buffer.index), 3)
    np.testing.assert_allclose(buffer.observation[3:], self.obs[jnp.array([1, 3, 4])])

  def test_gate_extremes_and_stopped_gradient(self):
    for score, expected in ((lambda o, a: a[:, 0], 0.),
                            (lambda o, a: o[:, 0], 1.),
                            (lambda o, a: jnp.zeros(o.shape[0]), 1.)):
      gate, _, _ = jax.jit(lambda o: bc.local_gate(
          score, o, self.action, self.key, 2))(self.obs)
      # Float32 reductions can leave tiny nonzero std for identical scores.
      # Jubail's JAX/XLA produced a 2.74e-6 gate error in the state-only case.
      # Allow that roundoff without relaxing the exact stopped-gradient check.
      self.assertTrue(np.isfinite(np.asarray(gate)).all())
      self.assertTrue(np.all((np.asarray(gate) >= 0.) & (np.asarray(gate) <= 1.)))
      np.testing.assert_allclose(gate, expected, rtol=0., atol=1e-5)
    def gate_sum(scale):
      return bc.local_gate(lambda o, a: o[:, 0] + scale * a[:, 0],
                           self.obs, self.action, self.key, 2)[0].sum()
    self.assertEqual(float(jax.grad(gate_sum)(1.)), 0.)

  def test_bc_inactive_when_empty_then_updates_policy_not_critic(self):
    networks = SimpleNamespace(
        policy_network=SimpleNamespace(apply=lambda p, o: jnp.full((o.shape[0], 1), p)),
        sample=lambda dist, key: dist,
        log_prob=lambda dist, a: -jnp.sum((dist - a) ** 2, axis=-1),
        q_network=SimpleNamespace(apply=lambda q, o, a:
            (jnp.broadcast_to(q * o[:, :1], (o.shape[0], o.shape[0])), None, None)))
    buffer = bc.empty_buffer(6, 4, 1)
    def loss(p, q, buf):
      return bc.loss(networks, p, q, buf, self.key, 2, batch_size=6)[0]
    value_grad = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    val, (pg, qg) = value_grad(0., 1., buffer)
    self.assertEqual(float(val), 0.)
    self.assertEqual(float(pg), 0.)
    buffer = buffer._replace(observation=self.obs, action=self.action,
                             size=jnp.asarray(6, jnp.int32))
    val, (pg, qg) = value_grad(0., 1., buffer)
    self.assertGreater(float(val), 0.)
    self.assertLess(float(pg), 0.)  # Gradient descent moves toward +0.5.
    self.assertEqual(float(qg), 0.)
    self.assertLess(float(loss(-0.1 * pg, 1., buffer)), float(val))

  def test_launch_flags_are_registered(self):
    repo = Path(__file__).resolve().parents[1]
    known = set()
    for file in ('run_continual_contrastive.py', 'default.py'):
      for node in ast.walk(ast.parse((repo / file).read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith('DEFINE_') and node.args
            and isinstance(node.args[0], ast.Constant)):
          known.add(node.args[0].value)
    self.assertEqual(set(config_for_seed(5)) - known, set())

  def test_paper_seeds_and_no_decomposition(self):
    for seed in range(5, 15):
      c = config_for_seed(seed)
      self.assertEqual(c['critic_mode'], 'reset')
      self.assertEqual(c['actor_mode'], 'reset')
      self.assertFalse(c['encoder_from_base'])
      self.assertNotIn('phi_task_width', c)
      self.assertEqual(c['steps_per_task'], 8_000_000)
      self.assertEqual(c['success_bc_label_mode'], 'episode_sparse_reward')
      self.assertEqual(c['success_bc_weight'], .1)


if __name__ == '__main__':
  unittest.main()
