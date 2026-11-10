"""Run on Jubail in contrastive_rl before releasing the production array.

Uses real networks, optimizers and the JIT learner on synthetic successful
transitions. No simulator or W&B needed; the subsequent smoke tests those.
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme import specs, types
from dm_env import specs as dm_specs
from contrastive.config import ContrastiveConfig
from contrastive.continual_config import ContinualConfig
from contrastive.continual_learning import ContinualContrastiveLearner
from contrastive.networks import make_networks


class LearnerIntegrationTest(unittest.TestCase):
  def test_actual_jit_update_bc_and_task_reset(self):
    spec = specs.EnvironmentSpec(
        observations=dm_specs.Array((4,), np.float32),
        actions=dm_specs.BoundedArray((1,), np.float32, -1., 1.),
        rewards=dm_specs.Array((), np.float32),
        discounts=dm_specs.BoundedArray((), np.float32, 0., 1.))
    networks = make_networks(spec, obs_dim=2, repr_dim=4,
                             network_width=16, critic_depth=4, actor_depth=4)
    config = ContrastiveConfig(obs_dim=2, batch_size=4,
                              num_sgd_steps_per_step=2, use_cpc=True,
                              random_goals=0.5, target_entropy=-1.)
    continual = ContinualConfig(success_bc_weight=.1, success_bc_critic_gate=True,
                                success_bc_critic_gate_mode='local',
                                success_buffer_capacity=16, success_bc_batch_size=4)
    def make(task_id=0, prev=None):
      return ContinualContrastiveLearner(
          networks, jax.random.PRNGKey(5 + task_id), optax.adam(3e-4),
          optax.adam(3e-4), optax.adam(3e-4), optax.adam(3e-4),
          iterator=iter(()), counter=None, logger=object(), obs_to_goal=lambda x: x,
          config=config, continual_config=continual, task_id=task_id,
          actor_mode='reset', critic_mode='reset', encoder_from_base=False,
          action_dim=1, prev_q_params=prev)
    learner = make()
    initial = learner.save()
    observation = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 20
    data = types.Transition(
        observation=observation, action=jnp.full((8, 1), .25),
        reward=jnp.ones(8), discount=jnp.ones(8), next_observation=observation,
        extras={'outcome_task_success': jnp.ones(8),
                'outcome_retention_observation': observation})
    state, metrics = learner._update_step(initial, data)
    self.assertEqual(int(state.success_buffer.size), 8)
    self.assertEqual(float(metrics['retention/bc_active']), 1.)
    self.assertNotEqual(float(metrics['retention/weighted_bc_loss']), 0.)
    for leaf in jax.tree_util.tree_leaves(state):
      self.assertTrue(np.isfinite(np.asarray(leaf)).all())
    self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(initial.v_k), jax.tree_util.tree_leaves(state.v_k))))
    self.assertFalse(any('task' in str(k) for k in state.q_params))
    next_task = make(1, state.q_params).save()
    self.assertEqual(int(next_task.success_buffer.size), 0)
    self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
        jax.tree_util.tree_leaves(state.q_params), jax.tree_util.tree_leaves(next_task.q_params))))


if __name__ == '__main__':
  unittest.main()
