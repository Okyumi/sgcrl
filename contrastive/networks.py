"""Contrastive RL networks definition."""
import dataclasses
from typing import Optional, Tuple, Callable

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from itertools import product


# modified Tanh mean to be mapped to tanh(mean) to keep within [-1, 1]
from distributional import NormalTanhDistribution

# ===========================================================================
# Head identification (Fix E in docs/audit_apr26_cka_sgcrl.md)
# ===========================================================================
#
# The CKA path masks gradients on the body (encoder) of v_k, retaining
# gradients only for the actor head. Detection used to be a brittle
# substring match against the literal string ``'Normal'`` scattered across
# the codebase. We centralise the tag here so that any future renames
# (e.g. swapping in a different head type) only need to touch this list.
#
# A leaf is part of the actor head iff its Haiku path string contains any
# of ``ACTOR_HEAD_PATH_TAGS``. The current head module is
# ``NormalTanhDistribution`` whose submodule name is ``'Normal'``.
# Adding a new head module: register its name (or a unique substring of
# its submodule path) here.
ACTOR_HEAD_PATH_TAGS: Tuple[str, ...] = ('Normal',)


def is_actor_head_path(path_str: str) -> bool:
  """Return True if a Haiku path string belongs to the policy head."""
  return any(tag in path_str for tag in ACTOR_HEAD_PATH_TAGS)


@dataclasses.dataclass
class ContrastiveNetworks:
  """Network and pure functions for the Contrastive RL agent."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  repr_fn: Callable[Ellipsis, networks_lib.NetworkOutput]
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None
  actor_repr_fn: Optional[Callable] = None  # (params, obs) -> trunk features
  critic_hidden_repr_fn: Optional[Callable] = None  # (params, obs, act) -> (sa_hidden, g_hidden)


def apply_policy_and_sample(
    networks,
    eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return apply_and_sample


def apply_policy_k_sample_argmax(networks, k=20):
  """K-sample-argmax: sample K actions, score with critic, pick the best.

  The critic inner product phi(s,a)^T psi(g) scores how well action a moves
  from state s toward goal g.  Sampling K candidates and picking the
  highest-scoring one can improve evaluation performance over the
  deterministic policy mean.

  Args:
    networks: ContrastiveNetworks.
    k: number of candidate actions to sample.
  Returns:
    A function (params, key, obs) -> action, where
    params = (policy_params, q_params).
  """
  def apply_and_select(params, key, obs):
    policy_params, q_params = params
    dist_params = networks.policy_network.apply(policy_params, obs)
    keys = jax.random.split(key, k)
    # Sample K actions: [K, batch, action_dim]
    actions = jnp.stack([networks.sample(dist_params, ki) for ki in keys])

    def score_one(a):
      logits, _, _ = networks.q_network.apply(q_params, obs, a)
      if len(logits.shape) == 3:  # twin_q
        logits = jnp.min(logits, axis=-1)
      return jnp.diag(logits)  # [batch]

    scores = jax.vmap(score_one)(actions)  # [K, batch]
    best_k = jnp.argmax(scores, axis=0)   # [batch]
    best_actions = actions[best_k, jnp.arange(actions.shape[1])]
    return best_actions
  return apply_and_select


# ═══════════════════════════════════════════════════════════════════════
# Residual MLP (matches the architecture from Wang et al., 2025:
# "1000 Layer Networks for Self-Supervised RL")
#
# Each residual block = 4x (Dense → LayerNorm → Swish) + skip.
# Total depth = 4 * num_blocks (Dense layers).  The default "plain MLP"
# is depth=0 blocks (residual disabled), falling back to hk.nets.MLP.
# ═══════════════════════════════════════════════════════════════════════

class ResidualMLP(hk.Module):
  """Residual MLP with LayerNorm + Swish, matching the scaling-CRL paper.

  Architecture:
    1. Input projection: Dense(width) → LayerNorm → Swish
    2. N residual blocks, each containing 4 × (Dense → LayerNorm → Swish)
       with a skip connection:  h_{i+1} = h_i + block(h_i)
    3. Output projection: Dense(output_dim)

  With depth=4: one residual block (4 Dense layers in the block, plus
  input projection + output = 6 total Dense layers).

  Args:
    output_dim: final output dimensionality.
    width: hidden dimension of all intermediate Dense layers.
    depth: total Dense layers inside residual blocks. Must be a multiple
           of 4 (block_size). depth=4 gives 1 block, depth=8 gives 2, etc.
    name: Haiku module name.
  """

  def __init__(self, output_dim: int, width: int = 256, depth: int = 4,
               name: Optional[str] = None):
    super().__init__(name=name)
    assert depth >= 4 and depth % 4 == 0, (
        f'depth must be a positive multiple of 4, got {depth}')
    self._output_dim = output_dim
    self._width = width
    self._num_blocks = depth // 4

  def __call__(self, x: jnp.ndarray,
               return_hidden: bool = False) -> jnp.ndarray:
    w_init = hk.initializers.VarianceScaling(
        1.0 / 3.0, 'fan_in', 'uniform')  # LeCun uniform
    b_init = hk.initializers.Constant(0.0)

    # --- input projection ---
    x = hk.Linear(self._width, w_init=w_init, b_init=b_init)(x)
    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
    x = jax.nn.swish(x)

    # --- residual blocks ---
    for _ in range(self._num_blocks):
      identity = x
      for _ in range(4):
        x = hk.Linear(self._width, w_init=w_init, b_init=b_init)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.swish(x)
      x = x + identity

    if return_hidden:
      return x  # [batch, width] — pre-output-projection hidden state

    # --- output projection ---
    x = hk.Linear(self._output_dim, w_init=w_init, b_init=b_init)(x)
    return x


def make_networks(
    spec,
    obs_dim,
    repr_dim = 64,
    repr_norm = False,
    repr_norm_temp = False,
    hidden_layer_sizes = (256, 256),
    actor_min_std = 1e-6,
    twin_q = False,
    use_image_obs = False,
    # --- scaling architecture flags ---
    use_residual = True,
    network_width = 256,
    critic_depth = 4,
    actor_depth = 4,
    energy_fn = 'inner_product',
    ):
  """Creates networks used by the agent.

  Args:
    use_residual: If True, use ResidualMLP (LayerNorm + Swish + skip) for
        both actor and critic encoders, matching the 1000-layer GCRL paper.
        If False (default), use plain hk.nets.MLP with ReLU (SGCRL default).
    network_width: Hidden dim for ResidualMLP. Ignored if use_residual=False.
    critic_depth: Total Dense layers in residual blocks for critic encoders.
        Must be a multiple of 4. Ignored if use_residual=False.
    actor_depth: Total Dense layers in residual blocks for the actor.
        Must be a multiple of 4. Ignored if use_residual=False.
    energy_fn: 'inner_product' (SGCRL default) or 'l2' (1000-layer paper).
  """

  num_dimensions = np.prod(spec.actions.shape, dtype=int)
  TORSO = networks_lib.AtariTorso  # pylint: disable=invalid-name

  def _unflatten_obs(obs):
    state = jnp.reshape(obs[:, :obs_dim], (-1, 64, 64, 3)) / 255.0
    goal = jnp.reshape(obs[:, obs_dim:], (-1, 64, 64, 3)) / 255.0
    return state, goal

  def _repr_fn(obs, action, hidden=None):
    # The optional input hidden is the image representations. We include this
    # as an input for the second Q value when twin_q = True, so that the two Q
    # values use the same underlying image representation.
    if hidden is None:
      if use_image_obs:
        state, goal = _unflatten_obs(obs)
        img_encoder = TORSO()
        state = img_encoder(state)
        goal = img_encoder(goal)
      else:
        state = obs[:, :obs_dim]
        goal = obs[:, obs_dim:]
    else:
      state, goal = hidden

    if use_residual:
      sa_encoder = ResidualMLP(
          repr_dim, width=network_width, depth=critic_depth,
          name='sa_encoder')
      g_encoder = ResidualMLP(
          repr_dim, width=network_width, depth=critic_depth,
          name='g_encoder')
    else:
      sa_encoder = hk.nets.MLP(
          list(hidden_layer_sizes) + [repr_dim],
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu,
          name='sa_encoder')
      g_encoder = hk.nets.MLP(
          list(hidden_layer_sizes) + [repr_dim],
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu,
          name='g_encoder')

    sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1))
    g_repr = g_encoder(goal)

    if repr_norm:
      sa_repr = sa_repr / jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
      g_repr = g_repr / jnp.linalg.norm(g_repr, axis=1, keepdims=True)

      if repr_norm_temp:
        log_scale = hk.get_parameter('repr_log_scale', [], dtype=sa_repr.dtype,
                                     init=jnp.zeros)
        sa_repr = sa_repr / jnp.exp(log_scale)
    return sa_repr, g_repr, (state, goal)

    
  def _combine_repr(sa_repr, g_repr):
    if energy_fn == 'l2':
      # Negative L2 distance (1000-layer paper)
      return -jnp.sqrt(
          jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)
          + 1e-6)
    else:
      # Inner product (SGCRL default)
      return jax.numpy.einsum('ik,jk->ij', sa_repr, g_repr)

  def _critic_fn(obs, action):
    sa_repr, g_repr, hidden = _repr_fn(obs, action)
    critic_val = _combine_repr(sa_repr, g_repr)
    if twin_q:
      sa_repr2, g_repr2, _ = _repr_fn(obs, action, hidden=hidden)
      product2 = _combine_repr(sa_repr2, g_repr2)
      # outer.shape = [batch_size, batch_size, 2]
      critic_val = jnp.stack([critic_val, product2], axis=-1)
      sa_repr = sa_repr2
      g_repr = g_repr2
    return critic_val, sa_repr, g_repr

  def _actor_fn(obs):
    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      obs = jnp.concatenate([state, goal], axis=-1)
      obs = TORSO()(obs)
    if use_residual:
      # ResidualMLP body + normalize + NormalTanhDistribution head.
      # The body's output projection is linear (no activation), so we add
      # LayerNorm + Swish before the policy head to match the scaling-CRL
      # paper's pattern where the head sees well-conditioned features.
      # Without this, the policy head receives raw linear outputs whose
      # magnitude depends on the body's skip connections, leading to poor
      # initial exploration and slow learning.
      body = ResidualMLP(
          network_width, width=network_width, depth=actor_depth,
          name='actor_body')
      trunk = body(obs)
      trunk = hk.LayerNorm(
          axis=-1, create_scale=True, create_offset=True,
          name='actor_trunk_ln')(trunk)
      trunk = jax.nn.swish(trunk)
      head = NormalTanhDistribution(num_dimensions, min_scale=actor_min_std)
      return head(trunk)
    else:
      network = hk.Sequential([
          hk.nets.MLP(
              list(hidden_layer_sizes),
              w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
              activation=jax.nn.relu,
              activate_final=True),
          NormalTanhDistribution(num_dimensions, min_scale=actor_min_std),
      ])
      return network(obs)

  # Critic hidden feature extractor: runs the encoder body (input projection
  # + residual blocks) but stops before the output projection.  Returns
  # the width-dimensional hidden state.  Shares params with q_network via
  # matching Haiku module names.
  def _critic_hidden_repr_fn(obs, action):
    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      state = TORSO()(state)
      goal = TORSO()(goal)
    else:
      state = obs[:, :obs_dim]
      goal = obs[:, obs_dim:]
    if use_residual:
      sa_encoder = ResidualMLP(
          repr_dim, width=network_width, depth=critic_depth,
          name='sa_encoder')
      g_encoder = ResidualMLP(
          repr_dim, width=network_width, depth=critic_depth,
          name='g_encoder')
      sa_hidden = sa_encoder(jnp.concatenate([state, action], axis=-1),
                             return_hidden=True)
      g_hidden = g_encoder(goal, return_hidden=True)
      return sa_hidden, g_hidden
    else:
      # Plain MLP: build a shorter MLP with just the hidden layers
      # (dropping the repr_dim output projection) to get the last
      # hidden layer activations.  Uses the same module name so
      # Haiku reuses the body parameters from the full encoder.
      sa_body = hk.nets.MLP(
          list(hidden_layer_sizes),
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu,
          activate_final=True,
          name='sa_encoder')
      g_body = hk.nets.MLP(
          list(hidden_layer_sizes),
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu,
          activate_final=True,
          name='g_encoder')
      sa_hidden = sa_body(jnp.concatenate([state, action], axis=-1))
      g_hidden = g_body(goal)
      return sa_hidden, g_hidden

  # Actor trunk feature extractor: runs the body + LayerNorm + Swish but
  # NOT the NormalTanhDistribution head.  Shares the same parameter tree as
  # policy_network; Haiku reuses params by module name.
  def _actor_repr_fn(obs):
    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      obs = jnp.concatenate([state, goal], axis=-1)
      obs = TORSO()(obs)
    if use_residual:
      body = ResidualMLP(
          network_width, width=network_width, depth=actor_depth,
          name='actor_body')
      trunk = body(obs)
      trunk = hk.LayerNorm(
          axis=-1, create_scale=True, create_offset=True,
          name='actor_trunk_ln')(trunk)
      trunk = jax.nn.swish(trunk)
      return trunk
    else:
      # Plain MLP: return the last hidden layer output (before
      # NormalTanh head).  We rebuild the same MLP as _actor_fn.
      mlp = hk.nets.MLP(
          list(hidden_layer_sizes),
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
          activation=jax.nn.relu,
          activate_final=True)
      return mlp(obs)

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))
  critic_hidden_repr = hk.without_apply_rng(hk.transform(_critic_hidden_repr_fn))
  actor_repr = hk.without_apply_rng(hk.transform(_actor_repr_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return ContrastiveNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply),
      q_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
      repr_fn=repr_fn.apply,
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode(),
      actor_repr_fn=actor_repr.apply,
      critic_hidden_repr_fn=critic_hidden_repr.apply,
      )
