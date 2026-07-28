"""Goal-conditioned SAC networks (baseline for continual CRL comparison).

#Modified Adapted from JAXGCRL's SAC networks
(``sac/reference/jaxgcrl_sac_networks.py`` :: ``make_sac_networks`` /
``make_q_network`` / ``make_policy_network``), which itself wraps
``brax.training.agents.sac``.  Signatures and tensor layouts match:

  * twin scalar Q:  (obs, action) → [B, n_critics=2], concatenated from
    independent heads Q1, Q2.
  * policy: obs → diagonal Gaussian with tanh squashing
    (``NormalTanhDistribution`` here is the haiku/acme equivalent of
    brax's ``distribution.NormalTanhDistribution``).

Two deliberate deviations (both marked below):

  1. The body uses the CRL codebase's ``ResidualMLP``
     (LayerNorm + Swish + residual, depth-4 default) rather than the flat
     ``linen.MLP`` + ReLU stack JAXGCRL uses.  This keeps the actor
     architecture identical to contrastive.networks.make_networks so the
     CKA knowledge-pool decomposition on policy params transfers without
     any per-leaf changes.  Setting ``use_residual=False`` recovers the
     JAXGCRL-style plain MLP.
  2. The observation fed to the networks is the HER concatenation
     ``[state, goal]``; brax's SAC uses a flat env observation.  This is
     a data-pipeline choice, not an architectural one — the Q/policy
     modules still see a single rank-2 input tensor.

``apply_policy_and_sample`` and ``apply_policy_k_sample_argmax`` are the
sampling-side analogues of JAXGCRL's ``make_inference_fn`` (deterministic
``mode()`` for eval, stochastic ``sample()`` for behavior; the
K-sample-argmax variant scores by ``min(Q₁, Q₂)`` to mirror brax's twin-Q
convention in the evaluator).
"""
import dataclasses
from typing import Callable, Optional, Tuple

from acme.jax import networks as networks_lib
from acme.jax import utils as acme_utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from distributional import NormalTanhDistribution
from contrastive.networks import ResidualMLP


@dataclasses.dataclass
class SACNetworks:
  """Network / pure-function bundle for goal-conditioned SAC."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None
  actor_repr_fn: Optional[Callable] = None
  # Kept (as None) so SACNetworks can be passed where ContrastiveNetworks is
  # expected without breaking attribute lookups.
  repr_fn: Optional[Callable] = None
  critic_hidden_repr_fn: Optional[Callable] = None


def apply_policy_and_sample(networks, eval_mode: bool = False):
  """Return a pure function that samples an action from the policy."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if sample_fn is None:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return apply_and_sample


def apply_policy_k_sample_argmax(networks, k: int = 20):
  """K-sample-argmax evaluation: sample K actions, pick highest-Q.

  #Modified CRL scores candidate actions via φ(s,a)ᵀψ(g) (a representation
  inner-product).  For SAC we read the scalar Q(s,g,a) directly.
  """
  def apply_and_select(params, key, obs):
    policy_params, q_params = params
    dist_params = networks.policy_network.apply(policy_params, obs)
    keys = jax.random.split(key, k)
    actions = jnp.stack([networks.sample(dist_params, ki) for ki in keys])

    def score_one(a):
      q = networks.q_network.apply(q_params, obs, a)  # [B, 1 or 2]
      return jnp.min(q, axis=-1)  # [B]

    scores = jax.vmap(score_one)(actions)  # [K, B]
    best = jnp.argmax(scores, axis=0)
    return actions[best, jnp.arange(actions.shape[1])]
  return apply_and_select


def make_sac_networks(
    spec,
    obs_dim: int,
    twin_q: bool = True,
    actor_min_std: float = 1e-6,
    hidden_layer_sizes: Tuple[int, ...] = (256, 256),
    use_residual: bool = True,
    network_width: int = 256,
    critic_depth: int = 4,
    actor_depth: int = 4,
):
  """Build policy + scalar (twin) Q networks for goal-conditioned SAC.

  The observation fed to the networks is ``[state, goal]`` of total width
  ``2 * obs_dim``.  ``obs_dim`` is kept as an argument for parity with the
  contrastive factory (and so the actor trunk MLPs see identical inputs).

  #Modified Critic architecture: a single MLP head consuming the full
  concatenation ``[state, goal, action]`` and returning a 1-D scalar per
  head.  With ``twin_q=True`` two independent heads are stacked to
  ``[batch, 2]``; the SAC learner takes the min over the last axis for
  clipped-double-Q bootstraps.
  """
  num_dimensions = int(np.prod(spec.actions.shape))

  def _actor_fn(obs):
    if use_residual:
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
      net = hk.Sequential([
          hk.nets.MLP(
              list(hidden_layer_sizes),
              w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
              activation=jax.nn.relu, activate_final=True),
          NormalTanhDistribution(num_dimensions, min_scale=actor_min_std),
      ])
      return net(obs)

  def _critic_fn(obs, action):
    # obs already encodes [state, goal]; SAC Q takes all three inputs.
    x = jnp.concatenate([obs, action], axis=-1)

    def _head(tag):
      if use_residual:
        mlp = ResidualMLP(1, width=network_width, depth=critic_depth, name=tag)
        out = mlp(x)
      else:
        mlp = hk.nets.MLP(
            list(hidden_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(
                1.0, 'fan_avg', 'uniform'),
            activation=jax.nn.relu, name=tag)
        out = mlp(x)
      return out  # [batch, 1]

    q1 = _head('q1')
    if twin_q:
      q2 = _head('q2')
      return jnp.concatenate([q1, q2], axis=-1)  # [batch, 2]
    return q1  # [batch, 1]

  def _actor_repr_fn(obs):
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
      mlp = hk.nets.MLP(
          list(hidden_layer_sizes),
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
          activation=jax.nn.relu, activate_final=True)
      return mlp(obs)

  # #Modified Critic hidden-feature extractor — analogue of
  # ``contrastive.networks._critic_hidden_repr_fn`` but for the SAC twin-Q
  # architecture.  Returns the penultimate-layer activations of each head
  # (shape [B, network_width] per head), i.e. the representation used by
  # the final Dense(1) regression layer that outputs the Q scalar.  The
  # submodules are built with the same Haiku names ('q1', 'q2') as in
  # ``_critic_fn`` so the same q_params pytree feeds both: in the residual
  # case we reuse ``return_hidden=True`` to stop before the Dense(1)
  # output projection; in the plain-MLP case we rebuild a body-only MLP
  # (hidden_layer_sizes, activate_final=True) so Haiku shares linear_0,
  # linear_1, ... while skipping the final linear that maps to 1.
  def _critic_hidden_repr_fn(obs, action):
    x = jnp.concatenate([obs, action], axis=-1)

    def _hidden_head(tag):
      if use_residual:
        mlp = ResidualMLP(1, width=network_width, depth=critic_depth,
                          name=tag)
        return mlp(x, return_hidden=True)  # [B, network_width]
      else:
        body = hk.nets.MLP(
            list(hidden_layer_sizes),
            w_init=hk.initializers.VarianceScaling(
                1.0, 'fan_avg', 'uniform'),
            activation=jax.nn.relu, activate_final=True, name=tag)
        return body(x)  # [B, hidden_layer_sizes[-1]]

    q1_hidden = _hidden_head('q1')
    if twin_q:
      q2_hidden = _hidden_head('q2')
      return q1_hidden, q2_hidden
    return q1_hidden, None

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  actor_repr = hk.without_apply_rng(hk.transform(_actor_repr_fn))
  critic_hidden = hk.without_apply_rng(hk.transform(_critic_hidden_repr_fn))

  dummy_action = acme_utils.add_batch_dim(acme_utils.zeros_like(spec.actions))
  dummy_obs = acme_utils.add_batch_dim(acme_utils.zeros_like(spec.observations))

  return SACNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply),
      q_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode(),
      actor_repr_fn=actor_repr.apply,
      critic_hidden_repr_fn=critic_hidden.apply,
  )
