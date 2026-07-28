"""Representation-level RL metrics for the SAC critic.

The contrastive aggregator (:func:`contrastive.rl_metrics.compute_all_metrics`)
needs the dual ``phi(s,a)`` / ``psi(g)`` encoder exposed by
``ContrastiveNetworks.repr_fn``, which the SAC critic does not have.  The
aggregator here reuses every primitive from ``contrastive.rl_metrics``
(entropy / gini / feature_rank / NRC1 / NRC2 / dormant_ratio) but applies the
critic-side ones to the penultimate-layer activations of each Q head — the
hidden state the final ``Dense(1)`` consumes to emit the Q scalar.  Actor-side
metrics are computed by the very same helpers the CRL driver uses, since SAC
and CRL share the actor architecture.

The collaborator's patch appended these functions to
``contrastive/rl_metrics.py``.  They live in the ``sac`` package instead so
the shared contrastive/DCC module is untouched; the primitives are imported
from it rather than duplicated, so there is still exactly one implementation
of each metric.
"""
from contrastive.rl_metrics import (
    _get_actor_head_weights,
    compute_nrc1,
    compute_nrc2,
    dormant_ratio,
    extract_actor_features,
    feature_entropy,
    feature_rank,
    final_layer_norm,
    gini_sparsity,
    weight_norm_l2,
)


def extract_q_hidden_features(networks, q_params, obs, actions):
  """Extract SAC critic penultimate-layer activations.

  Returns ``(q1_hidden, q2_hidden)``, each of shape ``[batch, network_width]``
  (or ``hidden_layer_sizes[-1]`` for the plain-MLP body).  ``q2_hidden`` is
  ``None`` when ``twin_q=False``.
  """
  if getattr(networks, 'critic_hidden_repr_fn', None) is None:
    return None, None
  return networks.critic_hidden_repr_fn(q_params, obs, actions)


def _get_q_head_final_weights(q_params, head_name):
  """Final ``Dense(1)`` weight for SAC critic head ``head_name`` (``q1``/``q2``).

  Haiku keys look like ``q1/residual_m_l_p/linear_N`` (residual body) or
  ``q1/mlp/linear_N`` (plain MLP).  Scan the keys containing both the head
  name and ``linear``, keep only leaves whose output dim is 1 (the scalar
  projection), and pick the highest numeric suffix among those.
  """
  best_w = None
  best_idx = -1
  for key in q_params:
    key_str = str(key)
    if head_name in key_str and 'linear' in key_str:
      node = q_params[key]
      if not (isinstance(node, dict) and 'w' in node):
        continue
      suffix = key_str.split('linear')[-1].rstrip("']")
      idx = int(suffix.lstrip('_')) if suffix.lstrip('_').isdigit() else 0
      if node['w'].shape[-1] == 1 and idx > best_idx:
        best_idx = idx
        best_w = node['w']
  return best_w


def compute_sac_metrics(networks, actor_params, q_params, obs_batch,
                        action_batch, level='frequent'):
  """SAC analogue of ``contrastive.rl_metrics.compute_all_metrics``.

  Metric namespaces:

  * ``actor/*`` — identical to the CRL driver's actor metrics (same
    architecture, same feature extractor).
  * ``critic_q1/*``, ``critic_q2/*`` — metrics on the penultimate layer of
    each Q head (``critic_q2/*`` omitted when ``twin_q=False``).

  ``level='frequent'`` computes only the cheap metrics (parameter norms,
  entropy, gini); ``level='occasional'`` adds the expensive ones (SVD-based
  rank, NRC1/NRC2, dormant ratio).

  NRC1's ``target_dim`` is 1 for the Q heads because Q is a scalar regressor.
  NRC2 uses each head's ``Dense(1)`` weight (shape ``[width, 1]``): the column
  space is a single direction, so NRC2 measures how much of the feature
  variance lies along the Q-output direction.
  """
  metrics = {}

  # ---- FREQUENT: parameter norms ----
  metrics['actor/weight_norm'] = weight_norm_l2(actor_params)
  metrics['critic/weight_norm'] = weight_norm_l2(q_params)
  metrics['actor/final_layer_norm'] = final_layer_norm(actor_params)

  # ---- FREQUENT: feature extraction ----
  q1_feats, q2_feats = extract_q_hidden_features(
      networks, q_params, obs_batch, action_batch)
  actor_feats = extract_actor_features(networks, actor_params, obs_batch)

  for feats, ns in ((q1_feats, 'critic_q1'), (q2_feats, 'critic_q2'),
                    (actor_feats, 'actor')):
    if feats is not None:
      metrics[f'{ns}/entropy'] = feature_entropy(feats)
      metrics[f'{ns}/gini'] = gini_sparsity(feats)

  if level == 'occasional':
    action_dim = action_batch.shape[-1]

    # ---- OCCASIONAL: critic (per head) ----
    for head_name, feats in (('q1', q1_feats), ('q2', q2_feats)):
      if feats is None:
        continue
      ns = f'critic_{head_name}'
      metrics[f'{ns}/feature_rank'] = feature_rank(feats, tau=0.99)
      metrics[f'{ns}/nrc1'] = compute_nrc1(feats, target_dim=1)
      head_w = _get_q_head_final_weights(q_params, head_name)
      if head_w is not None:
        metrics[f'{ns}/nrc2'] = compute_nrc2(feats, head_w)
      metrics[f'{ns}/dormant_ratio'] = dormant_ratio(feats)

    # ---- OCCASIONAL: actor (same as CRL) ----
    if actor_feats is not None:
      metrics['actor/feature_rank'] = feature_rank(actor_feats, tau=0.99)
      metrics['actor/nrc1'] = compute_nrc1(actor_feats, target_dim=action_dim)
      actor_head_w = _get_actor_head_weights(actor_params)
      if actor_head_w is not None:
        metrics['actor/nrc2'] = compute_nrc2(actor_feats, actor_head_w)
      metrics['actor/dormant_ratio'] = dormant_ratio(actor_feats)

  return metrics
