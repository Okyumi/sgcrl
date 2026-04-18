"""RL representation metrics in JAX.

Ported from a PyTorch implementation. All functions operate on JAX arrays
and are compatible with JIT compilation where noted.

Metrics are organized by computational cost:

  FREQUENT (every eval_every steps):
    - grad_norm_l2: L2 norm of all gradients
    - weight_norm_l2: L2 norm of all parameters
    - final_layer_norm: L2 norm of the last Dense layer's weights
    - feature_entropy: Shannon entropy of |feature| distributions
    - gini_sparsity: Gini coefficient measuring feature sparsity

  OCCASIONAL (every 5 * eval_every steps):
    - feature_rank: effective rank via SVD (tau=0.99)
    - nrc1: Neural Rank Collapse metric 1
    - nrc2: Neural Rank Collapse metric 2
    - dormant_ratio: fraction of neurons with negligible activation

  RARE (every 20 * eval_every steps):
    - intrinsic_dimension: TWO-NN estimator of manifold dimension
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# FREQUENT metrics (cheap to compute)
# ═══════════════════════════════════════════════════════════════════════

def grad_norm_l2(grads) -> float:
  """L2 norm of a gradient pytree."""
  leaves = jax.tree_util.tree_leaves(grads)
  total = sum(jnp.sum(g ** 2) for g in leaves)
  return float(jnp.sqrt(total))


def weight_norm_l2(params) -> float:
  """L2 norm of a parameter pytree."""
  leaves = jax.tree_util.tree_leaves(params)
  total = sum(jnp.sum(p ** 2) for p in leaves)
  return float(jnp.sqrt(total))


def final_layer_norm(params, layer_key='Normal') -> float:
  """L2 norm of the final layer's weight matrix.

  For our actor, the final layer is inside NormalTanhDistribution,
  whose Haiku module name is 'Normal'. The first Linear inside it
  (the loc layer) has key 'Normal/~/linear'.

  Args:
    params: Haiku parameter pytree.
    layer_key: top-level key of the final module.
  Returns:
    L2 norm of the weight matrix, or -1.0 if not found.
  """
  # Direct key lookup
  if layer_key in params:
    node = params[layer_key]
    if isinstance(node, dict) and '~' in node:
      sub = node['~']
      for k in ('linear', 'linear_0'):
        if k in sub and isinstance(sub[k], dict) and 'w' in sub[k]:
          w = sub[k]['w']
          return float(jnp.sqrt(jnp.sum(w ** 2)))
  # Fallback: search all top-level keys for one containing 'Normal'
  for key in params:
    if 'Normal' in str(key) or 'normal' in str(key).lower():
      node = params[key]
      if isinstance(node, dict) and '~' in node:
        sub = node['~']
        for k in ('linear', 'linear_0'):
          if k in sub and isinstance(sub[k], dict) and 'w' in sub[k]:
            w = sub[k]['w']
            return float(jnp.sqrt(jnp.sum(w ** 2)))
  return -1.0  # return -1 to distinguish 'not found' from 'zero'


def feature_entropy(features: jnp.ndarray, eps: float = 1e-8) -> float:
  """Shannon entropy of the absolute feature distribution per sample.

  H = -Σ p_i log(p_i) where p_i = |f_i| / Σ |f_j|.
  Averaged across the batch.
  """
  X = jnp.abs(features)
  Z = jnp.sum(X, axis=1, keepdims=True)
  Z = jnp.maximum(Z, eps)
  p = X / Z
  H = -jnp.sum(p * jnp.log(p + eps), axis=1)
  return float(jnp.mean(H))


def gini_sparsity(features: jnp.ndarray, eps: float = 1e-12) -> float:
  """Gini coefficient measuring feature sparsity.

  G = 1 - 2 * Σ_i (D - i + 0.5) * x_sorted_i / (D * Σ x)
  Averaged across the batch. G ∈ [0, 1]; higher = sparser.
  """
  X = jnp.abs(features)
  B, D = X.shape
  Xs = jnp.sort(X, axis=1)
  row_sums = jnp.maximum(jnp.sum(Xs, axis=1), eps)
  idx = jnp.arange(1, D + 1, dtype=X.dtype)
  weights = (D - idx + 0.5) / D
  numer = jnp.sum(Xs * weights[None, :], axis=1)
  G = 1 - 2 * numer / row_sums
  return float(jnp.mean(G))


# ═══════════════════════════════════════════════════════════════════════
# OCCASIONAL metrics (moderate cost)
# ═══════════════════════════════════════════════════════════════════════

def feature_rank(features: jnp.ndarray, tau: float = 0.99) -> int:
  """Effective rank: minimum k such that top-k singular values
  explain >= tau fraction of total variance."""
  X = features - jnp.mean(features, axis=0, keepdims=True)
  s = jnp.linalg.svdvals(X)
  s2 = s * s
  denom = jnp.maximum(jnp.sum(s2), 1e-12)
  cumsum = jnp.cumsum(s2) / denom
  k = int(jnp.argmax(cumsum >= tau) + 1)
  return k


def compute_nrc1(features: jnp.ndarray, target_dim: int) -> float:
  """NRC1: measures how well features lie in a target_dim-dimensional subspace.

  Lower = features are more collapsed into a low-rank subspace.
  """
  H = features
  H_centered = H - jnp.mean(H, axis=0, keepdims=True)
  H_norm = jnp.maximum(jnp.linalg.norm(H_centered, axis=1, keepdims=True), 1e-8)
  H_normalized = H_centered / H_norm

  _, S, Vh = jnp.linalg.svd(H_centered, full_matrices=False)
  PCs = Vh[:target_dim, :]
  P = PCs.T @ PCs  # projection matrix

  H_proj = H_normalized @ P
  nrc1 = jnp.sum((H_proj - H_normalized) ** 2) / H.shape[0]
  return float(nrc1)


def compute_nrc2(features: jnp.ndarray, final_weights: jnp.ndarray) -> float:
  """NRC2: measures alignment between features and the final layer's row space.

  Lower = features are more aligned with what the final layer can use.

  Args:
    features: (B, D) feature matrix.
    final_weights: (out_dim, D) weight matrix of the final layer.
  """
  H = features
  H_centered = H - jnp.mean(H, axis=0, keepdims=True)
  H_norm = jnp.maximum(jnp.linalg.norm(H_centered, axis=1, keepdims=True), 1e-8)
  H_normalized = H_centered / H_norm

  _, _, Vh = jnp.linalg.svd(final_weights, full_matrices=False)
  P = Vh.T @ Vh  # projection onto row space

  H_proj = H_normalized @ P
  nrc2 = jnp.sum((H_proj - H_normalized) ** 2) / H.shape[0]
  return float(nrc2)


def dormant_ratio(features: jnp.ndarray, dormant_pct: float = 0.025) -> float:
  """Fraction of neurons with negligible mean activation.

  A neuron is 'dormant' if its mean absolute activation is less than
  dormant_pct × (average neuron activation).

  This simplified version operates on a single layer's activations.
  For multi-layer analysis, call this on each layer's features separately.
  """
  mean_act = jnp.mean(jnp.abs(features), axis=0)
  avg_neuron = jnp.mean(mean_act)
  normalized = mean_act / jnp.maximum(avg_neuron, 1e-9)
  n_dormant = jnp.sum(normalized <= dormant_pct)
  return float(n_dormant / features.shape[1])


# ═══════════════════════════════════════════════════════════════════════
# RARE metrics (expensive)
# ═══════════════════════════════════════════════════════════════════════

def _compute_twonn_mu(X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
  """Compute TWO-NN distance ratios (numpy, for large matrices)."""
  N = X.shape[0]
  X_norm = np.sum(X ** 2, axis=1, keepdims=True)
  mu_list = []
  for i in range(0, N, batch_size):
    j = min(i + batch_size, N)
    X_batch = X[i:j]
    X_batch_norm = np.sum(X_batch ** 2, axis=1, keepdims=True)
    dists = np.sqrt(np.maximum(X_batch_norm + X_norm.T - 2.0 * X_batch @ X.T, 0.0))
    sorted_dists = np.sort(dists, axis=1)
    mask = sorted_dists[:, 1] > 1e-10
    if mask.any():
      mu_list.append(sorted_dists[mask, 2] / sorted_dists[mask, 1])
  if not mu_list:
    return np.ones(1)
  return np.concatenate(mu_list)


def intrinsic_dimension(features: jnp.ndarray, batch_size: int = 1000) -> float:
  """TWO-NN estimator of intrinsic dimensionality.

  Facco et al., "Estimating the intrinsic dimension of datasets by a
  minimal neighborhood information," Scientific Reports, 2017.

  Computes the MLE of the Pareto exponent from nearest-neighbor
  distance ratios.
  """
  X = np.asarray(features)
  if X.shape[0] < 10:
    return float('nan')

  mu = _compute_twonn_mu(X, batch_size=batch_size)
  mu = np.sort(mu)
  N = len(mu)
  log_mu = np.log(mu[:-1])
  log_1mF = -np.log(1 - np.arange(1, N, dtype=np.float64) / N)

  # Fit line through origin: slope = intrinsic dimension
  n_keep = max(int(N * 0.9), 1)
  x, y = log_mu[:n_keep], log_1mF[:n_keep]
  denom = np.sum(x ** 2)
  if denom < 1e-12:
    return float('nan')
  return float(np.sum(x * y) / denom)


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction helper
# ═══════════════════════════════════════════════════════════════════════

def extract_features(networks, params, obs, actions):
  """Extract penultimate-layer features from critic encoders.

  Returns:
    sa_features: (B, repr_dim) features from the sa_encoder
    g_features: (B, repr_dim) features from the g_encoder
  """
  sa_repr, g_repr, _ = networks.repr_fn(params, obs, actions)
  return sa_repr, g_repr


def compute_all_metrics(
    networks, actor_params, q_params,
    obs_batch, action_batch, obs_dim,
    level='frequent'):
  """Compute RL metrics at the specified frequency level.

  Args:
    networks: ContrastiveNetworks.
    actor_params: actor policy parameters (composed if CKA).
    q_params: critic parameters.
    obs_batch: (B, obs_dim + goal_dim) observations.
    action_batch: (B, action_dim) actions.
    obs_dim: dimension of state (to split obs into state and goal).
    level: 'frequent', 'occasional', or 'rare'.

  Returns:
    dict of metric_name -> value.
  """
  metrics = {}

  # ---- FREQUENT (always computed) ----
  # Weight norms
  metrics['actor/weight_norm'] = weight_norm_l2(actor_params)
  metrics['critic/weight_norm'] = weight_norm_l2(q_params)
  metrics['actor/final_layer_norm'] = final_layer_norm(actor_params)

  # Extract features for feature-based metrics
  sa_feats, g_feats = extract_features(networks, q_params, obs_batch, action_batch)

  # Feature entropy and Gini
  metrics['critic_sa/entropy'] = feature_entropy(sa_feats)
  metrics['critic_g/entropy'] = feature_entropy(g_feats)
  metrics['critic_sa/gini'] = gini_sparsity(sa_feats)
  metrics['critic_g/gini'] = gini_sparsity(g_feats)

  if level in ('occasional', 'rare'):
    # ---- OCCASIONAL ----
    action_dim = action_batch.shape[-1]

    # Feature rank
    metrics['critic_sa/feature_rank'] = feature_rank(sa_feats, tau=0.99)
    metrics['critic_g/feature_rank'] = feature_rank(g_feats, tau=0.99)

    # NRC1
    metrics['critic_sa/nrc1'] = compute_nrc1(sa_feats, target_dim=action_dim)
    metrics['critic_g/nrc1'] = compute_nrc1(g_feats, target_dim=1)

    # NRC2 — need the final layer weights
    # For the critic encoders, the final layer is the output projection
    # of the sa_encoder and g_encoder.
    sa_final_w = _get_encoder_final_weights(q_params, 'sa_encoder')
    g_final_w = _get_encoder_final_weights(q_params, 'g_encoder')
    if sa_final_w is not None:
      metrics['critic_sa/nrc2'] = compute_nrc2(sa_feats, sa_final_w)
    if g_final_w is not None:
      metrics['critic_g/nrc2'] = compute_nrc2(g_feats, g_final_w)

    # Dormant ratio
    metrics['critic_sa/dormant_ratio'] = dormant_ratio(sa_feats, dormant_pct=1e-5)
    metrics['critic_g/dormant_ratio'] = dormant_ratio(g_feats, dormant_pct=1e-5)

  if level == 'rare':
    # ---- RARE ----
    # Intrinsic dimension (uses numpy internally — expensive)
    metrics['critic_sa/intrinsic_dim'] = intrinsic_dimension(sa_feats, batch_size=500)
    metrics['critic_g/intrinsic_dim'] = intrinsic_dimension(g_feats, batch_size=500)

  return metrics


def _get_encoder_final_weights(q_params, encoder_name):
  """Extract the final Dense layer's weight from an encoder in the critic params.

  Works for both plain MLP ('sa_encoder/~/linear_2') and ResidualMLP
  ('sa_encoder/~/linear' — the output projection).
  """
  # Try direct key, then fallback search
  node = q_params.get(encoder_name)
  if node is None:
    for key in q_params:
      if encoder_name in str(key):
        node = q_params[key]
        break
  if node is None or not isinstance(node, dict):
    return None
  sub = node
  if '~' not in sub:
    return None
  layers = sub['~']

  # Find the highest-numbered linear layer (the output projection)
  max_idx = -1
  max_key = None
  for key in layers:
    if key.startswith('linear'):
      suffix = key[len('linear'):]
      idx = int(suffix.lstrip('_')) if suffix.lstrip('_').isdigit() else 0
      if idx > max_idx:
        max_idx = idx
        max_key = key

  if max_key is not None and 'w' in layers[max_key]:
    return layers[max_key]['w']
  return None
