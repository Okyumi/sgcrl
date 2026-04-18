"""RL representation metrics in JAX.

Metrics organized by computational cost:

  FREQUENT (every eval_every steps):
    - weight_norm_l2: L2 norm of all parameters
    - final_layer_norm: L2 norm of the actor's policy head weights
    - feature_entropy: Shannon entropy of |feature| distributions
    - gini_sparsity: Gini coefficient measuring feature sparsity

  OCCASIONAL (every 5 * eval_every steps):
    - feature_rank: effective rank via SVD (tau=0.99)
    - nrc1 / nrc2: Neural Rank Collapse metrics
    - dormant_ratio: fraction of neurons with negligible activation
"""
import jax
import jax.numpy as jnp
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Parameter-level metrics (no forward pass needed)
# ═══════════════════════════════════════════════════════════════════════

def weight_norm_l2(params) -> float:
  """L2 norm of a parameter pytree."""
  leaves = jax.tree_util.tree_leaves(params)
  total = sum(float(jnp.sum(p ** 2)) for p in leaves)
  return float(np.sqrt(total))


def final_layer_norm(params) -> float:
  """L2 norm of the actor's policy head (NormalTanhDistribution) weights.

  Searches the param pytree for the 'Normal' module's loc layer weight.
  Returns -1.0 if not found.
  """
  # Walk all leaves looking for 'Normal' scope with a 'w' param
  flat, treedef = jax.tree_util.tree_flatten_with_path(params)
  for path, val in flat:
    path_str = '/'.join(str(p) for p in path)
    if "['Normal']" in path_str and "['w']" in path_str and "['linear']" in path_str:
      # Take only the loc layer (first linear), skip linear_1 (scale layer)
      if "['linear_1']" not in path_str:
        return float(jnp.sqrt(jnp.sum(val ** 2)))
  return -1.0


# ═══════════════════════════════════════════════════════════════════════
# Feature-level metrics (need forward pass)
# ═══════════════════════════════════════════════════════════════════════

def feature_entropy(features: jnp.ndarray, eps: float = 1e-8) -> float:
  """Shannon entropy of |feature| distribution. Higher = more uniform."""
  X = jnp.abs(features)
  Z = jnp.maximum(jnp.sum(X, axis=1, keepdims=True), eps)
  p = X / Z
  H = -jnp.sum(p * jnp.log(p + eps), axis=1)
  return float(jnp.mean(H))


def gini_sparsity(features: jnp.ndarray, eps: float = 1e-12) -> float:
  """Gini coefficient. Higher = sparser features."""
  X = jnp.abs(features)
  B, D = X.shape
  Xs = jnp.sort(X, axis=1)
  row_sums = jnp.maximum(jnp.sum(Xs, axis=1), eps)
  idx = jnp.arange(1, D + 1, dtype=X.dtype)
  weights = (D - idx + 0.5) / D
  numer = jnp.sum(Xs * weights[None, :], axis=1)
  G = 1 - 2 * numer / row_sums
  return float(jnp.mean(G))


def feature_rank(features: jnp.ndarray, tau: float = 0.99) -> int:
  """Effective rank: min k s.t. top-k singular values explain >= tau variance."""
  X = features - jnp.mean(features, axis=0, keepdims=True)
  _, s, _ = jnp.linalg.svd(X, full_matrices=False)
  s2 = s * s
  denom = jnp.maximum(jnp.sum(s2), 1e-12)
  cumsum = jnp.cumsum(s2) / denom
  k = int(jnp.argmax(cumsum >= tau) + 1)
  return k


def compute_nrc1(features: jnp.ndarray, target_dim: int) -> float:
  """NRC1: how much features lie in a target_dim-dimensional subspace."""
  H = features
  H_centered = H - jnp.mean(H, axis=0, keepdims=True)
  H_norm = jnp.maximum(jnp.linalg.norm(H_centered, axis=1, keepdims=True), 1e-8)
  H_normalized = H_centered / H_norm
  _, S, Vh = jnp.linalg.svd(H_centered, full_matrices=False)
  PCs = Vh[:target_dim, :]
  P = PCs.T @ PCs
  H_proj = H_normalized @ P
  nrc1 = jnp.sum((H_proj - H_normalized) ** 2) / H.shape[0]
  return float(nrc1)


def compute_nrc2(features: jnp.ndarray, final_weights: jnp.ndarray) -> float:
  """NRC2: alignment between features and the final layer's row space."""
  H = features
  H_centered = H - jnp.mean(H, axis=0, keepdims=True)
  H_norm = jnp.maximum(jnp.linalg.norm(H_centered, axis=1, keepdims=True), 1e-8)
  H_normalized = H_centered / H_norm
  _, _, Vh = jnp.linalg.svd(final_weights, full_matrices=False)
  P = Vh.T @ Vh
  H_proj = H_normalized @ P
  nrc2 = jnp.sum((H_proj - H_normalized) ** 2) / H.shape[0]
  return float(nrc2)


def dormant_ratio(features: jnp.ndarray, dormant_pct: float = 0.025) -> float:
  """Fraction of neurons with negligible activation."""
  mean_act = jnp.mean(jnp.abs(features), axis=0)
  avg_neuron = jnp.mean(mean_act)
  normalized = mean_act / jnp.maximum(avg_neuron, 1e-9)
  n_dormant = jnp.sum(normalized <= dormant_pct)
  return float(n_dormant / features.shape[1])


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_features(networks, params, obs, actions):
  """Extract repr_dim features from critic encoders."""
  sa_repr, g_repr, _ = networks.repr_fn(params, obs, actions)
  return sa_repr, g_repr


def _get_encoder_final_weights(q_params, encoder_name):
  """Extract the final Dense layer's weight from a critic encoder."""
  flat, _ = jax.tree_util.tree_flatten_with_path(q_params)
  # Find the highest-indexed linear 'w' under the encoder scope
  best_w = None
  best_idx = -1
  for path, val in flat:
    path_str = '/'.join(str(p) for p in path)
    if encoder_name in path_str and "['w']" in path_str and 'linear' in path_str:
      # Extract the index (linear=0, linear_1=1, etc.)
      for p in path:
        s = str(p)
        if 'linear' in s:
          suffix = s.replace("['linear", '').replace("']", '').replace('_', '')
          idx = int(suffix) if suffix.isdigit() else 0
          if idx > best_idx:
            best_idx = idx
            best_w = val
          break
  return best_w


# ═══════════════════════════════════════════════════════════════════════
# Main compute function
# ═══════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    networks, actor_params, q_params,
    obs_batch, action_batch, obs_dim,
    level='frequent'):
  """Compute RL metrics at the specified frequency level.

  Args:
    level: 'frequent' or 'occasional'.
  Returns:
    dict of metric_name -> value.
  """
  metrics = {}

  # ---- FREQUENT (no forward pass) ----
  metrics['actor/weight_norm'] = weight_norm_l2(actor_params)
  metrics['critic/weight_norm'] = weight_norm_l2(q_params)
  metrics['actor/final_layer_norm'] = final_layer_norm(actor_params)

  # Feature extraction (forward pass through critic encoders)
  sa_feats, g_feats = extract_features(networks, q_params, obs_batch, action_batch)

  # Feature entropy and Gini
  metrics['critic_sa/entropy'] = feature_entropy(sa_feats)
  metrics['critic_g/entropy'] = feature_entropy(g_feats)
  metrics['critic_sa/gini'] = gini_sparsity(sa_feats)
  metrics['critic_g/gini'] = gini_sparsity(g_feats)

  if level == 'occasional':
    # ---- OCCASIONAL ----
    action_dim = action_batch.shape[-1]

    metrics['critic_sa/feature_rank'] = feature_rank(sa_feats, tau=0.99)
    metrics['critic_g/feature_rank'] = feature_rank(g_feats, tau=0.99)

    metrics['critic_sa/nrc1'] = compute_nrc1(sa_feats, target_dim=action_dim)
    metrics['critic_g/nrc1'] = compute_nrc1(g_feats, target_dim=1)

    sa_final_w = _get_encoder_final_weights(q_params, 'sa_encoder')
    g_final_w = _get_encoder_final_weights(q_params, 'g_encoder')
    if sa_final_w is not None:
      metrics['critic_sa/nrc2'] = compute_nrc2(sa_feats, sa_final_w)
    if g_final_w is not None:
      metrics['critic_g/nrc2'] = compute_nrc2(g_feats, g_final_w)

    metrics['critic_sa/dormant_ratio'] = dormant_ratio(sa_feats, dormant_pct=1e-5)
    metrics['critic_g/dormant_ratio'] = dormant_ratio(g_feats, dormant_pct=1e-5)

  return metrics
