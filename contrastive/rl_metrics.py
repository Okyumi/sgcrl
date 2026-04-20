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

Dormant ratio threshold:
  We use τ=0.025 for Swish/SiLU activations. Rationale:
    - ReDo (Sokar et al., 2023) tested τ ∈ {0, 0.025, 0.1} for ReLU in
      Atari RL.  τ=0.1 worked best for ReLU, where dead neurons produce
      exact zeros.
    - Swish(x) = x·σ(x) has a non-zero derivative floor: even for large
      negative inputs, |swish(x)| → 0 asymptotically but never exactly.
      A healthy Swish neuron with small but non-zero pre-activations would
      be falsely flagged as dormant at τ=0.1.
    - τ=0.025 catches genuinely dormant neurons (activation score < 2.5%
      of the layer mean) while tolerating Swish's smooth tail.  This is
      more conservative than the ReLU optimum (0.1) but much less
      permissive than our previous value (1e-5, which never triggers).
    - The Nature plasticity paper (Lyle et al., 2024) and the activation-
      design paper (Abbas et al., 2026) both show Swish has lower dormancy
      than ReLU but it is NOT immune; a threshold of 0.025 is sensitive
      enough to detect it.
"""
import jax
import jax.numpy as jnp
import numpy as np


# Default dormant ratio threshold for Swish/SiLU activations.
DORMANT_THRESHOLD = 0.025


# ═══════════════════════════════════════════════════════════════════════
# Parameter-level metrics (no forward pass needed)
# ═══════════════════════════════════════════════════════════════════════

def weight_norm_l2(params) -> float:
  """L2 norm of a parameter pytree."""
  leaves = jax.tree_util.tree_leaves(params)
  total = sum(float(jnp.sum(p ** 2)) for p in leaves)
  return float(np.sqrt(total))


def final_layer_norm(params) -> float:
  """L2 norm of the actor's policy head (NormalTanhDistribution) loc weight.

  Haiku flattens module paths into top-level keys like 'Normal/linear'.
  We look for the loc layer (first linear under 'Normal').
  Returns -1.0 if not found.
  """
  # Haiku top-level keys are like 'Normal/linear', 'Normal/linear_1', etc.
  for key in params:
    key_str = str(key)
    if 'Normal' in key_str and 'linear' in key_str and 'linear_1' not in key_str:
      node = params[key]
      if isinstance(node, dict) and 'w' in node:
        w = node['w']
        return float(jnp.sqrt(jnp.sum(w ** 2)))
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


def dormant_ratio(features: jnp.ndarray,
                  dormant_pct: float = DORMANT_THRESHOLD) -> float:
  """Fraction of neurons with negligible activation.

  A neuron i is τ-dormant when its normalised score
      s_i = E|h_i| / (1/D · Σ_j E|h_j|) ≤ τ
  where D is the number of neurons.  This follows Definition 3.1 from
  the ReDo paper (Sokar et al., 2023).

  Default threshold τ=0.025 is calibrated for Swish activations.
  See module docstring for rationale.
  """
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


def extract_actor_features(networks, actor_params, obs):
  """Extract actor trunk features (body output before the policy head).

  Uses networks.actor_repr_fn which runs the actor body + LayerNorm +
  Swish but NOT the NormalTanhDistribution head.  The returned features
  have shape [batch, network_width].

  For ResidualMLP: body → LayerNorm → Swish → features.
  For plain MLP: body → ReLU-activated hidden → features.
  """
  if networks.actor_repr_fn is None:
    return None
  return networks.actor_repr_fn(actor_params, obs)


def _get_encoder_final_weights(q_params, encoder_name):
  """Extract the final Dense layer's weight from a critic encoder.

  Haiku keys look like 'sa_encoder/linear_5' (the highest-indexed linear
  under the encoder scope). We find the one with the highest index.
  """
  best_w = None
  best_idx = -1
  for key in q_params:
    key_str = str(key)
    if encoder_name in key_str and 'linear' in key_str:
      node = q_params[key]
      if isinstance(node, dict) and 'w' in node:
        # Extract index: 'sa_encoder/linear' -> 0, 'sa_encoder/linear_5' -> 5
        parts = key_str.split('linear')
        suffix = parts[-1].rstrip("']")
        idx = int(suffix.lstrip('_')) if suffix.lstrip('_').isdigit() else 0
        if idx > best_idx:
          best_idx = idx
          best_w = node['w']
  return best_w


def _get_actor_head_weights(actor_params):
  """Extract the actor head (NormalTanh mean layer) weight.

  Returns the 'Normal/linear' weight matrix (mean projection), which is
  the final layer of the actor's body → head pipeline.
  """
  for key in actor_params:
    key_str = str(key)
    if 'Normal' in key_str and 'linear' in key_str and 'linear_1' not in key_str:
      node = actor_params[key]
      if isinstance(node, dict) and 'w' in node:
        return node['w']
  return None


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

  # ---- FREQUENT (no forward pass for weight norms) ----
  metrics['actor/weight_norm'] = weight_norm_l2(actor_params)
  metrics['critic/weight_norm'] = weight_norm_l2(q_params)
  metrics['actor/final_layer_norm'] = final_layer_norm(actor_params)

  # Feature extraction — critic (forward pass through critic encoders)
  sa_feats, g_feats = extract_features(networks, q_params, obs_batch, action_batch)

  # Feature extraction — actor (forward pass through actor body only)
  actor_feats = extract_actor_features(networks, actor_params, obs_batch)

  # Critic feature entropy and Gini
  metrics['critic_sa/entropy'] = feature_entropy(sa_feats)
  metrics['critic_g/entropy'] = feature_entropy(g_feats)
  metrics['critic_sa/gini'] = gini_sparsity(sa_feats)
  metrics['critic_g/gini'] = gini_sparsity(g_feats)

  # Actor feature entropy and Gini
  if actor_feats is not None:
    metrics['actor/entropy'] = feature_entropy(actor_feats)
    metrics['actor/gini'] = gini_sparsity(actor_feats)

  if level == 'occasional':
    # ---- OCCASIONAL: critic ----
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

    metrics['critic_sa/dormant_ratio'] = dormant_ratio(sa_feats)
    metrics['critic_g/dormant_ratio'] = dormant_ratio(g_feats)

    # ---- OCCASIONAL: actor ----
    if actor_feats is not None:
      actor_feat_dim = actor_feats.shape[-1]
      metrics['actor/feature_rank'] = feature_rank(actor_feats, tau=0.99)

      # NRC1: target_dim = action_dim (the actor maps features → actions)
      metrics['actor/nrc1'] = compute_nrc1(actor_feats, target_dim=action_dim)

      # NRC2: alignment with the policy head (Normal/linear weight)
      actor_head_w = _get_actor_head_weights(actor_params)
      if actor_head_w is not None:
        metrics['actor/nrc2'] = compute_nrc2(actor_feats, actor_head_w)

      metrics['actor/dormant_ratio'] = dormant_ratio(actor_feats)

  return metrics
