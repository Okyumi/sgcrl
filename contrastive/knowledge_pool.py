"""Knowledge vector pool and merging utilities for CKA-style continual RL in JAX.

Each knowledge vector v_k is a JAX pytree with the same structure as the
policy parameters (or a chosen subset).  The pool stores these vectors and
supports the CKA merging strategy:

  * If |V| > K_max after appending v_k, find the most-similar pair (cosine
    similarity over the flattened vector) and replace them with their average.
"""
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Pytree utilities
# ---------------------------------------------------------------------------

def _flatten_pytree(tree):
  """Flatten a pytree of arrays into a single 1-D array."""
  leaves = jax.tree_util.tree_leaves(tree)
  return jnp.concatenate([l.reshape(-1) for l in leaves])


def _pytree_add(a, b):
  """Element-wise addition of two pytrees with the same structure."""
  return jax.tree_map(lambda x, y: x + y, a, b)


def _pytree_scalar_mul(alpha, tree):
  """Multiply every leaf of *tree* by scalar *alpha*."""
  return jax.tree_map(lambda x: alpha * x, tree)


def _pytree_zeros_like(tree):
  """Return a pytree of zeros with the same structure / shapes."""
  return jax.tree_map(jnp.zeros_like, tree)


# ---------------------------------------------------------------------------
# Compose policy: θ' = θ_base + Σ α_j v_j + v_k
# ---------------------------------------------------------------------------

def compose_policy_params(theta_base, pool_vectors, alpha, v_k):
  """Build the effective policy parameters for the current task.

  Args:
    theta_base: pytree – frozen base policy parameters.
    pool_vectors: list of pytrees – knowledge vectors from previous tasks.
    alpha: 1-D jnp array of length len(pool_vectors), the softmax weights.
    v_k: pytree – the current task's knowledge vector (being optimised).

  Returns:
    theta_prime: pytree with the same structure as theta_base.
  """
  # Start from base
  theta_prime = theta_base

  # Add weighted sum of previous knowledge vectors
  if pool_vectors:
    for j, v_j in enumerate(pool_vectors):
      theta_prime = _pytree_add(theta_prime, _pytree_scalar_mul(alpha[j], v_j))

  # Add current task vector
  theta_prime = _pytree_add(theta_prime, v_k)
  return theta_prime


# ---------------------------------------------------------------------------
# Knowledge pool management
# ---------------------------------------------------------------------------

class KnowledgePool:
  """Maintains the pool V of knowledge vectors and handles CKA merging."""

  def __init__(self, k_max: int = 5):
    self.k_max = k_max
    self.vectors: List = []  # list of pytrees

  def __len__(self):
    return len(self.vectors)

  # ---- core API -----------------------------------------------------------

  def append(self, v_k):
    """Append a new knowledge vector (copy to avoid aliasing)."""
    v_copy = jax.tree_map(lambda x: jnp.array(x), v_k)
    self.vectors.append(v_copy)

  def merge_if_needed(self):
    """If |V| > k_max, merge the most-similar pair (cosine) and shrink."""
    while len(self.vectors) > self.k_max:
      self._merge_most_similar_pair()

  def get_vectors(self):
    """Return the current list of vectors (read-only view)."""
    return list(self.vectors)

  # ---- merging logic -------------------------------------------------------

  def _merge_most_similar_pair(self):
    n = len(self.vectors)
    assert n >= 2, 'Need at least 2 vectors to merge.'

    flat = [_flatten_pytree(v) for v in self.vectors]
    norms = [jnp.linalg.norm(f) for f in flat]

    best_sim = -2.0
    best_i, best_j = 0, 1
    for i in range(n):
      for j in range(i + 1, n):
        denom = norms[i] * norms[j]
        if denom < 1e-12:
          sim = 0.0
        else:
          sim = float(jnp.dot(flat[i], flat[j]) / denom)
        if sim > best_sim:
          best_sim = sim
          best_i, best_j = i, j

    # Merge: average
    v_merge = jax.tree_map(
        lambda a, b: (a + b) / 2.0,
        self.vectors[best_i],
        self.vectors[best_j],
    )
    # Remove the two (higher index first to avoid shifting)
    del self.vectors[best_j]
    del self.vectors[best_i]
    self.vectors.append(v_merge)

  # ---- serialisation -------------------------------------------------------

  def state_dict(self):
    """Return a serialisable snapshot (list of pytrees)."""
    return [jax.tree_map(lambda x: np.array(x), v) for v in self.vectors]

  def load_state_dict(self, vec_list):
    """Restore from a list of numpy pytrees."""
    self.vectors = [jax.tree_map(jnp.array, v) for v in vec_list]
