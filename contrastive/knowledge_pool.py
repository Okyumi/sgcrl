"""Knowledge pool and CKA state for continual contrastive RL (JAX/Haiku).

The pool stores per-task knowledge vectors at fixed shape so the whole
structure can live inside a JAX ``training_state`` pytree. Each pool slot
has the same pytree structure as the policy (or critic) parameters; an
inactive slot is zero-filled and masked out via ``mask``.

This module exports:

  * ``CKAPool`` (Flax struct): pool of knowledge vectors stored as a
    pytree with leading axis of size ``capacity`` (= ``k_max + 1``)
    plus a boolean mask of the same length marking active slots.
  * ``CKAState`` (Flax struct): bundle of ``base_params``, ``v_k``,
    ``pool``, ``alpha_logits`` (==``beta_k`` in the previous SGCRL
    code), and ``alpha_scale``. The entire bundle lives inside the
    learner's training state so that JIT-compiled inner functions can
    differentiate through ``alpha_logits`` and ``alpha_scale`` and
    reuse their trace cache across tasks.
  * ``compute_contribution(pool, alpha_logits, alpha_scale)``:
    differentiable masked-softmax blend of the pool. Inactive slots are
    masked to ``-inf`` so they contribute exactly zero, regardless of
    ``alpha_scale``.
  * ``compose(cka_state, trainable)``: build the effective parameters
    ``theta_base + sum_j alpha_j v_j + v_k`` from a CKA state and a
    trainable bundle ``{'v_k', 'alpha_logits', 'alpha_scale'}``.
  * ``empty_pool_like``, ``init_cka_state``, ``reinit_for_new_task``,
    ``append_vector_host``: lifecycle helpers (host-side, run only at
    task boundaries).
  * Backwards-compatible: the old ``KnowledgePool`` class,
    ``compose_policy_params``, ``_pytree_zeros_like`` and
    ``_flatten_pytree`` shims are kept so older code paths that import
    them still work during the transition.
"""
from __future__ import annotations

from typing import List, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np


# ===========================================================================
# Public new API
# ===========================================================================

@flax.struct.dataclass
class CKAPool:
  """Fixed-capacity knowledge pool that lives inside a JAX pytree.

  Attributes:
    vectors: pytree of arrays with leading dim ``capacity``.
    mask: bool array of shape ``(capacity,)`` marking active slots.
  """
  vectors: 'jax.tree_util.PyTreeDef'  # type: ignore[type-arg]
  mask: jnp.ndarray


def empty_pool_like(template, capacity: int) -> CKAPool:
  """Create an empty CKAPool whose slots match ``template``'s structure."""
  vectors = jax.tree_util.tree_map(
      lambda x: jnp.zeros((capacity,) + x.shape, dtype=x.dtype),
      template,
  )
  mask = jnp.zeros((capacity,), dtype=jnp.bool_)
  return CKAPool(vectors=vectors, mask=mask)


def pool_size(pool: CKAPool) -> int:
  """Number of active slots, host-side."""
  return int(jnp.sum(pool.mask))


def compute_contribution(
    pool: CKAPool,
    alpha_logits: jnp.ndarray,
    alpha_scale: jnp.ndarray,
):
  """Compute Sigma alpha_j v_j with masked softmax over active slots.

  The softmax is taken over ``alpha_logits * alpha_scale`` with
  inactive slots set to ``-inf`` so their weight is 0 regardless of
  ``alpha_scale``. The output is a pytree with the same structure as a
  single knowledge vector (the leading capacity axis is contracted).
  """
  masked_logits = jnp.where(
      pool.mask, alpha_logits * alpha_scale, -jnp.inf)
  # Numerically valid softmax when no slot is active: fall back to a
  # uniform-zero distribution. The product with the (zero) vectors is
  # zero anyway, so this is purely a NaN guard.
  any_active = jnp.any(pool.mask)
  safe_logits = jnp.where(any_active, masked_logits,
                          jnp.zeros_like(masked_logits))
  alphas = jax.nn.softmax(safe_logits, axis=0)
  alphas = jnp.where(any_active, alphas, jnp.zeros_like(alphas))

  def _blend(v_stack):
    broadcast_shape = (alphas.shape[0],) + (1,) * (v_stack.ndim - 1)
    return jnp.sum(alphas.reshape(broadcast_shape) * v_stack, axis=0)

  return jax.tree_util.tree_map(_blend, pool.vectors)


@flax.struct.dataclass
class CKAState:
  """All state needed to drive one CKA-decomposed component.

  Attributes:
    base_params: frozen theta_base (matches policy/critic param shape).
    v_k: trainable per-task delta (same shape as base_params).
    pool: CKAPool of past-task knowledge vectors.
    alpha_logits: trainable logits, shape ``(capacity,)``.
    alpha_scale: trainable scalar, shape ``()``.

  ``v_k``, ``alpha_logits``, ``alpha_scale`` are jointly updated by the
  learner's actor optimiser (a single Adam over the trainable bundle
  ``{'v_k', 'alpha_logits', 'alpha_scale'}``).  ``base_params`` and
  ``pool`` are constants from the optimiser's perspective and are
  rewritten only at task boundaries.
  """
  base_params: 'jax.tree_util.PyTreeDef'  # type: ignore[type-arg]
  v_k: 'jax.tree_util.PyTreeDef'          # type: ignore[type-arg]
  pool: CKAPool
  alpha_logits: jnp.ndarray
  alpha_scale: jnp.ndarray


def init_cka_state(base_params, capacity: int,
                   alpha_scale_init: float = 1.0) -> CKAState:
  """Build a CKAState whose pool is empty and trainables are zeroed.

  ``base_params`` should already reflect any post-task-zero freezing
  (i.e., it is the theta_base for the upcoming task).  ``v_k`` is
  zero-initialised so the composed policy at the first inner-loop step
  equals ``base_params``.
  """
  v_k = jax.tree_util.tree_map(jnp.zeros_like, base_params)
  pool = empty_pool_like(base_params, capacity)
  alpha_logits = jnp.zeros((capacity,), dtype=jnp.float32)
  alpha_scale = jnp.array(alpha_scale_init, dtype=jnp.float32)
  return CKAState(
      base_params=base_params,
      v_k=v_k,
      pool=pool,
      alpha_logits=alpha_logits,
      alpha_scale=alpha_scale,
  )


def reinit_for_new_task(
    cka: CKAState,
    new_base_params,
    rng_key: jax.Array,
    alpha_logits_init_std: float = 0.01,
    alpha_scale_init: float = 1.0,
) -> CKAState:
  """Refresh v_k, alpha_logits and alpha_scale at a task boundary.

  Pool and base_params are taken from the caller (typically the pool
  inherited from the prior task and a possibly-updated base).
  """
  v_k = jax.tree_util.tree_map(jnp.zeros_like, new_base_params)
  capacity = cka.pool.mask.shape[0]
  alpha_logits = (
      jax.random.normal(rng_key, (capacity,)) * alpha_logits_init_std
  ).astype(jnp.float32)
  # Mask inactive slots' logits to zero (purely cosmetic; the softmax
  # ignores them via the mask) for cleaner logging.
  alpha_logits = jnp.where(cka.pool.mask, alpha_logits, 0.0)
  alpha_scale = jnp.array(alpha_scale_init, dtype=jnp.float32)
  return cka.replace(
      base_params=new_base_params,
      v_k=v_k,
      alpha_logits=alpha_logits,
      alpha_scale=alpha_scale,
  )


def compose(cka: CKAState):
  """theta' = theta_base + Sigma alpha_j v_j + v_k."""
  contribution = compute_contribution(
      cka.pool, cka.alpha_logits, cka.alpha_scale)
  return jax.tree_util.tree_map(lambda b, p, v: b + p + v,
                      cka.base_params, contribution, cka.v_k)


def compose_from_trainable(cka: CKAState, trainable: dict):
  """Compose using a trainable bundle separated from the CKAState.

  Used inside the inner JIT loop where the optimiser's params are the
  bundle ``{'v_k', 'alpha_logits', 'alpha_scale'}``; the CKAState then
  carries only ``base_params`` and ``pool`` as constants.
  """
  contribution = compute_contribution(
      cka.pool, trainable['alpha_logits'], trainable['alpha_scale'])
  return jax.tree_util.tree_map(lambda b, p, v: b + p + v,
                      cka.base_params, contribution, trainable['v_k'])


# ===========================================================================
# Host-side pool mutation (only at task boundaries)
# ===========================================================================

def _flatten_for_sim(v) -> jnp.ndarray:
  return jnp.concatenate(
      [x.reshape(-1) for x in jax.tree_util.tree_leaves(v)])


def append_vector_host(pool: CKAPool, new_vector, k_max: int) -> CKAPool:
  """Append ``new_vector`` and merge if ``mask.sum() > k_max`` (host-side)."""
  capacity = pool.mask.shape[0]
  n_active = int(jnp.sum(pool.mask))
  if n_active >= capacity:
    pool = _merge_most_similar_pair_host(pool)
    n_active = int(jnp.sum(pool.mask))
  insert_idx = n_active
  new_vectors = jax.tree_util.tree_map(
      lambda stack, leaf: stack.at[insert_idx].set(leaf),
      pool.vectors, new_vector,
  )
  new_mask = pool.mask.at[insert_idx].set(True)
  pool = CKAPool(vectors=new_vectors, mask=new_mask)
  if int(jnp.sum(pool.mask)) > k_max:
    pool = _merge_most_similar_pair_host(pool)
  return pool


def _merge_most_similar_pair_host(pool: CKAPool) -> CKAPool:
  """Merge the two most cosine-similar active slots into one (host)."""
  capacity = pool.mask.shape[0]
  active_indices = [i for i in range(capacity) if bool(pool.mask[i])]
  if len(active_indices) < 2:
    return pool
  actives = []
  for idx in active_indices:
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda stack: stack[idx], pool.vectors))
    actives.append(jnp.concatenate([l.reshape(-1) for l in leaves]))
  flat = jnp.stack(actives, axis=0)
  norms = jnp.linalg.norm(flat, axis=1) + 1e-8
  sims = (flat @ flat.T) / (norms[:, None] * norms[None, :])
  n = flat.shape[0]
  # Mask out the diagonal and lower triangle.
  inf_mask = jnp.tril(jnp.ones((n, n), dtype=jnp.bool_), k=0)
  sims = jnp.where(inf_mask, -jnp.inf, sims)
  flat_argmax = int(jnp.argmax(sims))
  i, j = divmod(flat_argmax, n)
  src_a = active_indices[i]
  src_b = active_indices[j]
  avg_vectors = jax.tree_util.tree_map(
      lambda stack: stack.at[src_a].set((stack[src_a] + stack[src_b]) / 2.0),
      pool.vectors,
  )
  avg_vectors = jax.tree_util.tree_map(
      lambda stack: stack.at[src_b].set(jnp.zeros_like(stack[src_b])),
      avg_vectors,
  )
  new_mask = pool.mask.at[src_b].set(False)
  return _compact_pool(CKAPool(vectors=avg_vectors, mask=new_mask))


def _compact_pool(pool: CKAPool) -> CKAPool:
  """Move all active slots to the leading positions, in-order."""
  capacity = pool.mask.shape[0]
  active_indices = [i for i in range(capacity) if bool(pool.mask[i])]
  inactive_indices = [i for i in range(capacity)
                      if not bool(pool.mask[i])]
  perm = jnp.array(active_indices + inactive_indices, dtype=jnp.int32)
  permuted_vectors = jax.tree_util.tree_map(lambda stack: stack[perm], pool.vectors)
  new_mask = jnp.zeros_like(pool.mask)
  new_mask = new_mask.at[:len(active_indices)].set(True)
  permuted_vectors = jax.tree_util.tree_map(
      lambda stack: jnp.where(
          new_mask.reshape((-1,) + (1,) * (stack.ndim - 1)),
          stack,
          jnp.zeros_like(stack),
      ),
      permuted_vectors,
  )
  return CKAPool(vectors=permuted_vectors, mask=new_mask)


# ===========================================================================
# Pytree utilities (kept as backwards-compatible exports)
# ===========================================================================

def _pytree_zeros_like(tree):
  return jax.tree_util.tree_map(jnp.zeros_like, tree)


def _flatten_pytree(tree):
  leaves = jax.tree_util.tree_leaves(tree)
  return jnp.concatenate([l.reshape(-1) for l in leaves])


def _pytree_add(a, b):
  return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _pytree_scalar_mul(alpha, tree):
  return jax.tree_util.tree_map(lambda x: alpha * x, tree)


def compose_policy_params(theta_base, pool_vectors, alpha, v_k):
  """Legacy compose: theta' = theta_base + sum_j alpha_j v_j + v_k.

  Retained for backwards compatibility with older callers (e.g.
  ``run_continual_contrastive`` checkpoints from before the refactor).
  Prefer ``compose`` / ``compose_from_trainable`` in new code.
  """
  theta_prime = theta_base
  if pool_vectors:
    for j, v_j in enumerate(pool_vectors):
      theta_prime = _pytree_add(theta_prime,
                                _pytree_scalar_mul(alpha[j], v_j))
  theta_prime = _pytree_add(theta_prime, v_k)
  return theta_prime


# ===========================================================================
# Legacy KnowledgePool (transitional shim)
# ---------------------------------------------------------------------------
# The new code uses ``CKAState`` (with a fixed-shape ``CKAPool`` inside the
# JAX training_state pytree).  ``KnowledgePool`` is preserved here so that:
#   1. external scripts that import it continue to work; and
#   2. ``run_continual_contrastive``'s checkpoint shape stays loadable
#      while we migrate.
# Uses ``jax.tree_util.tree_map`` so the file works on both legacy JAX
# (0.4.x, where ``jax.tree.map`` does not exist) and modern JAX (0.6+,
# where ``jax.tree_util.tree_map`` is still available).
# ===========================================================================

class KnowledgePool:
  """Legacy variable-length pool. Use ``CKAPool``+``CKAState`` for new code."""

  def __init__(self, k_max: int = 5):
    self.k_max = k_max
    self.vectors: List = []

  def __len__(self):
    return len(self.vectors)

  def append(self, v_k):
    v_copy = jax.tree_util.tree_map(lambda x: jnp.array(x), v_k)
    self.vectors.append(v_copy)

  def merge_if_needed(self):
    while len(self.vectors) > self.k_max:
      self._merge_most_similar_pair()

  def get_vectors(self):
    return list(self.vectors)

  def _merge_most_similar_pair(self):
    n = len(self.vectors)
    assert n >= 2, 'Need at least 2 vectors to merge.'
    flat = [_flatten_pytree(v) for v in self.vectors]
    flat_stack = jnp.stack(flat, axis=0)
    norms = jnp.linalg.norm(flat_stack, axis=1) + 1e-8
    sims = (flat_stack @ flat_stack.T) / (norms[:, None] * norms[None, :])
    inf_mask = jnp.tril(jnp.ones((n, n), dtype=jnp.bool_), k=0)
    sims = jnp.where(inf_mask, -jnp.inf, sims)
    flat_argmax = int(jnp.argmax(sims))
    best_i, best_j = divmod(flat_argmax, n)
    v_merge = jax.tree_util.tree_map(
        lambda a, b: (a + b) / 2.0,
        self.vectors[best_i], self.vectors[best_j],
    )
    del self.vectors[best_j]
    del self.vectors[best_i]
    self.vectors.append(v_merge)

  def state_dict(self):
    return [jax.tree_util.tree_map(lambda x: np.array(x), v) for v in self.vectors]

  def load_state_dict(self, vec_list):
    self.vectors = [jax.tree_util.tree_map(jnp.array, v) for v in vec_list]
