"""State-index mask for the cross-task-stable subset of the observation.

The decomposed-critic algorithm (proposal 1) anchors a shared encoder
``b_shared`` against a forward-dynamics auxiliary, but only on the
subset of state indices whose semantics are stable across the 10-task
Sawyer Meta-World sequence. Object-slot indices have task-conditional
semantics (e.g., index 4 is the hammer in one task and the button in
another), so regressing them with a single shared head would smear the
target across tasks; we mask them out.

See ``docs/2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md`` for the
per-task index audit. Across every task in ``env_utils.py`` the first
four state indices have constant meaning:

  index 0  end-effector x
  index 1  end-effector y
  index 2  end-effector z
  index 3  gripper distance apart

Indices 4-10 carry object positions, quaternions, or zero-padding,
none of which are stable across tasks.

Constants and helpers in this module are kept tiny and dependency-free
so they can be imported from anywhere in the pipeline.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
import numpy as np

# Indices of state dimensions whose semantic meaning is constant across
# all tasks in the 10-task Meta-World sequence. The four indices are
# end-effector xyz plus gripper distance apart.
STABLE_INDICES: Tuple[int, ...] = (0, 1, 2, 3)

# Default state dim mirrors ``env_utils.STATE_DIM_UNIFIED``. Hard-coded
# here so this module is importable without bringing in the gym /
# Meta-World stack just to read the constant.
DEFAULT_STATE_DIM: int = 11


def stable_state_mask(state_dim: int = DEFAULT_STATE_DIM) -> np.ndarray:
  """Return a boolean mask of length ``state_dim`` selecting stable indices.

  Defaults to ``env_utils.STATE_DIM_UNIFIED`` (= 11) so the mask aligns
  with the unified-padding observation layout enforced by every wrapper
  in ``env_utils``.
  """
  mask = np.zeros(state_dim, dtype=bool)
  for i in STABLE_INDICES:
    if i < state_dim:
      mask[i] = True
  return mask


def stable_state_mask_jnp(
    state_dim: int = DEFAULT_STATE_DIM,
) -> jnp.ndarray:
  """Same mask as ``stable_state_mask`` but returned as a ``jnp.ndarray``."""
  return jnp.asarray(stable_state_mask(state_dim))


def num_stable_indices() -> int:
  """Number of stable indices, ``d_M`` in the algorithm."""
  return len(STABLE_INDICES)


def select_stable(s_or_obs: jnp.ndarray) -> jnp.ndarray:
  """Slice the stable indices out of a state or full-observation tensor.

  Accepts either a ``(B, state_dim)`` state tensor or a
  ``(B, FULL_OBS_DIM)`` full observation tensor; in both cases returns
  a ``(B, d_M)`` tensor by gathering ``STABLE_INDICES`` along the last
  axis. The function does not multiply by the mask, it slices, which is
  cheaper and keeps gradient and shape semantics simple.
  """
  return s_or_obs[..., jnp.asarray(STABLE_INDICES)]
