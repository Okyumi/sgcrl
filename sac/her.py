"""Sparse HER reward + terminal-discount rule for the continual SAC baseline.

The contrastive (CRL) pipeline never reads the reward — InfoNCE does not need
it — so ``run_continual_contrastive.py``'s ``flatten_fn`` passes the env reward
straight through.  SAC needs a TD signal that is consistent with the
*relabeled* goal, so the SAC driver recomputes it:

    r_t        = 1[ ||achieved_goal(s_{t+1}) - g_relabeled|| < tau ]
    discount_t = (1 - r_t) * gamma_env                (terminal bootstrap)

Two decisions are baked in (collaborator's notes, ``SAC_README.md`` section 2):

* **Where the relabeled goal comes from.** CRL's sampler is kept — geometric
  sampling over in-trajectory *future* states.  JaxGCRL's SAC instead uses the
  in-trajectory *terminal* state, falling back to the commanded goal when no
  terminal is found.  The CRL sampler is used so SAC and CRL runs see an
  identical goal distribution and the comparison isolates the critic.
* **Which state supplies the achieved goal.** ``s_{t+1}`` is used; JaxGCRL's
  ``flatten_batch`` uses ``s_t``.  The next state is what ``env.step`` actually
  reports on the true trajectory (reward delivered on arrival at the goal).

The rule itself lives here rather than inline in the ``tf.function`` so it can
be exercised without TensorFlow: ``ops`` selects the array backend.  The driver
calls it with :func:`tensorflow_ops`; the tests call it with :data:`NUMPY_OPS`.
"""
import dataclasses
from typing import Any, Callable, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class ArrayOps:
  """Minimal array backend: the two primitives the reward rule needs."""
  norm: Callable[[Any, int], Any]     # (x, axis) -> Euclidean norm
  to_float: Callable[[Any], Any]      # (x) -> float32 array


NUMPY_OPS = ArrayOps(
    norm=lambda x, axis: np.linalg.norm(x, axis=axis),
    to_float=lambda x: np.asarray(x, dtype=np.float32),
)


def tensorflow_ops() -> ArrayOps:
  """TensorFlow backend, for use inside the driver's ``flatten_fn``."""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel
  return ArrayOps(
      norm=lambda x, axis: tf.linalg.norm(x, axis=axis),
      to_float=lambda x: tf.cast(x, tf.float32),
  )


def her_reward_and_discount(
    achieved_next: Any,
    goal: Any,
    env_discount: Any,
    threshold: float,
    step_penalty_reward: bool,
    ops: ArrayOps = NUMPY_OPS,
) -> Tuple[Any, Any]:
  """Return ``(reward, discount)`` for a batch of relabeled transitions.

  Args:
    achieved_next: goal-slice projection of ``s_{t+1}``, shape ``[B, goal_dim]``.
    goal: relabeled goal, shape ``[B, goal_dim]``.
    env_discount: per-transition env discount ``gamma_env``, shape ``[B]``.
    threshold: goal-reach radius ``tau`` (``--her_reward_threshold``).
    step_penalty_reward: if True, ``r = 0`` on goal reach and ``-1`` otherwise
      (step penalty, the default); if False, ``r = +1`` / ``0``.
    ops: array backend (:data:`NUMPY_OPS` or :func:`tensorflow_ops`).

  The discount is zeroed on goal-reaching transitions regardless of reward
  shape, so the TD target collapses to the immediate reward — the standard
  sparse-goal trick that keeps Q bounded.
  """
  dist = ops.norm(achieved_next - goal, -1)
  reached = ops.to_float(dist < threshold)
  reward = reached - 1.0 if step_penalty_reward else reached
  discount = (1.0 - reached) * ops.to_float(env_discount)
  return reward, discount


def reached_from_reward(reward: Any, step_penalty_reward: bool) -> Any:
  """Recover the HER reached mask from either supported reward shape.

  Comparison operators keep this backend-independent for NumPy, JAX and
  TensorFlow arrays.
  """
  if step_penalty_reward:
    return reward > -0.5
  return reward > 0.5
