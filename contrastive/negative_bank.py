"""Previous-replay negative bank for continual contrastive RL.

Motivation:
  Continual RL has a natural offline-to-online structure: while learning
  task k (online), we already have replay buffers from tasks 0..k-1
  (offline).  Rather than discarding that data, we can use its reached
  goals as extra NEGATIVE samples in the contrastive objective, expanding
  the pool of negatives beyond the current batch.

Problem with vanilla cross-task negatives:
  MetaWorld Sawyer tasks occupy different workspace regions (hammer
  near the nail, push_wall near the wall, etc.), so a goal from task A
  is TRIVIALLY distinguishable from goals in task B's batch.  The
  critic can solve the contrastive task by memorising task identity,
  not by learning goal-relevant features.  Categorical accuracy goes
  high but representation quality degrades.

Principled approach (hard_weighted):
  1. Hard-negative mining: for each (s_i, a_i), score a candidate pool
     of bank goals and keep the top-M by score.  Hard negatives are
     those the critic currently struggles to push down — they provide
     the strongest gradient signal.
  2. Difficulty cap: down-weight bank negatives by `w_bank < 1` so they
     contribute less than in-batch negatives.  Prevents the critic from
     leaning too heavily on cross-task contrasts.
  3. FIFO refresh: bank stores per-task goal arrays, older tasks get
     trimmed when the bank is full.

Usage:
  bank = NegativeBank(goal_dim=22, per_task_capacity=10_000)
  # After each task finishes:
  bank.add_task(task_id=0, goals=relabeled_goals_array)
  # During training:
  neg_goals = bank.sample(n=256, rng=key)  # shape [n, goal_dim]
"""
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


class NegativeBank:
  """FIFO buffer of HER-relabeled goals from previous tasks.

  Stored as a single concatenated numpy array to allow fast JAX sampling
  via jnp.asarray + jax.random.choice.
  """

  def __init__(self, goal_dim: int, per_task_capacity: int = 10_000,
               max_tasks: int = 20):
    self._goal_dim = goal_dim
    self._per_task_capacity = per_task_capacity
    self._max_tasks = max_tasks
    # List of per-task arrays, oldest first.
    self._task_arrays = []  # each entry: np.ndarray [N_k, goal_dim]
    self._task_ids = []     # parallel list of task IDs (for debugging)

  def add_task(self, task_id: int, goals: np.ndarray) -> None:
    """Add goals from a completed task to the bank.

    Args:
      task_id: integer task identifier.
      goals: numpy array of shape [N, goal_dim].  If N exceeds
        per_task_capacity, a random subsample is stored.
    """
    goals = np.asarray(goals)
    assert goals.ndim == 2 and goals.shape[1] == self._goal_dim, (
        f'goals shape {goals.shape} incompatible with goal_dim={self._goal_dim}')
    if goals.shape[0] > self._per_task_capacity:
      idx = np.random.choice(goals.shape[0],
                             size=self._per_task_capacity, replace=False)
      goals = goals[idx]
    self._task_arrays.append(goals.astype(np.float32))
    self._task_ids.append(task_id)
    # FIFO trim
    while len(self._task_arrays) > self._max_tasks:
      self._task_arrays.pop(0)
      self._task_ids.pop(0)

  def size(self) -> int:
    """Total number of goals in the bank."""
    return sum(a.shape[0] for a in self._task_arrays)

  def num_tasks(self) -> int:
    return len(self._task_arrays)

  def as_array(self) -> Optional[np.ndarray]:
    """Return all bank goals concatenated, or None if empty."""
    if not self._task_arrays:
      return None
    return np.concatenate(self._task_arrays, axis=0)

  def sample(self, n: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    """Sample `n` goals uniformly at random.

    Returns None when the bank is empty (caller should skip bank negatives).
    """
    flat = self.as_array()
    if flat is None or flat.shape[0] == 0:
      return None
    # sample with replacement if bank is smaller than n
    replace = flat.shape[0] < n
    idx = rng.choice(flat.shape[0], size=n, replace=replace)
    return flat[idx]

  def state_dict(self):
    """Serialise for checkpointing."""
    return {
        'task_arrays': [np.asarray(a) for a in self._task_arrays],
        'task_ids':    list(self._task_ids),
        'goal_dim':    self._goal_dim,
        'per_task_capacity': self._per_task_capacity,
        'max_tasks':   self._max_tasks,
    }

  def load_state_dict(self, sd):
    self._task_arrays = [np.asarray(a) for a in sd.get('task_arrays', [])]
    self._task_ids    = list(sd.get('task_ids', []))
    self._goal_dim    = sd.get('goal_dim', self._goal_dim)
    self._per_task_capacity = sd.get('per_task_capacity', self._per_task_capacity)
    self._max_tasks   = sd.get('max_tasks', self._max_tasks)


# =======================================================================
# Hard-negative mining utilities
# =======================================================================

def pick_hard_negatives(sa_repr: jnp.ndarray, candidate_g_repr: jnp.ndarray,
                        n_keep: int) -> jnp.ndarray:
  """Per-anchor top-k hard negative selection.

  Args:
    sa_repr: [B, D] state-action representations
    candidate_g_repr: [C, D] candidate goal representations
    n_keep: number of hard negatives to keep per anchor
  Returns:
    hard_g_repr: [B, n_keep, D] per-anchor hard negative representations
  """
  # scores[i, c] = sa_repr[i] · candidate_g_repr[c]
  scores = sa_repr @ candidate_g_repr.T  # [B, C]
  # top-k per row
  _, topk_idx = jax.lax.top_k(scores, n_keep)  # [B, n_keep]
  # Gather: [B, n_keep, D]
  hard = candidate_g_repr[topk_idx]
  return hard
