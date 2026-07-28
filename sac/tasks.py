"""Task sequence, fixed goals, and task-boundary transfer rules.

Kept free of ``reverb`` / ``tensorflow`` / ``acme`` imports so the sequencing
and transfer logic can be unit-tested (and ``--help`` can run) on a machine
without the full training stack.

``FIXED_GOALS`` duplicates the table in ``run_continual_contrastive.py``
rather than importing it, because importing that module pulls in the whole
reverb + TensorFlow data pipeline.  ``tests/test_sac_tasks.py`` AST-parses the
CRL driver and asserts the two tables are identical, so the copy cannot drift.
"""
from typing import Callable, Optional, Tuple

import numpy as np

from contrastive.continual_config import (
    CONTINUAL_TASK_SEQUENCE,
    CONTINUAL_TASK_SEQUENCE_20,
)

# Single fixed goal per MetaWorld task ("A Single Goal Is All You Need").
# Must stay identical to ``run_continual_contrastive.FIXED_GOALS``.
FIXED_GOALS = {
    'sawyer_hammer': np.array([0.24, 0.74, 0.11]),
    'sawyer_push_wall': np.array([0.05, 0.85, 0.015]),
    'sawyer_faucet_close': np.array([-0.14, 0.82, 0.13]),
    'sawyer_push_back': np.array([0.06, 0.62, 0.02]),
    'sawyer_stick_pull': np.array([0.41, 0.54, 0.02]),
    'sawyer_handle_press_side': np.array([-0.07, 0.68, 0.07]),
    'sawyer_push': np.array([0.02, 0.89, 0.02]),
    'sawyer_shelf_place': np.array([0.02, 0.89, 0.30]),
    'sawyer_window_close': np.array([0., 0.80, 0.2]),
    'sawyer_peg_unplug_side': np.array([0.01, 0.66, 0.13]),
}

ACTOR_MODES = ('cka', 'reset', 'persistent')
CRITIC_MODES = ('persistent', 'reset', 'cka')


def resolve_task_sequence(
    num_tasks: int,
    single_task: str = '',
    use_20_tasks: bool = False,
    task_sequence: str = '',
) -> Tuple[Tuple[str, ...], int]:
  """Return ``(task_sequence, num_tasks)`` from the mutually exclusive flags.

  Precedence, highest first:

  1. ``single_task`` — one environment, ``num_tasks == 1``.
  2. ``task_sequence`` — comma-separated env names, in order (SAC-only flag;
     the CRL driver has no equivalent).  ``num_tasks`` still truncates it.
  3. ``use_20_tasks`` — two passes of the 10-task CKA-RL sequence.
  4. default — the 10-task CKA-RL sequence.

  ``num_tasks`` is clamped to the sequence length, matching the CRL driver.
  """
  if single_task:
    return (single_task,), 1
  if task_sequence:
    names = tuple(n.strip() for n in task_sequence.split(',') if n.strip())
    if not names:
      raise ValueError('--task_sequence was set but parsed to zero task names.')
  elif use_20_tasks:
    names = CONTINUAL_TASK_SEQUENCE_20
  else:
    names = CONTINUAL_TASK_SEQUENCE
  return names, min(num_tasks, len(names))


def validate_task_sequence(names: Tuple[str, ...]) -> None:
  """Raise ``ValueError`` if any task lacks a fixed goal.

  Called before training starts so a typo fails immediately instead of after
  the first task has run for hours.
  """
  unknown = [n for n in names if n not in FIXED_GOALS]
  if unknown:
    raise ValueError(
        f'No fixed goal defined for {unknown}. Known tasks: '
        f'{sorted(FIXED_GOALS)}. Add an entry to sac.tasks.FIXED_GOALS (and '
        f'run_continual_contrastive.FIXED_GOALS) to use a new environment.')


def pre_task_actor_state(actor_mode: str, task_id: int, theta_base,
                         pool, fresh_pool: Callable[[], object]):
  """Actor state handed *into* task ``task_id``.

  * ``reset`` — drop the base and the pool (each task starts from scratch).
  * ``persistent`` — keep the base (which is the previous task's composed
    policy) but drop the pool, so there is no CKA mixture.
  * ``cka`` — keep both; the pool supplies ``sum_j alpha_j v_j``.

  Task 0 has nothing to inherit, so all three modes fall through to the
  ``cka`` branch (base ``None``, empty pool).
  """
  if task_id > 0 and actor_mode == 'reset':
    return None, fresh_pool()
  if task_id > 0 and actor_mode == 'persistent':
    return theta_base, fresh_pool()
  return theta_base, pool


def post_task_actor_state(actor_mode: str, theta_base, pool, composed_policy,
                          fresh_pool: Callable[[], object]):
  """Actor state carried *out of* a finished task (and checkpointed).

  ``cka`` keeps whatever the learner produced.  ``persistent`` collapses the
  composed policy into the new base.  ``reset`` discards everything, so the
  checkpoint records ``theta_base=None`` and an empty pool.
  """
  if actor_mode == 'reset':
    return None, fresh_pool()
  if actor_mode == 'persistent':
    return composed_policy, fresh_pool()
  return theta_base, pool


def steps_for_task(task_id: int, base_steps: int, steps_per_task: int) -> int:
  """Env steps budgeted for ``task_id`` (task 0 uses ``base_steps``)."""
  return base_steps if task_id == 0 else steps_per_task


def fixed_goal(env_name: str) -> np.ndarray:
  """Fixed goal for ``env_name``; raises a helpful error if unknown."""
  validate_task_sequence((env_name,))
  return FIXED_GOALS[env_name]


def task_id_args(use_task_id: bool, task_id: int,
                 num_tasks: int) -> Tuple[Optional[int], Optional[int]]:
  """``(task_id, num_tasks)`` for ``make_environment``, or ``(None, None)``.

  ``contrastive.utils.make_environment`` appends a one-hot task ID to both the
  state and the goal only when both arguments are non-``None``.
  """
  if not use_task_id:
    return None, None
  return task_id, num_tasks
