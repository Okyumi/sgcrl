"""Tests for task sequencing, fixed goals, and task-boundary transfer rules."""
import ast
import pathlib

import numpy as np
import pytest

from contrastive.continual_config import (
    CONTINUAL_TASK_SEQUENCE,
    CONTINUAL_TASK_SEQUENCE_20,
)
from sac import tasks

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---- fixed goals ---------------------------------------------------------

def _parse_fixed_goals(path):
  """Read a ``FIXED_GOALS`` dict literal without importing the module.

  ``run_continual_contrastive.py`` imports reverb and TensorFlow at module
  level, so it cannot be imported here; the table is recovered from the AST.
  """
  tree = ast.parse(path.read_text())
  for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
      continue
    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
    if 'FIXED_GOALS' not in targets:
      continue
    goals = {}
    for key, value in zip(node.value.keys, node.value.values):
      # value is np.array([...]); take the literal list argument.
      goals[ast.literal_eval(key)] = ast.literal_eval(value.args[0])
    return goals
  raise AssertionError(f'No FIXED_GOALS assignment found in {path}')


def test_fixed_goals_match_the_contrastive_driver():
  """Guards the duplicated table in sac.tasks against drift."""
  crl_goals = _parse_fixed_goals(REPO_ROOT / 'run_continual_contrastive.py')
  assert set(crl_goals) == set(tasks.FIXED_GOALS)
  for name, goal in crl_goals.items():
    np.testing.assert_allclose(tasks.FIXED_GOALS[name], goal)


def test_every_task_in_both_sequences_has_a_fixed_goal():
  tasks.validate_task_sequence(CONTINUAL_TASK_SEQUENCE)
  tasks.validate_task_sequence(CONTINUAL_TASK_SEQUENCE_20)


def test_fixed_goals_are_3d():
  for name, goal in tasks.FIXED_GOALS.items():
    assert np.asarray(goal).shape == (3,), name


def test_validate_task_sequence_rejects_unknown_task():
  with pytest.raises(ValueError, match='sawyer_not_a_task'):
    tasks.validate_task_sequence(('sawyer_push', 'sawyer_not_a_task'))


def test_fixed_goal_lookup_raises_for_unknown_task():
  with pytest.raises(ValueError, match='typo'):
    tasks.fixed_goal('typo')


# ---- sequence resolution -------------------------------------------------

def test_default_sequence_is_the_10_task_cka_rl_sequence():
  sequence, num_tasks = tasks.resolve_task_sequence(num_tasks=10)
  assert sequence == CONTINUAL_TASK_SEQUENCE
  assert num_tasks == 10


def test_num_tasks_truncates_without_reordering():
  sequence, num_tasks = tasks.resolve_task_sequence(num_tasks=3)
  assert num_tasks == 3
  assert sequence[:3] == CONTINUAL_TASK_SEQUENCE[:3]


def test_num_tasks_is_clamped_to_the_sequence_length():
  _, num_tasks = tasks.resolve_task_sequence(num_tasks=99)
  assert num_tasks == len(CONTINUAL_TASK_SEQUENCE)


def test_use_20_tasks_gives_two_passes_of_the_10_task_sequence():
  sequence, num_tasks = tasks.resolve_task_sequence(
      num_tasks=20, use_20_tasks=True)
  assert num_tasks == 20
  assert sequence == CONTINUAL_TASK_SEQUENCE + CONTINUAL_TASK_SEQUENCE
  assert sequence[10:] == sequence[:10]


def test_single_task_overrides_everything_and_forces_one_task():
  sequence, num_tasks = tasks.resolve_task_sequence(
      num_tasks=10, single_task='sawyer_push', use_20_tasks=True,
      task_sequence='sawyer_hammer,sawyer_push_wall')
  assert sequence == ('sawyer_push',)
  assert num_tasks == 1


def test_explicit_task_sequence_is_used_in_order():
  sequence, num_tasks = tasks.resolve_task_sequence(
      num_tasks=10, task_sequence='sawyer_push, sawyer_hammer')
  assert sequence == ('sawyer_push', 'sawyer_hammer')
  assert num_tasks == 2  # clamped to the sequence length


def test_explicit_task_sequence_beats_use_20_tasks():
  sequence, _ = tasks.resolve_task_sequence(
      num_tasks=10, use_20_tasks=True, task_sequence='sawyer_push')
  assert sequence == ('sawyer_push',)


def test_blank_task_sequence_is_rejected():
  with pytest.raises(ValueError, match='zero task names'):
    tasks.resolve_task_sequence(num_tasks=2, task_sequence=' , ,')


# ---- step budget ---------------------------------------------------------

def test_task_zero_uses_base_steps_and_later_tasks_use_steps_per_task():
  assert tasks.steps_for_task(0, base_steps=5, steps_per_task=7) == 5
  assert tasks.steps_for_task(1, base_steps=5, steps_per_task=7) == 7
  assert tasks.steps_for_task(9, base_steps=5, steps_per_task=7) == 7


# ---- task-id plumbing ---------------------------------------------------

def test_task_id_args_are_none_unless_use_task_id():
  assert tasks.task_id_args(False, 3, 10) == (None, None)
  assert tasks.task_id_args(True, 3, 10) == (3, 10)


# ---- transfer / reset at task boundaries --------------------------------

class _Pool:
  """Stand-in for KnowledgePool: identity is all the transfer rules touch."""

  _next_id = 0

  def __init__(self):
    _Pool._next_id += 1
    self.uid = _Pool._next_id


def _fresh():
  return _Pool()


@pytest.mark.parametrize('actor_mode', tasks.ACTOR_MODES)
def test_task_zero_always_inherits_nothing_regardless_of_mode(actor_mode):
  pool = _Pool()
  base, out_pool = tasks.pre_task_actor_state(
      actor_mode, 0, theta_base=None, pool=pool, fresh_pool=_fresh)
  assert base is None
  assert out_pool is pool


def test_pre_task_reset_drops_base_and_pool():
  pool, base = _Pool(), {'w': 1}
  out_base, out_pool = tasks.pre_task_actor_state(
      'reset', 1, base, pool, _fresh)
  assert out_base is None
  assert out_pool is not pool


def test_pre_task_persistent_keeps_base_but_drops_pool():
  pool, base = _Pool(), {'w': 1}
  out_base, out_pool = tasks.pre_task_actor_state(
      'persistent', 1, base, pool, _fresh)
  assert out_base is base
  assert out_pool is not pool


def test_pre_task_cka_keeps_base_and_pool():
  pool, base = _Pool(), {'w': 1}
  out_base, out_pool = tasks.pre_task_actor_state('cka', 1, base, pool, _fresh)
  assert out_base is base
  assert out_pool is pool


def test_post_task_reset_discards_everything():
  out_base, out_pool = tasks.post_task_actor_state(
      'reset', {'w': 1}, _Pool(), {'w': 2}, _fresh)
  assert out_base is None
  assert isinstance(out_pool, _Pool)


def test_post_task_persistent_promotes_the_composed_policy_to_base():
  composed = {'w': 2}
  pool = _Pool()
  out_base, out_pool = tasks.post_task_actor_state(
      'persistent', {'w': 1}, pool, composed, _fresh)
  assert out_base is composed
  assert out_pool is not pool


def test_post_task_cka_keeps_the_learner_output_untouched():
  base, pool = {'w': 1}, _Pool()
  out_base, out_pool = tasks.post_task_actor_state(
      'cka', base, pool, {'w': 2}, _fresh)
  assert out_base is base
  assert out_pool is pool


def test_reset_mode_round_trip_never_leaks_state_across_tasks():
  """Full R/R loop: whatever a task produces must not reach the next task."""
  base, pool = None, _Pool()
  for task_id in range(3):
    in_base, in_pool = tasks.pre_task_actor_state(
        'reset', task_id, base, pool, _fresh)
    if task_id > 0:
      assert in_base is None
      assert in_pool is not pool
    # Pretend the task trained and produced fresh parameters.
    base, pool = tasks.post_task_actor_state(
        'reset', {'trained': task_id}, _Pool(), {'composed': task_id}, _fresh)
    assert base is None


def test_persistent_mode_round_trip_chains_composed_policies():
  base, pool = None, _Pool()
  for task_id in range(3):
    in_base, _ = tasks.pre_task_actor_state(
        'persistent', task_id, base, pool, _fresh)
    assert in_base is base
    composed = {'composed': task_id}
    base, pool = tasks.post_task_actor_state(
        'persistent', {'trained': task_id}, _Pool(), composed, _fresh)
    assert base is composed
