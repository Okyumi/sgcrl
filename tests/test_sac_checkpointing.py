"""Tests for checkpoint path keying, serialization and resume (``sac.checkpointing``).

Needs jax (for the pytree conversion) but not reverb / TensorFlow / acme.
"""
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from sac import checkpointing as ckpt

SEED = 7


def _payload():
  """A checkpoint shaped like the one the driver writes."""
  return {
      'theta_base': {'actor_body/~/linear_0': {'w': jnp.ones((3, 2)),
                                               'b': jnp.zeros((2,))}},
      'pool_vectors': [{'q1/~/linear_0': {'w': jnp.full((2, 2), 0.5)}}],
      'q_params': {'q1/~/linear_0': {'w': jnp.arange(4.).reshape(2, 2)}},
      'target_q_params': {'q1/~/linear_0': {'w': jnp.arange(4.).reshape(2, 2)}},
      'q_optimizer_state': (jnp.array(3.0),),
      'composed_policy': {'actor_body/~/linear_0': {'w': jnp.ones((3, 2))}},
      'task_id': 2,
      'env_name': 'sawyer_push',
  }


# ---- path keying ---------------------------------------------------------

def test_reward_tag_names_both_shapes():
  assert ckpt.reward_tag(True) == 'steppen'
  assert ckpt.reward_tag(False) == 'sparse01'


def test_config_key_contains_every_ablation_axis():
  key = ckpt.config_key(critic_mode='reset', use_task_id=False,
                        adapt_heads_only=True, actor_mode='persistent',
                        step_penalty_reward=False,
                        her_reward_threshold=0.12)
  assert key == ('actor_persistent_critic_reset_tid_False_heads_True'
                 '_rew_sparse01_tau_0p12')


def test_ckpt_path_layout_is_config_then_seed_then_task():
  path = ckpt.ckpt_path('/root', task_id=4, seed=3)
  assert path == ('/root/actor_cka_critic_persistent_tid_True_heads_True'
                  '_rew_steppen_tau_0p05/seed_3/task_4.pkl')


@pytest.mark.parametrize('kwargs', [
    {'actor_mode': 'reset'},
    {'critic_mode': 'reset'},
    {'use_task_id': False},
    {'adapt_heads_only': False},
    {'step_penalty_reward': False},
    {'her_reward_threshold': 0.12},
])
def test_every_ablation_axis_changes_the_path(kwargs):
  """No two cells may share a checkpoint file."""
  assert ckpt.ckpt_path('/root', 0, SEED) != ckpt.ckpt_path(
      '/root', 0, SEED, **kwargs)


def test_seed_separates_checkpoints():
  assert ckpt.ckpt_path('/root', 0, 1) != ckpt.ckpt_path('/root', 0, 2)


# ---- save / load round trip ---------------------------------------------

def test_round_trip_preserves_nested_pytree_values(tmp_path):
  data = _payload()
  ckpt.save_ckpt(str(tmp_path), 2, SEED, data)
  loaded = ckpt.load_ckpt(str(tmp_path), 2, SEED)

  assert set(loaded) == set(data)
  np.testing.assert_allclose(
      loaded['theta_base']['actor_body/~/linear_0']['w'],
      data['theta_base']['actor_body/~/linear_0']['w'])
  np.testing.assert_allclose(
      loaded['pool_vectors'][0]['q1/~/linear_0']['w'],
      data['pool_vectors'][0]['q1/~/linear_0']['w'])
  np.testing.assert_allclose(loaded['q_optimizer_state'][0],
                             data['q_optimizer_state'][0])


def test_round_trip_keeps_non_array_metadata_as_is(tmp_path):
  ckpt.save_ckpt(str(tmp_path), 2, SEED, _payload())
  loaded = ckpt.load_ckpt(str(tmp_path), 2, SEED)
  assert loaded['task_id'] == 2
  assert loaded['env_name'] == 'sawyer_push'


def test_load_returns_jax_arrays(tmp_path):
  ckpt.save_ckpt(str(tmp_path), 2, SEED, _payload())
  loaded = ckpt.load_ckpt(str(tmp_path), 2, SEED)
  assert isinstance(loaded['q_params']['q1/~/linear_0']['w'], jnp.ndarray)


def test_the_pickle_on_disk_holds_numpy_so_analysis_scripts_need_no_jax(
    tmp_path):
  path = ckpt.save_ckpt(str(tmp_path), 2, SEED, _payload())
  with open(path, 'rb') as handle:
    raw = pickle.load(handle)
  leaf = raw['q_params']['q1/~/linear_0']['w']
  assert isinstance(leaf, np.ndarray)
  assert not isinstance(leaf, jnp.ndarray)


def test_save_creates_the_directory_tree_and_returns_the_path(tmp_path):
  root = tmp_path / 'not' / 'yet' / 'there'
  path = ckpt.save_ckpt(str(root), 0, SEED, {'task_id': 0})
  assert path == ckpt.ckpt_path(str(root), 0, SEED)
  assert (tmp_path / 'not' / 'yet' / 'there').is_dir()


def test_load_of_a_missing_checkpoint_names_the_config_in_the_error(tmp_path):
  with pytest.raises(FileNotFoundError, match='seed=7'):
    ckpt.load_ckpt(str(tmp_path), 0, SEED)


def test_a_checkpoint_cannot_be_loaded_under_the_other_reward_shape(tmp_path):
  ckpt.save_ckpt(str(tmp_path), 0, SEED, {'task_id': 0},
                 step_penalty_reward=True)
  with pytest.raises(FileNotFoundError):
    ckpt.load_ckpt(str(tmp_path), 0, SEED, step_penalty_reward=False)


def test_a_checkpoint_cannot_be_loaded_under_another_her_threshold(tmp_path):
  ckpt.save_ckpt(str(tmp_path), 0, SEED, {'task_id': 0},
                 her_reward_threshold=0.05)
  with pytest.raises(FileNotFoundError):
    ckpt.load_ckpt(
        str(tmp_path), 0, SEED, her_reward_threshold=0.12)


def test_legacy_checkpoint_is_reported_as_ambiguous(tmp_path):
  legacy_dir = (
      tmp_path
      / 'actor_cka_critic_persistent_tid_True_heads_True_rew_steppen'
      / f'seed_{SEED}')
  legacy_dir.mkdir(parents=True)
  with open(legacy_dir / 'task_0.pkl', 'wb') as handle:
    pickle.dump({'task_id': 0}, handle)
  with pytest.raises(FileNotFoundError, match='legacy checkpoint'):
    ckpt.load_ckpt(str(tmp_path), 0, SEED)


# ---- resume -------------------------------------------------------------

def test_no_checkpoints_means_no_resume(tmp_path):
  assert ckpt.find_resume_task(str(tmp_path), 10, SEED) is None


def test_resume_returns_the_task_after_the_newest_checkpoint(tmp_path):
  for task_id in range(3):
    ckpt.save_ckpt(str(tmp_path), task_id, SEED, {'task_id': task_id})
  assert ckpt.find_resume_task(str(tmp_path), 10, SEED) == 3


def test_a_finished_sequence_resumes_at_num_tasks(tmp_path):
  """Re-running a completed cell must be a no-op, not a restart."""
  for task_id in range(4):
    ckpt.save_ckpt(str(tmp_path), task_id, SEED, {'task_id': task_id})
  assert ckpt.find_resume_task(str(tmp_path), 4, SEED) == 4


def test_resume_probes_backwards_and_ignores_gaps(tmp_path):
  ckpt.save_ckpt(str(tmp_path), 0, SEED, {'task_id': 0})
  ckpt.save_ckpt(str(tmp_path), 5, SEED, {'task_id': 5})
  assert ckpt.find_resume_task(str(tmp_path), 10, SEED) == 6


def test_resume_ignores_checkpoints_beyond_num_tasks(tmp_path):
  ckpt.save_ckpt(str(tmp_path), 8, SEED, {'task_id': 8})
  assert ckpt.find_resume_task(str(tmp_path), 3, SEED) is None


def test_resume_ignores_other_seeds(tmp_path):
  ckpt.save_ckpt(str(tmp_path), 2, SEED + 1, {'task_id': 2})
  assert ckpt.find_resume_task(str(tmp_path), 10, SEED) is None


@pytest.mark.parametrize('kwargs', [
    {'actor_mode': 'reset'},
    {'critic_mode': 'reset'},
    {'step_penalty_reward': False},
    {'her_reward_threshold': 0.12},
])
def test_resume_ignores_other_ablation_cells(tmp_path, kwargs):
  ckpt.save_ckpt(str(tmp_path), 2, SEED, {'task_id': 2}, **kwargs)
  assert ckpt.find_resume_task(str(tmp_path), 10, SEED) is None


def test_resume_then_load_recovers_the_previous_task(tmp_path):
  """The driver's actual flow: find the task, then load task-1."""
  ckpt.save_ckpt(str(tmp_path), 0, SEED, _payload(), actor_mode='reset',
                 critic_mode='reset')
  resume_at = ckpt.find_resume_task(str(tmp_path), 10, SEED,
                                    actor_mode='reset', critic_mode='reset')
  assert resume_at == 1
  loaded = ckpt.load_ckpt(str(tmp_path), resume_at - 1, SEED,
                          actor_mode='reset', critic_mode='reset')
  assert loaded['env_name'] == 'sawyer_push'
