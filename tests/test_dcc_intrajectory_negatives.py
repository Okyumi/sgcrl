"""Dependency-light checks for DCC in-trajectory negative sampling."""
from pathlib import Path

import pytest

from contrastive import intrajectory
import experiment_configs


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / 'run_continual_contrastive.py').read_text()
DRAFT_3 = (ROOT / 'draft_3.sh').read_text()
DRAFT_4 = (ROOT / 'draft_4.sh').read_text()


def test_stablecrl_default_grouping_preserves_legacy_batch_size():
  assert intrajectory.trajectories_per_batch(256, 12) == 22
  counts = intrajectory.in_batch_repetition_counts(256, 12)
  assert counts == (12,) * 21 + (4,)
  assert sum(counts) == 256


@pytest.mark.parametrize('repeats', (0, -1, 151))
def test_invalid_repetition_factors_are_rejected(repeats):
  with pytest.raises(ValueError):
    intrajectory.validate_repetition_factor(
        repeats, batch_size=256, episode_transitions=150)


def test_task5_and_task8_cells_use_reported_seeds_and_r12():
  configs = experiment_configs.build_configs()
  probes = configs[6:]
  assert len(probes) == 6
  assert {config['single_task'] for config in probes} == {
      'sawyer_handle_press_side', 'sawyer_window_close'}
  assert {config['seed'] for config in probes} == {5, 6, 7}
  assert {config['in_trajectory_negative_repeats'] for config in probes} == {
      12}
  assert {config['dyn_aux_weight'] for config in probes} == {1.0}
  assert {config['actor_mode'] for config in probes} == {'reset'}
  assert all(config['shortcut_diagnostic_interval'] == 1000
             for config in probes)
  assert all(config['log_probe_data'] is True for config in probes)


def test_sampler_uses_same_episode_rows_and_strictly_future_goals():
  for token in (
      'def flatten_intrajectory_fn(sample):',
      'anchor_index[:, None] < candidate_index[None, :]',
      'tf.gather(all_state, anchor_index)',
      'tf.gather(all_goal, goal_index)',
      'tf.gather(sample.data.action, anchor_index)',
  ):
    assert token in RUNNER
  assert 'if in_trajectory_repeats > 1 else flatten_fn' in RUNNER
  assert '# Legacy path is intentionally unchanged.' in RUNNER


def test_sampler_keeps_episode_major_groups_before_truncation():
  assert 'Dataset.batch`` produces [episode, repetition, ...]' in RUNNER
  assert 'packed = tf.reshape(\n              t,' in RUNNER
  assert 'return packed[:config.batch_size]' in RUNNER


def test_launchers_forward_sampler_and_wandb_group():
  for launcher in (DRAFT_3, DRAFT_4):
    assert 'IN_TRAJECTORY_NEGATIVE_REPEATS="${IN_TRAJECTORY_NEGATIVE_REPEATS:-1}"' in launcher
    assert '--in_trajectory_negative_repeats=$IN_TRAJECTORY_NEGATIVE_REPEATS' in launcher
    assert 'WANDB_GROUP="${WANDB_GROUP:-}"' in launcher
    assert '--wandb_group=$WANDB_GROUP' in launcher
  assert '#SBATCH --array=0-5' in DRAFT_4


def test_new_sampler_has_disjoint_checkpoint_identity():
  for token in (
      "config_key += f'_itn{int(in_trajectory_negative_repeats)}'",
      "config_key += f'_env_{single_task}'",
      'and int(in_trajectory_negative_repeats) == 1',
  ):
    assert token in RUNNER
