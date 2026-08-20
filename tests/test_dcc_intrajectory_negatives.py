"""Dependency-light checks for DCC in-trajectory negative sampling."""
from pathlib import Path

import pytest

from contrastive import intrajectory
import experiment_configs


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / 'run_continual_contrastive.py').read_text()


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
  probes = experiment_configs.build_configs()[6:]
  assert len(probes) == 6
  assert {config['single_task'] for config in probes} == {
      'sawyer_handle_press_side', 'sawyer_window_close'}
  assert {config['seed'] for config in probes} == {5, 6, 7}
  assert {config['in_trajectory_negative_repeats'] for config in probes} == {12}
  assert {config['dyn_aux_weight'] for config in probes} == {1.0}
  assert {config['actor_mode'] for config in probes} == {'reset'}


def test_sampler_and_checkpoint_identity_are_present():
  for token in (
      'def flatten_intrajectory_fn(sample):',
      'anchor_index[:, None] < candidate_index[None, :]',
      'if in_trajectory_repeats > 1 else flatten_fn',
      '# Legacy path is intentionally unchanged.',
      "config_key += f'_itn{int(in_trajectory_negative_repeats)}'",
      "config_key += f'_env_{single_task}'",
  ):
    assert token in RUNNER


def test_all_launchers_forward_sampler():
  for name in ('draft_3.sh', 'draft_4.sh', 'DRAFT.sh',
               'submit_continual_torch.sh'):
    launcher = (ROOT / name).read_text()
    assert 'IN_TRAJECTORY_NEGATIVE_REPEATS="${IN_TRAJECTORY_NEGATIVE_REPEATS:-1}"' in launcher
    assert '--in_trajectory_negative_repeats=$IN_TRAJECTORY_NEGATIVE_REPEATS' in launcher


def test_torch_wrapper_selects_only_new_intrajectory_cells():
  canonical = (ROOT / 'DRAFT.sh').read_text()
  wrapper = (ROOT / 'DRAFT_intrajectory.sh').read_text()
  assert 'CONFIG_INDEX_OFFSET="${CONFIG_INDEX_OFFSET:-0}"' in canonical
  assert 'CONFIG_LIMIT="${CONFIG_LIMIT:-0}"' in canonical
  assert 'CONFIG_INDEX_OFFSET=6' in wrapper
  assert 'CONFIG_LIMIT=6' in wrapper
  assert '#SBATCH --array=0-1' in wrapper
  assert 'exec bash /scratch/yd2247/sgcrl/DRAFT.sh' in wrapper
