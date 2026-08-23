"""Dependency-light checks for staged counterfactual experiments."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive import counterfactual_outcomes  # noqa: E402
from contrastive.counterfactual_ranking import (  # noqa: E402
    CounterfactualRankingBatch,
    summarize_counterfactual_scores,
    summarize_oracle,
)
from contrastive.phase_gated_control import PhaseGatedChunkActor  # noqa: E402
from experiment_configs_counterfactual_stages import build_configs  # noqa: E402


class _Timestep:
  def __init__(self, reward):
    self.reward = reward


class _ActionSpec:
  shape = (4,)
  minimum = -np.ones(shape, dtype=np.float32)
  maximum = np.ones(shape, dtype=np.float32)


class _Actor:
  def __init__(self):
    self.select_calls = 0

  def observe_first(self, timestep):
    del timestep

  def select_action(self, observation):
    del observation
    self.select_calls += 1
    return np.zeros((4,), dtype=np.float32)

  def observe(self, action, next_timestep):
    del action, next_timestep

  def update(self, *args, **kwargs):
    del args, kwargs


def _observation(contact=True):
  observation = np.zeros((14,), dtype=np.float32)
  observation[:3] = [0.0, 0.0, 0.0]
  observation[4:7] = [0.05 if contact else 0.5, 0.0, 0.0]
  observation[7:14] = observation[:7]
  return observation


def _batch():
  outcomes = np.asarray(
      [[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]],
      dtype=np.float32)
  return CounterfactualRankingBatch(
      observations=np.zeros((2, 4, 14), dtype=np.float32),
      actions=np.zeros((2, 4, 4), dtype=np.float32),
      outcomes=outcomes,
      progress=outcomes,
      success=(outcomes > 0.15).astype(np.float32),
      informative=np.ones((2,), dtype=bool),
      near_interaction=np.ones((2,), dtype=bool),
      interaction_distance=np.zeros((2,), dtype=np.float32))


def test_sparse_reward_success_semantics():
  observation = _observation(contact=True)
  success, available = counterfactual_outcomes.benchmark_success(
      _Timestep(0.0), observation, 7, 0.05, 'zero_reward')
  assert success == 1.0 and available == 1.0
  failure, available = counterfactual_outcomes.benchmark_success(
      _Timestep(-1.0), observation, 7, 0.05, 'zero_reward')
  assert failure == 0.0 and available == 1.0


def test_rank_metrics_are_named_and_permutation_sensitive():
  batch = _batch()
  metrics = summarize_counterfactual_scores(
      batch.outcomes, batch, min_outcome_gap=0.01)
  assert metrics['counterfactual_rank/score_vs_outcome_spearman'] > 0.999
  assert metrics[
      'counterfactual_rank/action_permutation_spearman_drop'] > 0.5
  assert 'counterfactual_rank/action_shuffle_retention' not in metrics
  oracle = summarize_oracle(batch)
  assert oracle['oracle/best_success_fraction'] == 1.0
  assert oracle['oracle/success_gain'] > 0.0


def test_phase_gated_actor_repeats_selected_chunk():
  base = _Actor()
  actor = PhaseGatedChunkActor(
      base, obs_dim=7, action_spec=_ActionSpec(),
      score_actions_fn=lambda obs, actions: actions[:, 0],
      rng=np.random.default_rng(7), reach_mode='policy',
      interaction_threshold=0.09, chunk_length=5, num_candidates=8)
  actor.observe_first(_Timestep(None))
  actions = [actor.select_action(_observation(contact=True)) for _ in range(5)]
  for action in actions[1:]:
    np.testing.assert_allclose(action, actions[0])
  assert base.select_calls == 5
  metrics = actor.get_and_reset_metrics()
  assert metrics['phase_control/contact_step_fraction'] == 1.0
  assert metrics['phase_control/contact_episode_reach_rate'] == 1.0
  assert metrics['phase_control/first_contact_step_mean'] == 1.0
  assert metrics['phase_control/chunk_selections'] == 1.0


def test_staged_config_grids():
  for stage in ('A', 'B', 'C'):
    configs = build_configs(stage=stage)
    assert len(configs) == 4
    assert Counter(config['seed'] for config in configs) == {5: 2, 6: 2}
    assert all(config['counterfactual_stage'] == stage for config in configs)
  stage_b = build_configs(stage='B')
  assert all(not config['action_effect_enabled'] for config in stage_b)
  assert all(config['counterfactual_oracle_interval_steps'] > 0
             for config in stage_b)
  for reach_mode in ('policy', 'scripted_contact'):
    stage_d = build_configs(stage='D', reach_mode=reach_mode)
    assert len(stage_d) == 4
    assert all(config['phase_gated_control'] for config in stage_d)
    assert all(config['phase_gate_reach_mode'] == reach_mode
               for config in stage_d)
    assert all(config['phase_gate_chunk_length']
               == config['counterfactual_rank_action_repeat'] == 5
               for config in stage_d)


def test_stage_flags_reach_torch_runner():
  root = Path(__file__).resolve().parents[1]
  draft = (root / 'DRAFT.sh').read_text()
  runner = (root / 'run_continual_contrastive.py').read_text()
  stage_specific = {
      key for stage in ('A', 'B', 'C', 'D')
      for config in build_configs(stage=stage)
      for key in config
      if key.startswith(('counterfactual_', 'phase_gate_',
                         'action_landscape_'))
  }
  stage_specific.discard('counterfactual_stage')
  for key in stage_specific:
    assert key.upper() in draft, f'{key} missing from DRAFT.sh'
    assert f"'{key}'" in runner, f'{key} missing from runner flags/config'


if __name__ == '__main__':
  test_sparse_reward_success_semantics()
  test_rank_metrics_are_named_and_permutation_sensitive()
  test_phase_gated_actor_repeats_selected_chunk()
  test_staged_config_grids()
  test_stage_flags_reach_torch_runner()
  print('counterfactual staged checks passed')
