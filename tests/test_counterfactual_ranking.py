"""Dependency-light checks for task-goal counterfactual ranking."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contrastive.counterfactual_ranking import (  # noqa: E402
    CounterfactualRankingBatch,
    CounterfactualRankingBuffer,
    scripted_contact_action,
    summarize_counterfactual_scores,
)
from experiment_configs_counterfactual_ranking import build_configs  # noqa: E402


def _batch():
  outcomes = np.asarray(
      [[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]],
      dtype=np.float32)
  return CounterfactualRankingBatch(
      observations=np.zeros((2, 4, 14), dtype=np.float32),
      actions=np.zeros((2, 4, 4), dtype=np.float32),
      outcomes=outcomes,
      progress=outcomes,
      success=np.zeros_like(outcomes),
      informative=np.ones((2,), dtype=bool),
      near_interaction=np.ones((2,), dtype=bool),
      interaction_distance=np.zeros((2,), dtype=np.float32))


def test_config_grid():
  configs = build_configs()
  assert len(configs) == 4
  assert Counter(config['seed'] for config in configs) == {5: 2, 6: 2}
  assert {config['single_task'] for config in configs} == {
      'sawyer_handle_press_side', 'sawyer_window_close'}
  for config in configs:
    assert config['steps_per_task'] == 1_000_000
    assert config['action_effect_actor_mode'] == 'effect_only'
    assert config['action_effect_target_mode'] == 'counterfactual_rank'
    assert config['counterfactual_rank_rollout_horizon'] == 100
    assert config['counterfactual_rank_action_repeat'] == 5
    assert config['counterfactual_rank_interval_steps'] == 50_000


def test_grouped_buffer_and_metrics():
  batch = _batch()
  buffer = CounterfactualRankingBuffer(capacity=3)
  buffer.add(batch)
  sampled = buffer.sample(5, np.random.default_rng(3))
  assert sampled.observations.shape == (5, 4, 14)
  metrics = summarize_counterfactual_scores(
      batch.outcomes, batch, min_outcome_gap=0.01)
  assert metrics[
      'counterfactual_rank/score_vs_task_progress_spearman'] > 0.999
  assert metrics['counterfactual_rank/pairwise_accuracy'] == 1.0
  assert metrics['counterfactual_rank/top_action_regret'] == 0.0


def test_scripted_contact_action_points_toward_mechanism():
  observation = np.zeros((14,), dtype=np.float32)
  observation[:3] = [0.0, 0.0, 0.0]
  observation[4:7] = [0.2, -0.1, 0.05]
  action = scripted_contact_action(
      observation, obs_dim=7,
      action_min=-np.ones((4,), dtype=np.float32),
      action_max=np.ones((4,), dtype=np.float32), gain=2.0)
  np.testing.assert_allclose(action[:3], [0.4, -0.2, 0.1])
  assert action[3] == 0.0


if __name__ == '__main__':
  test_config_grid()
  test_grouped_buffer_and_metrics()
  test_scripted_contact_action_points_toward_mechanism()
  print('counterfactual ranking checks passed')
