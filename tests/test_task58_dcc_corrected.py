#!/usr/bin/env python3
"""Dependency-light checks for the corrected-wrapper DCC baselines."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.task58_reevaluation import PairedTask58SuccessObserver
from experiment_configs_task58_dcc_corrected import build_configs


class _TimeStep:

  def __init__(self, observation, reward):
    self.observation = observation
    self.reward = reward


def _observation(hand, mechanism, goal, obs_dim=11):
  observation = np.zeros(obs_dim * 2, dtype=np.float32)
  observation[:3] = hand
  observation[4:7] = mechanism
  observation[obs_dim + 4:obs_dim + 7] = goal
  return observation


def test_corrected_observer_identifies_failure_stage_without_new_rollouts():
  observer = PairedTask58SuccessObserver(
      11, 'sawyer_handle_press_side', emitted_success_mode='corrected',
      mechanism_target=np.asarray([-0.07, 0.68, 0.07], dtype=np.float32))
  observer.observe_first(None, _TimeStep(
      _observation(
          hand=[0.0, 0.0, 0.3],
          mechanism=[0.0, 0.0, 0.2],
          goal=[9.0, -4.0, 0.07]),
      reward=0.0))
  observer.observe(None, _TimeStep(
      _observation(
          hand=[0.0, 0.0, 0.071],
          mechanism=[0.0, 0.0, 0.071],
          goal=[9.0, -4.0, 0.07]),
      reward=1.0), action=np.zeros(4))
  metrics = observer.get_metrics()
  assert metrics['legacy_success'] == 0.0
  assert metrics['task_axis_success'] == 1.0
  assert metrics['approach_success'] == 1.0
  assert metrics['mechanism_moved'] == 1.0
  assert np.isclose(metrics['initial_task_axis_distance'], 0.13)
  assert metrics['max_task_axis_progress'] > 0.12
  assert metrics['success_reward_mismatch_steps'] == 0.0


def test_corrected_observer_rejects_missing_wrapper_target():
  try:
    PairedTask58SuccessObserver(
        11, 'sawyer_window_close', emitted_success_mode='corrected')
  except ValueError as error:
    assert 'full-state conditioning goal is not a success target' in str(error)
  else:
    raise AssertionError('Corrected scorer accepted no wrapper target.')


def test_corrected_observer_uses_wrapper_target_not_conditioning_goal():
  """A reachable goal may sit inside the success region, not at its center."""
  observer = PairedTask58SuccessObserver(
      11, 'sawyer_window_close', emitted_success_mode='corrected',
      mechanism_target=np.asarray([0.0, 0.80, 0.20], dtype=np.float32))
  conditioning_goal = [0.10, 0.81, 0.20]
  observer.observe_first(None, _TimeStep(
      _observation(
          hand=[0.0, 0.0, 0.3], mechanism=[0.20, 0.80, 0.20],
          goal=conditioning_goal),
      reward=0.0))
  # This state passes the wrapper threshold around x=0.0, even though it is
  # not within 0.05 of the conditioning goal's x=0.10.
  observer.observe(None, _TimeStep(
      _observation(
          hand=[0.0, 0.0, 0.2], mechanism=[0.049, 0.80, 0.20],
          goal=conditioning_goal),
      reward=1.0), action=np.zeros(4))
  metrics = observer.get_metrics()
  assert metrics['task_axis_success'] == 1.0
  assert metrics['success_reward_mismatch_steps'] == 0.0


def test_twelve_matched_single_task_cells():
  configs = build_configs()
  assert len(configs) == 12
  assert {(config['single_task'], config['dyn_aux_weight'], config['seed'])
          for config in configs} == {
      (env_name, dyn_aux_weight, seed)
      for env_name in ('sawyer_handle_press_side', 'sawyer_window_close')
      for dyn_aux_weight in (0.0, 1.0)
      for seed in (5, 6, 7)
  }
  assert all(config['critic_mode'] == 'decomposed' for config in configs)
  assert all(config['actor_mode'] == 'reset' for config in configs)
  assert all(config['sawyer_success_mode'] == 'corrected'
             for config in configs)
  assert all(config['base_steps'] == 8_000_000 for config in configs)
  assert all(config['steps_per_task'] == 8_000_000 for config in configs)
  assert all(config['adapt_heads_only'] is False for config in configs)
  assert {config['dyn_aux_weight'] for config in configs} == {0.0, 1.0}
  assert all(config['network_width'] == 1024 for config in configs)
  assert all(config['counterfactual_rank_interval_steps'] == 0
             for config in configs)
  assert all(config['counterfactual_oracle_interval_steps'] == 0
             for config in configs)
  assert all(config['action_landscape_diagnostic_interval_steps'] == 0
             for config in configs)
  assert all(config['log_rl_metrics'] is True for config in configs)
  assert all(config['log_pool_cosine'] is True for config in configs)
  assert all(config['eval_record_video'] is True for config in configs)
  assert all(config['eval_video_every'] == 50_000 for config in configs)
  assert all(config['wandb_group'] ==
             'TASK58-DCC-CORRECTED-FULLNET-8M-DYN-ABLATION'
             for config in configs)


def test_runner_logs_lightweight_task58_stage_metrics():
  source = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  assert 'PairedTask58SuccessObserver' in source
  assert "f'evaluator/task58/{name}'" in source
  assert 'eval_record_video' in source
  assert 'evaluator/rollout_video' in source
  assert "'approach_success'" in source
  assert "'mechanism_moved'" in source
  assert "'max_task_axis_progress'" in source
  assert "'initial_task_axis_distance'" in source
  assert 'ordinary repeat-1 Task-5 and Task-8 jobs overwrite' in source


def test_launcher_runs_twelve_cells_without_probe_preflights():
  launcher = (REPO_ROOT / 'DRAFT_task58_dcc_corrected.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-11' in launcher
  assert 'CONFIG_LIMIT=12' in launcher
  assert 'TASKS_PER_GPU=1' in launcher
  assert 'experiment_configs_task58_dcc_corrected.py' in launcher
  assert 'tests/test_task58_reachable_success_goals.py' in launcher
  assert 'task58_dcc_fullnet_8m_v1' in launcher
  assert '48:00:00' in launcher
  assert 'conda activate contrastive_rl' in launcher
  assert 'set_up/torch_hpc_env.sh' in launcher
  assert launcher.index('conda activate contrastive_rl') < launcher.index(
      'tests/test_task58_dcc_corrected.py')
  assert 'COUNTERFACTUAL' not in launcher
  assert 'ACTION_LANDSCAPE_SELF_TEST' not in launcher


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'Corrected Task-5/Task-8 DCC tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
