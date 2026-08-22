"""Dependency-light checks for the causal DCC action-landscape probe."""
from pathlib import Path

import numpy as np

from contrastive import action_ranking_diagnostics as diagnostics
import experiment_configs


ROOT = Path(__file__).resolve().parents[1]


class _FakeData:

  def __init__(self):
    self.mocap_pos = np.array([[1.0, 2.0, 3.0]])
    self.mocap_quat = np.array([[1.0, 0.0, 0.0, 0.0]])
    self.ctrl = np.array([0.25, -0.25])
    self.qfrc_applied = np.zeros(2)
    self.xfrc_applied = np.zeros((1, 6))


class _FakeSim:

  def __init__(self):
    self.value = 0.0
    self.data = _FakeData()
    self.forward_count = 0

  def get_state(self):
    return {'value': self.value}

  def set_state(self, state):
    self.value = state['value']

  def forward(self):
    self.forward_count += 1


class _TimeStep:

  def __init__(self, observation, reward=0.0, step_type=1):
    self.observation = np.asarray(observation, dtype=np.float32)
    self.reward = reward
    self.discount = 1.0
    self.step_type = step_type

  def last(self):
    return False


class _ActionSpec:
  shape = (2,)
  minimum = np.array([-1.0, -1.0])
  maximum = np.array([1.0, 1.0])


class _FakeBaseEnv:

  def __init__(self):
    self.sim = _FakeSim()
    self.np_random = np.random.RandomState(7)
    self._last_obs = np.zeros(4)

  def reset(self):
    self.sim.value = 0.0
    self._last_obs = np.zeros(4)
    return _TimeStep(self._last_obs)

  def step(self, action):
    noise = self.np_random.normal(0.0, 0.01)
    self.sim.value += float(np.sum(action)) + noise
    self.sim.data.ctrl[:] = action
    self._last_obs = np.array(
        [self.sim.value, self.sim.data.ctrl[0], 0.0, 1.0])
    return _TimeStep(self._last_obs, reward=float(self.sim.value > 0.5))


class _FakeWrapper:

  def __init__(self, environment):
    self._environment = environment
    self._elapsed_steps = 0

  def action_spec(self):
    return _ActionSpec()

  def reset(self):
    self._elapsed_steps = 0
    return self._environment.reset()

  def step(self, action):
    self._elapsed_steps += 1
    return self._environment.step(action)


def test_restore_step_reproduces_sim_wrapper_cache_controller_and_rng():
  base = _FakeBaseEnv()
  environment = _FakeWrapper(base)
  diagnostics.assert_restore_step_reproducible(
      environment, np.array([0.4, -0.1]), atol=0.0)
  assert environment._elapsed_steps == 1
  np.testing.assert_allclose(base.sim.data.ctrl, [0.4, -0.1])


def test_summary_detects_bad_replay_landscape_and_actor_score_gap():
  family = np.array(
      ['policy', 'policy', 'replay', 'replay', 'local', 'uniform'])
  # Policy is scored highest but has the worst observed mechanism outcome.
  scores = np.array([10.0, 9.0, 1.0, 2.0, 3.0, 0.0])
  progress = np.array([-2.0, -1.5, 2.0, 1.0, 0.5, 0.0])
  outcomes = {
      'one_step_full_progress': progress,
      'one_step_mechanism_progress': progress,
      'rollout_full_progress': progress,
      'rollout_mechanism_progress': progress,
      'success': np.array([0, 0, 1, 0, 0, 0]),
  }
  actions = np.arange(12, dtype=float).reshape(6, 2)
  replay_actions = actions[2:4]
  metrics = diagnostics.summarize_action_ranking(
      scores, outcomes, family, actions, replay_actions)
  assert metrics['action_landscape/policy_score_percentile'] == 1.0
  assert metrics['action_landscape/policy_outcome_percentile'] < 0.5
  assert metrics[
      'action_landscape/policy_score_outcome_percentile_gap'] > 0.5
  assert metrics[
      'action_landscape/replay_score_vs_rollout_mechanism_spearman'] < 0


def test_config_indices_12_to_17_are_matched_plain_dcc_probes():
  configs = experiment_configs.build_configs()
  probes = configs[12:18]
  assert len(configs) == 36
  assert len(probes) == 6
  assert {config['seed'] for config in probes} == {5, 6, 7}
  assert {config['single_task'] for config in probes} == {
      'sawyer_handle_press_side', 'sawyer_window_close'}
  assert {config['critic_mode'] for config in probes} == {'decomposed'}
  assert {config['in_trajectory_negative_repeats'] for config in probes} == {1}
  assert {config['shortcut_diagnostic_interval'] for config in probes} == {50}
  assert {
      config['action_landscape_diagnostic_interval_steps']
      for config in probes} == {500_000}


def test_torch_wrapper_runs_restore_test_and_selects_six_probe_cells():
  wrapper = (ROOT / 'DRAFT_action_landscape.sh').read_text()
  assert '#SBATCH --array=0-1' in wrapper
  assert 'CONFIG_INDEX_OFFSET=12' in wrapper
  assert 'CONFIG_LIMIT=6' in wrapper
  assert '--self-test-env=sawyer_handle_press_side' in wrapper
  assert '--self-test-env=sawyer_window_close' in wrapper
  assert 'action_landscape_checkpoints' in wrapper
  assert 'exec bash "$REPO_DIR/DRAFT.sh"' in wrapper


def test_runner_logs_diagnostic_events_immediately_and_wires_causal_probe():
  runner = (ROOT / 'run_continual_contrastive.py').read_text()
  learner = (ROOT / 'contrastive' /
             'continual_learning_decomposed.py').read_text()
  shortcut = (ROOT / 'contrastive' / 'shortcut_diagnostics.py').read_text()
  for token in (
      'diagnostic_metrics = getattr(learner, \'last_diagnostic_metrics\'',
      'run_causal_action_ranking_probe(',
      "critic_params['phi_task']",
      "f'learner/{name}'",
  ):
    assert token in runner
  assert 'def last_diagnostic_metrics(self):' in learner
  for token in (
      'hand_goal_shuffled_categorical_accuracy',
      'gripper_goal_shuffled_categorical_accuracy',
      'mechanism_goal_shuffled_categorical_accuracy',
  ):
    assert token in shortcut


def test_canonical_torch_launcher_forwards_every_probe_setting():
  launcher = (ROOT / 'DRAFT.sh').read_text()
  for flag in (
      'action_landscape_diagnostic_interval_steps',
      'action_landscape_num_anchors',
      'action_landscape_candidates_per_family',
      'action_landscape_rollout_horizon',
      'action_landscape_anchor_prefix_steps',
      'action_landscape_local_noise_std',
  ):
    assert f'--{flag}=$' in launcher
