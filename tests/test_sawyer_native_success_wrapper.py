#!/usr/bin/env python3
"""Dependency-light checks for versioned native Sawyer success semantics."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive import sawyer_success


class _DummyEnvironment:

  def __init__(self):
    self.observation = ('custom', 'goal', 'observation')
    self.last_evaluated_action = None

  def _get_obs(self):
    return self.observation

  def evaluate_state(self, observation, action):
    assert observation == self.observation
    self.last_evaluated_action = action
    return 7.5, {'success': 1.0, 'fallback_metric': 3.0}


def test_legacy_is_default_and_returns_control_to_historical_predicate():
  environment = _DummyEnvironment()
  assert sawyer_success.native_sparse_transition(
      environment, ('native_obs', 4.2, False, {'success': 1.0})) is None


def test_native_info_is_authoritative_and_preserves_diagnostics():
  environment = _DummyEnvironment()
  sawyer_success.set_success_mode(environment, 'native_info')
  observation, reward, done, info = sawyer_success.native_sparse_transition(
      environment, ('native_obs', 4.2, True,
                    {'success': 1.0, 'near_object': 0.8}))
  assert observation == environment.observation
  assert reward == 1.0
  assert done is False
  assert info['success'] == 1.0
  assert info['native_reward'] == 4.2
  assert info['native_terminated'] is True
  assert info['native_truncated'] is False
  assert info['native_step_api'] == 'gym_4tuple'
  assert info['near_object'] == 0.8
  assert info['wrapper_success_mode'] == 'native_info'


def test_gymnasium_five_tuple_is_normalized_for_acme():
  environment = _DummyEnvironment()
  sawyer_success.set_success_mode(environment, 'native_info')
  observation, reward, done, info = sawyer_success.native_sparse_transition(
      environment,
      ('native_obs', 2.5, False, True,
       {'success': 0.0, 'unscaled_reward': 2.5}))
  assert observation == environment.observation
  assert reward == 0.0
  assert done is False
  assert info['native_terminated'] is False
  assert info['native_truncated'] is True
  assert info['native_step_api'] == 'gymnasium_5tuple'
  assert info['unscaled_reward'] == 2.5


def test_none_parent_result_uses_evaluate_state_without_second_step():
  environment = _DummyEnvironment()
  sawyer_success.set_success_mode(environment, 'native_info')
  action = ('the', 'executed', 'action')
  observation, reward, done, info = sawyer_success.native_sparse_transition(
      environment, None, action=action)
  assert observation == environment.observation
  assert reward == 1.0
  assert done is False
  assert environment.last_evaluated_action == action
  assert info['native_reward'] == 7.5
  assert info['fallback_metric'] == 3.0
  assert info['native_step_api'] == 'evaluate_state_fallback:NoneType:-1'


def test_reward_info_two_tuple_is_supported():
  environment = _DummyEnvironment()
  sawyer_success.set_success_mode(environment, 'native_info')
  _, reward, _, info = sawyer_success.native_sparse_transition(
      environment, (1.25, {'success': 0.0}))
  assert reward == 0.0
  assert info['native_reward'] == 1.25
  assert info['native_step_api'] == 'metaworld_reward_info_2tuple'


def test_native_mode_rejects_missing_or_nonbinary_success():
  environment = _DummyEnvironment()
  sawyer_success.set_success_mode(environment, 'native_info')
  for info in ({}, {'success': 0.5}):
    try:
      sawyer_success.native_sparse_transition(
          environment, ('native_obs', 0.0, False, info))
    except RuntimeError:
      pass
    else:
      raise AssertionError(f'Expected invalid native info to fail: {info}')


def test_all_custom_sawyer_steps_offer_native_mode():
  source = (REPO_ROOT / 'env_utils.py').read_text(encoding='utf-8')
  assert source.count('native_result = super(Sawyer') == 13
  assert source.count(
      'native_sparse = _native_sparse_transition(self, native_result, action)') == 13
  assert "sawyer_success_mode='legacy_distance'" in source


def test_runtime_propagation_and_checkpoint_separation():
  runner = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  launcher = (REPO_ROOT / 'DRAFT.sh').read_text(encoding='utf-8')
  assert "'sawyer_success_mode', 'legacy_distance'" in runner
  assert "config_key += f'_success_{sawyer_success_mode}'" in runner
  assert runner.count(
      'sawyer_success_mode=FLAGS.sawyer_success_mode') >= 10
  assert '--sawyer_success_mode=$SAWYER_SUCCESS_MODE' in launcher


def main():
  tests = [value for name, value in globals().items()
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'native Sawyer success-wrapper tests passed ({len(tests)})')


if __name__ == '__main__':
  main()
