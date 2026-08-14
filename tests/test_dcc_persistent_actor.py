"""Dependency-light checks for persistent-actor DCC."""
import ast
from pathlib import Path

import experiment_configs


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / 'run_continual_contrastive.py').read_text()
LEARNER = (
    ROOT / 'contrastive' / 'continual_learning_decomposed.py').read_text()


def test_active_batch_is_the_two_matched_persistent_actor_dcc_cells():
  configs = experiment_configs.build_configs()
  assert len(configs) == 6
  assert {config['actor_mode'] for config in configs} == {'persistent'}
  assert {config['critic_mode'] for config in configs} == {'decomposed'}
  assert {config['seed'] for config in configs} == {5, 6, 7}
  assert {config['dyn_aux_weight'] for config in configs} == {0.0, 1.0}
  assert all(not config['single_task'] for config in configs)
  assert all(config['actor_auto_reset'] is False for config in configs)
  assert all(config['shortcut_diagnostic_interval'] == 0
             for config in configs)


def test_decomposed_learner_carries_complete_actor_training_state():
  for token in (
      "carry_actor = actor_mode == 'persistent' and task_id > 0",
      'policy_params = prev_policy_params',
      'policy_opt_state = prev_policy_opt_state',
      'alpha_params = prev_alpha_params',
      'alpha_opt_state = prev_alpha_opt_state',
  ):
    assert token in LEARNER
  assert 'Persistent DCC actor requires previous policy parameters' in LEARNER
  assert 'Persistent DCC actor requires previous entropy parameters' in LEARNER


def test_runner_round_trips_persistent_dcc_actor_through_checkpoints():
  for token in (
      'prev_policy_params=prev_dcc_policy_params',
      'prev_policy_opt_state=prev_dcc_policy_opt_state',
      'prev_alpha_params=prev_dcc_alpha_params',
      'prev_alpha_opt_state=prev_dcc_alpha_opt_state',
      "ckpt_data['decomposed_policy_params']",
      "ckpt_data['decomposed_policy_opt_state']",
      "ckpt_data['decomposed_alpha_params']",
      "ckpt_data['decomposed_alpha_opt_state']",
  ):
    assert token in RUNNER


def test_train_single_task_return_shape_is_consistent():
  module = ast.parse(RUNNER)
  fn = next(
      node for node in module.body
      if isinstance(node, ast.FunctionDef) and node.name == 'train_single_task')
  tuple_returns = [
      node.value for node in ast.walk(fn)
      if (isinstance(node, ast.Return)
          and isinstance(node.value, ast.Tuple)
          and len(node.value.elts) == 21)]
  assert len(tuple_returns) == 2
