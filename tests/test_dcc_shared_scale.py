#!/usr/bin/env python3
"""Dependency-light checks for the fixed DCC shared-scale pilot."""
from __future__ import annotations

import ast
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from experiment_configs_dcc_shared_scale_task5 import (
    PREFIX_CHECKPOINT_DIR,
    build_configs,
    build_prefix_configs,
)


def test_exact_sweep_and_matched_seeds():
  configs = build_configs()
  assert len(configs) == 18
  assert {config['shared_repr_scale'] for config in configs} == {
      0.0, 0.25, 0.5, 0.75, 1.0, 1.5
  }
  for scale in {config['shared_repr_scale'] for config in configs}:
    assert {config['seed'] for config in configs
            if config['shared_repr_scale'] == scale} == {5, 6, 7}


def test_pilot_is_prefix_through_task5_and_keeps_dynamics_on():
  for config in build_configs():
    assert config['num_tasks'] == 6
    assert config['start_task'] == 5
    assert config['resume_checkpoint_dir'] == PREFIX_CHECKPOINT_DIR
    assert config['steps_per_task'] == 1_000_000
    assert config['critic_mode'] == 'decomposed'
    assert config['actor_mode'] == 'reset'
    assert config['dyn_aux_weight'] == 1.0
    assert config['combine_mode'] == 'add'
    assert config['energy_fn'] == 'inner_product'
    assert config['sawyer_success_mode'] == 'native_info'


def test_prefix_is_trained_once_per_seed():
  configs = build_prefix_configs()
  assert len(configs) == 3
  assert {config['seed'] for config in configs} == {5, 6, 7}
  for config in configs:
    assert config['num_tasks'] == 5
    assert config['shared_repr_scale'] == 1.0


def test_scale_changes_every_dcc_scoring_path_and_is_logged():
  network_source = (REPO_ROOT / 'contrastive/decomposed_networks.py').read_text()
  learner_source = (
      REPO_ROOT / 'contrastive/continual_learning_decomposed.py').read_text()
  runner_source = (REPO_ROOT / 'run_continual_contrastive.py').read_text()
  ast.parse(network_source)
  ast.parse(learner_source)
  ast.parse(runner_source)
  assert 'shared_repr_scale * sa_shared + sa_task' in network_source
  assert 'apply_score_with_components' in learner_source
  assert "'decomp/shared_score_fraction'" in learner_source
  assert "'decomp/scaled_shared_to_task_norm'" in learner_source
  assert "'shared_repr_scale': FLAGS.shared_repr_scale" in runner_source
  assert "FLAGS.resume_checkpoint_dir or FLAGS.checkpoint_dir" in runner_source


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'DCC shared-scale tests passed ({len(tests)})')
