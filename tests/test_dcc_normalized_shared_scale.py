#!/usr/bin/env python3
"""Dependency-light checks for the normalized DCC alpha sweep."""
from __future__ import annotations

import ast
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from experiment_configs_dcc_normalized_shared_scale_task5 import (
    PREFIX_CHECKPOINT_DIR,
    SEEDS,
    SHARED_SCALES,
    WANDB_GROUP,
    build_configs,
)


def test_exact_normalized_sweep_and_matched_seeds():
  configs = build_configs()
  assert len(configs) == 18
  assert {config['shared_repr_scale'] for config in configs} == set(
      SHARED_SCALES)
  for scale in SHARED_SCALES:
    assert {config['seed'] for config in configs
            if config['shared_repr_scale'] == scale} == set(SEEDS)
  assert {config['shared_repr_normalization'] for config in configs} == {
      'unit_mix'}
  assert {config['wandb_group'] for config in configs} == {WANDB_GROUP}


def test_sweep_reuses_existing_prefix_and_trains_only_task5():
  for config in build_configs():
    assert config['start_task'] == 5
    assert config['num_tasks'] == 6
    assert config['resume_checkpoint_dir'] == PREFIX_CHECKPOINT_DIR
    assert config['steps_per_task'] == 1_000_000
    assert config['critic_mode'] == 'decomposed'
    assert config['actor_mode'] == 'reset'
    assert config['combine_mode'] == 'add'
    assert config['energy_fn'] == 'inner_product'


def test_alpha_is_a_bounded_branch_mixture():
  for alpha in SHARED_SCALES:
    shared_weight = alpha / (alpha + 1.0)
    task_weight = 1.0 / (alpha + 1.0)
    assert 0.0 <= shared_weight < 1.0
    assert 0.0 < task_weight <= 1.0
    assert abs(shared_weight + task_weight - 1.0) < 1e-12


def test_implementation_covers_runner_learner_and_all_score_paths():
  network_source = (REPO_ROOT / 'contrastive/decomposed_networks.py').read_text()
  learner_source = (
      REPO_ROOT / 'contrastive/continual_learning_decomposed.py').read_text()
  runner_source = (REPO_ROOT / 'run_continual_contrastive.py').read_text()
  launcher_source = (
      REPO_ROOT / 'DRAFT_dcc_normalized_shared_scale_task5.sh').read_text()
  ast.parse(network_source)
  ast.parse(learner_source)
  ast.parse(runner_source)
  assert "shared_repr_normalization == 'unit_mix'" in network_source
  assert 'shared_repr_scale / denominator' in network_source
  assert '(1.0 / denominator) * _unit_norm(sa_task)' in network_source
  assert 'apply_goal_for_score' in network_source
  assert 'apply_sa_mixture_components' in learner_source
  assert "'decomp/shared_coefficient'" in learner_source
  assert "'decomp/effective_task_norm'" in learner_source
  assert 'FLAGS.shared_repr_normalization' in runner_source
  assert '--shared_repr_normalization="$SHARED_REPR_NORMALIZATION"' in launcher_source


def test_legacy_default_is_unchanged():
  config_source = (REPO_ROOT / 'contrastive/continual_config.py').read_text()
  network_source = (REPO_ROOT / 'contrastive/decomposed_networks.py').read_text()
  assert "shared_repr_normalization: str = 'none'" in config_source
  assert "shared_repr_normalization: str = 'none'" in network_source
  assert 'return shared_repr_scale * sa_shared + sa_task' in network_source


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'DCC normalized shared-scale tests passed ({len(tests)})')
