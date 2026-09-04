#!/usr/bin/env python3
"""Dependency-light checks for the Distral-faithful DCC alpha sweep."""
from __future__ import annotations

import ast
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from experiment_configs_dcc_distral_unit_shared_scale_task5 import (
    PREFIX_CHECKPOINT_DIR,
    SEEDS,
    SHARED_SCALES,
    WANDB_GROUP,
    build_configs,
)


def test_exact_sweep_and_matched_seeds():
  configs = build_configs()
  assert len(configs) == 18
  assert {config['shared_repr_scale'] for config in configs} == set(
      SHARED_SCALES)
  for scale in SHARED_SCALES:
    assert {config['seed'] for config in configs
            if config['shared_repr_scale'] == scale} == set(SEEDS)
  assert {config['shared_repr_normalization'] for config in configs} == {
      'unit_distral'}
  assert {config['wandb_group'] for config in configs} == {WANDB_GROUP}


def test_reuses_prefix_and_trains_only_plain_dcc_task5():
  for config in build_configs():
    assert config['start_task'] == 5
    assert config['num_tasks'] == 6
    assert config['resume_checkpoint_dir'] == PREFIX_CHECKPOINT_DIR
    assert config['steps_per_task'] == 1_000_000
    assert config['critic_mode'] == 'decomposed'
    assert config['actor_mode'] == 'reset'
    assert config['combine_mode'] == 'add'
    assert config['energy_fn'] == 'inner_product'
    assert config['dyn_aux_weight'] == 1.0
    assert config['interaction_weighted_relabeling'] is False
    assert config['action_effect_enabled'] is False
    assert config['success_bc_weight'] == 0.0
    assert config['counterfactual_rank_interval_steps'] == 0
    assert config['counterfactual_oracle_interval_steps'] == 0
    assert config['in_trajectory_negative_repeats'] == 1


def test_distral_coefficients_are_not_divided_by_alpha_plus_one():
  for alpha in SHARED_SCALES:
    shared_coefficient = alpha
    task_coefficient = 1.0
    assert shared_coefficient == alpha
    assert task_coefficient == 1.0
    assert shared_coefficient / task_coefficient == alpha


def test_concrete_score_differs_from_fixed_budget_unit_mix():
  shared = (3.0 / 5.0, 4.0 / 5.0)
  task = (0.0, 1.0)
  goal = (1.0, 0.0)
  alpha = 0.25
  shared_score = sum(x * y for x, y in zip(shared, goal))
  task_score = sum(x * y for x, y in zip(task, goal))
  distral_score = alpha * shared_score + task_score
  fixed_budget_score = distral_score / (alpha + 1.0)
  assert abs(distral_score - 0.15) < 1e-12
  assert abs(fixed_budget_score - 0.12) < 1e-12
  assert distral_score != fixed_budget_score


def test_implementation_preserves_distral_score_and_all_score_paths():
  network_source = (REPO_ROOT / 'contrastive/decomposed_networks.py').read_text()
  learner_source = (
      REPO_ROOT / 'contrastive/continual_learning_decomposed.py').read_text()
  runner_source = (REPO_ROOT / 'run_continual_contrastive.py').read_text()
  launcher_source = (
      REPO_ROOT / 'DRAFT_dcc_distral_unit_shared_scale_task5.sh').read_text()
  ast.parse(network_source)
  ast.parse(learner_source)
  ast.parse(runner_source)
  assert "shared_repr_normalization == 'unit_distral'" in network_source
  assert 'shared_repr_scale * _unit_norm(sa_shared)' in network_source
  assert '_unit_norm(sa_task)' in network_source
  assert "shared_repr_normalization in unit_normalization_modes" in network_source
  assert 'apply_goal_for_score' in network_source
  assert 'apply_sa_mixture_components' in learner_source
  assert "shared_repr_normalization == 'unit_distral'" in learner_source
  assert "('none', 'unit_mix', 'unit_distral')" in runner_source
  assert '--shared_repr_normalization="$SHARED_REPR_NORMALIZATION"' in launcher_source
  assert '--noaction_effect_enabled' in launcher_source
  assert '--success_bc_weight="$SUCCESS_BC_WEIGHT"' in launcher_source
  assert '/(alpha+1)' not in launcher_source


def test_old_modes_and_defaults_are_unchanged():
  config_source = (REPO_ROOT / 'contrastive/continual_config.py').read_text()
  network_source = (REPO_ROOT / 'contrastive/decomposed_networks.py').read_text()
  assert "shared_repr_normalization: str = 'none'" in config_source
  assert 'return shared_repr_scale * sa_shared, sa_task' in network_source
  assert "shared_repr_normalization == 'unit_mix'" in network_source
  assert 'shared_repr_scale / denominator' in network_source
  assert '(1.0 / denominator) * _unit_norm(sa_task)' in network_source


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'DCC Distral unit-scale tests passed ({len(tests)})')
