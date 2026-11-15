#!/usr/bin/env python3
"""Checks for the from-scratch continual Distral alpha pilot."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_dcc_distral_unit_shared_scale_continual as continual


def _clear_overrides():
  for key in ('ALPHA_SEEDS', 'ALPHA_SCALES', 'ALPHA_NUM_TASKS',
              'ALPHA_STEPS_PER_TASK'):
    os.environ.pop(key, None)


def _expect_exit(fn):
  try:
    fn()
  except SystemExit:
    return
  raise AssertionError('expected SystemExit')


def test_default_pilot_is_twelve_from_scratch_runs():
  _clear_overrides()
  configs = continual.build_configs()
  assert len(configs) == 12
  assert {config['shared_repr_scale'] for config in configs} == set(
      continual.SHARED_SCALES)
  for scale in continual.SHARED_SCALES:
    assert {config['seed'] for config in configs
            if config['shared_repr_scale'] == scale} == {5, 6}
  assert {config['num_tasks'] for config in configs} == {7}
  assert {config['steps_per_task'] for config in configs} == {1_000_000}
  assert {config['base_steps'] for config in configs} == {1_000_000}
  assert {config['shared_repr_normalization'] for config in configs} == {
      'unit_distral'}
  assert {config['wandb_group'] for config in configs} == {
      'DCC-DISTRAL-UNIT-ALPHA-CONTINUAL7-1M'}
  assert {config['wandb_project'] for config in configs} == {
      continual.WANDB_PROJECT}
  for config in configs:
    assert 'start_task' not in config
    assert 'resume_checkpoint_dir' not in config


def test_plain_dcc_settings_stay_fixed():
  _clear_overrides()
  for config in continual.build_configs():
    assert config['actor_mode'] == 'reset'
    assert config['critic_mode'] == 'decomposed'
    assert config['network_width'] == 1024
    assert config['critic_depth'] == 4
    assert config['actor_depth'] == 4
    assert config['dyn_aux_weight'] == 1.0
    assert config['phi_task_width'] == 256
    assert config['phi_task_depth'] == 4
    assert config['combine_mode'] == 'add'
    assert config['energy_fn'] == 'inner_product'
    assert config['eval_every'] == 50_000
    assert config['eval_episodes'] == 10
    assert config['sawyer_success_mode'] == 'native_info'
    assert config['goal_conditioning_mode'] == 'full_state'
    assert config['use_task_id'] is False
    assert config['post_task_eval_scope'] == 'current'
    assert config['interaction_weighted_relabeling'] is False
    assert config['action_effect_enabled'] is False
    assert config['success_bc_weight'] == 0.0
    assert config['counterfactual_rank_interval_steps'] == 0
    assert config['counterfactual_oracle_interval_steps'] == 0
    assert config['in_trajectory_negative_repeats'] == 1


def test_paths_include_alpha_and_seed_and_do_not_collide():
  _clear_overrides()
  configs = continual.build_configs()
  log_dirs = [config['log_dir'] for config in configs]
  checkpoint_dirs = [config['checkpoint_dir'] for config in configs]
  run_names = [config['run_name'] for config in configs]
  assert len(set(log_dirs)) == len(configs)
  assert len(set(checkpoint_dirs)) == len(configs)
  assert len(set(run_names)) == len(configs)
  for config in configs:
    tag = continual.scale_tag(config['shared_repr_scale'])
    needle = f"continual7_alpha{tag}_seed{config['seed']}"
    assert config['run_name'] == needle
    assert config['log_dir'].endswith(f'{needle}/runs')
    assert config['checkpoint_dir'].endswith(f'{needle}/checkpoints')
    assert f"seed{config['seed']}" in config['log_dir']


def test_array_covers_each_alpha_once():
  _clear_overrides()
  configs = continual.build_configs()
  n_seeds = len(continual.selected_seeds())
  n_alphas = len(continual.selected_scales())
  assert n_seeds == 2
  assert n_alphas == 6
  assert continual.selected_scales() == continual.SHARED_SCALES
  seen = []
  for array_id in range(n_alphas):
    chunk = configs[array_id * n_seeds:(array_id + 1) * n_seeds]
    assert len(chunk) == n_seeds
    assert len({config['shared_repr_scale'] for config in chunk}) == 1
    assert [config['seed'] for config in chunk] == [5, 6]
    seen.extend(chunk)
  assert seen == configs
  launcher = (
      REPO_ROOT / 'DRAFT_dcc_distral_unit_shared_scale_continual.sh'
  ).read_text()
  assert '#SBATCH --array=0-5' in launcher
  assert 'base_setting=$((ARRAY_TASK_ID * n_seeds))' in launcher
  assert '--seeds-per-alpha' in launcher
  assert '--shared_repr_scale="$SHARED_REPR_SCALE"' in launcher
  assert '--shared_repr_normalization="$SHARED_REPR_NORMALIZATION"' in launcher
  assert 'start_task' not in launcher
  assert 'resume_checkpoint' not in launcher
  assert 'SLURM_ARRAY_TASK_ID * 3' not in launcher


def test_full_three_seed_grid_is_still_available():
  _clear_overrides()
  os.environ['ALPHA_SEEDS'] = '5,6,7'
  try:
    configs = continual.build_configs()
  finally:
    os.environ.pop('ALPHA_SEEDS', None)
  assert len(configs) == 18
  for scale in continual.SHARED_SCALES:
    assert {config['seed'] for config in configs
            if config['shared_repr_scale'] == scale} == {5, 6, 7}


def test_num_tasks_accepts_only_seven_nine_or_ten():
  _clear_overrides()
  os.environ['ALPHA_NUM_TASKS'] = '9'
  try:
    configs = continual.build_configs()
    assert {config['num_tasks'] for config in configs} == {9}
    assert {config['wandb_group'] for config in configs} == {
        'DCC-DISTRAL-UNIT-ALPHA-CONTINUAL9-1M'}
  finally:
    os.environ.pop('ALPHA_NUM_TASKS', None)
  os.environ['ALPHA_NUM_TASKS'] = '10'
  try:
    assert {config['num_tasks']
            for config in continual.build_configs()} == {10}
  finally:
    os.environ.pop('ALPHA_NUM_TASKS', None)
  os.environ['ALPHA_NUM_TASKS'] = '8'
  try:
    _expect_exit(continual.build_configs)
  finally:
    os.environ.pop('ALPHA_NUM_TASKS', None)


def test_invalid_alpha_and_seed_fail_early():
  _clear_overrides()
  os.environ['ALPHA_SCALES'] = '0,0.25,0.5,0.75,1,2'
  try:
    _expect_exit(continual.selected_scales)
  finally:
    os.environ.pop('ALPHA_SCALES', None)
  os.environ['ALPHA_SEEDS'] = '5,8'
  try:
    _expect_exit(continual.selected_seeds)
  finally:
    os.environ.pop('ALPHA_SEEDS', None)
  os.environ['ALPHA_SEEDS'] = '5,5'
  try:
    _expect_exit(continual.selected_seeds)
  finally:
    os.environ.pop('ALPHA_SEEDS', None)


def test_completion_looks_for_the_final_task_checkpoint():
  with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    assert continual.is_curriculum_complete(root, 7) is False
    target = root / 'actor_reset' / 'seed_5'
    target.mkdir(parents=True)
    (target / 'task_5.pkl').write_bytes(b'partial')
    assert continual.is_curriculum_complete(root, 7) is False
    (target / 'task_6.pkl').write_bytes(b'done')
    assert continual.is_curriculum_complete(root, 7) is True


def test_original_unnormalized_default_is_unchanged():
  source = (REPO_ROOT / 'contrastive/continual_config.py').read_text()
  assert "shared_repr_normalization: str = 'none'" in source


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'DCC Distral continual tests passed ({len(tests)})')
