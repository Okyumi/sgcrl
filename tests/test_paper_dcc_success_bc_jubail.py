#!/usr/bin/env python3
"""Dependency-light checks for the Jubail DCC vs Success-BC 10-seed rerun."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_paper_dcc_success_bc_jubail as configs


def _load_status():
  spec = importlib.util.spec_from_file_location(
      'paper_dcc_success_bc_jubail_status',
      REPO_ROOT / 'scripts' / 'paper_dcc_success_bc_jubail_status.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


status = _load_status()


def test_exact_two_by_ten_matrix():
  runs = configs.build_configs()
  assert len(configs.VARIANTS) == 2
  assert len(configs.SEEDS) == 10
  assert len(runs) == 20
  assert set(configs.SEEDS) == set(range(5, 15))
  assert [run['variant'] for run in runs[:10]] == ['dcc_no_dyn_no_bc'] * 10
  assert [run['variant'] for run in runs[10:]] == ['dcc_success_bc'] * 10
  assert [run['seed'] for run in runs[:10]] == list(range(5, 15))
  assert [run['seed'] for run in runs[10:]] == list(range(5, 15))


def test_both_cells_disable_dynamics_and_keep_paper_metrics():
  for run in configs.build_configs():
    assert run['critic_mode'] == 'decomposed'
    assert run['actor_mode'] == 'reset'
    assert run['dyn_aux_weight'] == 0.0
    assert run['action_effect_enabled'] is False
    assert run['sawyer_success_mode'] == 'native_info'
    assert run['goal_conditioning_mode'] == 'full_state'
    assert run['num_tasks'] == 10
    assert run['steps_per_task'] == 8_000_000
    assert run['network_width'] == 1024
    assert run['eval_every'] == 100_000
    assert run['log_rl_metrics'] is True
    assert run['rl_metrics_occasional_multiplier'] == 2
    assert run['profile_runtime'] is True
    assert run['post_task_eval_scope'] == 'current'
    assert run['eval_record_video'] is False
    assert run['counterfactual_rank_interval_steps'] == 0
    assert run['shortcut_diagnostic_interval'] == 0
    assert run['success_bc_label_mode'] == 'episode_sparse_reward'


def test_ablation_has_no_bc_and_proposed_uses_success_bc():
  ablation = [c for c in configs.build_configs()
              if c['variant'] == 'dcc_no_dyn_no_bc']
  proposed = [c for c in configs.build_configs()
              if c['variant'] == 'dcc_success_bc']
  assert len(ablation) == 10 and len(proposed) == 10
  assert all(c['success_bc_weight'] == 0.0 for c in ablation)
  assert all(c['success_bc_weight'] == 0.1 for c in proposed)
  assert all(
      c['wandb_group'] == 'PAPER-DCC-NODYN-10SEED-ablation' for c in ablation)
  assert all(
      c['wandb_group'] == 'PAPER-DCC-NODYN-10SEED-success-bc' for c in proposed)


def test_seed_dir_match_separates_bc_from_ablation():
  prefix = configs.checkpoint_config_prefix()
  assert 'dyn0.000' in prefix
  assert configs.seed_dir_matches(prefix, 0.0)
  assert not configs.seed_dir_matches(prefix, 0.1)
  assert configs.seed_dir_matches(prefix + '_bridge_abc_ret_def', 0.1)
  assert not configs.seed_dir_matches(prefix + '_bridge_abc_ret_def', 0.0)


def test_status_treats_missing_checkpoints_as_incomplete():
  with tempfile.TemporaryDirectory() as tmp:
    ids = status.incomplete_array_ids(
        tasks_per_gpu=2, checkpoint_dir=tmp)
    assert ids == list(range(10))
    assert status.latest_completed_task(
        configs.build_configs()[0], tmp) == -1


def test_status_complete_requires_all_ten_task_pickles():
  run = configs.build_configs()[0]
  with tempfile.TemporaryDirectory() as tmp:
    seed_dir = (
        Path(tmp) / configs.checkpoint_config_prefix()
        / f'seed_{run["seed"]}')
    seed_dir.mkdir(parents=True)
    for task_id in range(9):
      (seed_dir / f'task_{task_id}.pkl').write_bytes(b'x')
    assert not status.is_complete(run, tmp)
    (seed_dir / 'task_9.pkl').write_bytes(b'x')
    assert status.is_complete(run, tmp)


def test_status_does_not_count_bc_ckpts_for_ablation():
  ablation = configs.build_configs()[0]
  proposed = configs.build_configs()[10]
  with tempfile.TemporaryDirectory() as tmp:
    bc_dir = (
        Path(tmp)
        / (configs.checkpoint_config_prefix() + '_bridge_abc_ret_def')
        / f'seed_{proposed["seed"]}')
    bc_dir.mkdir(parents=True)
    for task_id in range(10):
      (bc_dir / f'task_{task_id}.pkl').write_bytes(b'x')
    assert status.is_complete(proposed, tmp)
    assert not status.is_complete(ablation, tmp)


def test_learner_skips_dyn_step_when_weight_is_zero():
  source = (
      REPO_ROOT / 'contrastive' / 'continual_learning_decomposed.py'
  ).read_text(encoding='utf-8')
  tree = ast.parse(source)
  found = False
  for node in ast.walk(tree):
    if (isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == 'dyn_w'):
      found = True
      assert any(
          isinstance(cmp, ast.Gt) for cmp in node.test.ops)
  assert found
  assert 'if dyn_w > 0:' in source
  assert 'head is not forwarded' in source


def test_launcher_packs_two_jobs_and_chains_resubmits():
  launcher = (
      REPO_ROOT / 'DRAFT_paper_dcc_success_bc_jubail.sh').read_text(
          encoding='utf-8')
  draft = (REPO_ROOT / 'DRAFT_jubail.sh').read_text(encoding='utf-8')
  assert '#SBATCH --array=0-9' in launcher
  assert '#SBATCH --partition=nvidia' in launcher
  assert '#SBATCH --gres=gpu:a100:1' in launcher
  assert '#SBATCH --mail-type=FAIL' in launcher
  assert 'TASKS_PER_GPU="${TASKS_PER_GPU:-2}"' in launcher
  assert 'XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"' in launcher
  assert '--dependency="afterany:${SLURM_JOB_ID}"' in launcher
  assert 'DRAFT_jubail.sh' in launcher
  assert 'conda activate contrastive_rl' in launcher
  launcher_python = launcher.find('python tests/test_paper_dcc_success_bc_jubail.py')
  launcher_conda = launcher.find('conda activate contrastive_rl')
  assert launcher_conda != -1 and launcher_python != -1
  assert launcher_conda < launcher_python
  assert '--dyn_aux_weight=$DYN_AUX_WEIGHT' in draft
  assert '--success_bc_weight=$SUCCESS_BC_WEIGHT' in draft
  assert '--rl_metrics_occasional_multiplier=$RL_METRICS_OCCASIONAL_MULTIPLIER' in draft


def test_config_emission_forwards_no_dyn_and_bc_flags():
  ablation = subprocess.run(
      [sys.executable, 'experiment_configs_paper_dcc_success_bc_jubail.py',
       '--setting', '0'],
      cwd=REPO_ROOT, capture_output=True, text=True, check=True)
  assert 'DYN_AUX_WEIGHT=0.0' in ablation.stdout
  assert 'SUCCESS_BC_WEIGHT=0.0' in ablation.stdout
  assert 'CRITIC_MODE=decomposed' in ablation.stdout
  assert 'SAWYER_SUCCESS_MODE=native_info' in ablation.stdout
  proposed = subprocess.run(
      [sys.executable, 'experiment_configs_paper_dcc_success_bc_jubail.py',
       '--setting', '10'],
      cwd=REPO_ROOT, capture_output=True, text=True, check=True)
  assert 'DYN_AUX_WEIGHT=0.0' in proposed.stdout
  assert 'SUCCESS_BC_WEIGHT=0.1' in proposed.stdout
  assert 'SUCCESS_BC_LABEL_MODE=episode_sparse_reward' in proposed.stdout
  assert 'LOG_RL_METRICS=true' in proposed.stdout


if __name__ == '__main__':
  tests = [value for name, value in sorted(globals().items())
           if name.startswith('test_') and callable(value)]
  for test in tests:
    test()
  print(f'paper dcc success-bc jubail tests passed ({len(tests)})')
