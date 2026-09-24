#!/usr/bin/env python3
"""Checks for terminal Success-BC with λ warmup."""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_success_bc_warmup as cfg
from contrastive import success_bc_labels


def test_warmup_scale():
  assert success_bc_labels.warmup_scale(0, 8) == 0.0
  assert success_bc_labels.warmup_scale(150, 8, 150) == 150 / 1200
  assert success_bc_labels.warmup_scale(1200, 8, 150) == 1.0
  assert success_bc_labels.warmup_scale(4096, 8, 150) == 1.0
  assert success_bc_labels.warmup_scale(150, 0) == 1.0


def test_twenty_resume_cells():
  configs = cfg.build_configs()
  assert cfg.SEEDS == tuple(range(5, 15))
  assert len(configs) == 20
  assert [c['start_task'] for c in configs] == [4] * 10 + [7] * 10
  for config in configs:
    assert config['success_bc_label_mode'] == 'terminal_episode'
    assert config['success_bc_weight'] == 0.1
    assert config['success_bc_warmup_episodes'] == 8
    assert config['success_buffer_capacity'] == 4096
    assert config['steps_per_task'] == 8_000_000
    assert Path(config['resume_checkpoint_file']).is_file()


def test_learner_scales_lambda_not_the_buffer():
  learner = (REPO_ROOT / 'contrastive/continual_learning_decomposed.py').read_text(
      encoding='utf-8')
  assert 'success_bc_warmup_episodes' in learner
  assert 'bc_scale' in learner
  assert 'success_bc_weight * bc_active * bc_scale * bc_loss' in learner
  runner = (REPO_ROOT / 'run_continual_contrastive.py').read_text(
      encoding='utf-8')
  assert 'success_bc_warmup_episodes' in runner
  launcher = (REPO_ROOT / 'DRAFT_jubail.sh').read_text(encoding='utf-8')
  assert 'success_bc_warmup_episodes=$SUCCESS_BC_WARMUP_EPISODES' in launcher
  job = (REPO_ROOT / 'DRAFT_jubail_success_bc_warmup.sh').read_text(
      encoding='utf-8')
  assert 'experiment_configs_success_bc_warmup.py' in job
  assert '#SBATCH --array=0-19' in job


if __name__ == '__main__':
  test_warmup_scale()
  test_twenty_resume_cells()
  test_learner_scales_lambda_not_the_buffer()
  print('test_success_bc_warmup: ok')
