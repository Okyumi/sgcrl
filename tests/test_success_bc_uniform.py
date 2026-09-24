#!/usr/bin/env python3
"""Checks for recency sampling and the Task 7→5/8 uniform grid."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import experiment_configs_success_bc_uniform as cfg
from contrastive import success_bc_labels


def test_recency_prefers_newest_slot():
  logits = success_bc_labels.recency_logits(
      size=300, index=300, capacity=4096, half_life=4, episode_len=150)
  filled = logits > -1e8
  assert int(np.sum(filled)) == 300
  assert np.argmax(logits) == 299


def test_fifty_cells_task7_then_58():
  configs = cfg.build_configs()
  assert len(configs) == 50
  assert [c['start_task'] for c in configs[0:10]] == [7] * 10
  assert [c['start_task'] for c in configs[10:20]] == [5] * 10
  assert [c['start_task'] for c in configs[20:30]] == [5] * 10
  assert [c['start_task'] for c in configs[30:40]] == [8] * 10
  assert [c['start_task'] for c in configs[40:50]] == [8] * 10
  recency = configs[0:10] + configs[20:30] + configs[40:50]
  warmup = configs[10:20] + configs[30:40]
  for config in recency:
    assert config['success_bc_recency_half_life'] == 4.0
    assert config['success_bc_warmup_episodes'] == 0
    assert config['success_bc_label_mode'] == 'terminal_episode'
    assert Path(config['resume_checkpoint_file']).is_file()
  for config in warmup:
    assert config['success_bc_warmup_episodes'] == 8
    assert config['success_bc_recency_half_life'] == 0.0
    assert Path(config['resume_checkpoint_file']).is_file()


def test_learner_has_recency_branch():
  learner = (REPO_ROOT / 'contrastive/continual_learning_decomposed.py').read_text(
      encoding='utf-8')
  assert 'success_bc_recency_half_life' in learner
  assert 'jax.random.categorical' in learner
  launcher = (REPO_ROOT / 'DRAFT_jubail.sh').read_text(encoding='utf-8')
  assert 'success_bc_recency_half_life=$SUCCESS_BC_RECENCY_HALF_LIFE' in launcher
  job = (REPO_ROOT / 'DRAFT_jubail_success_bc_uniform.sh').read_text(
      encoding='utf-8')
  assert '#SBATCH --array=0-49' in job


if __name__ == '__main__':
  test_recency_prefers_newest_slot()
  test_fifty_cells_task7_then_58()
  test_learner_has_recency_branch()
  print('test_success_bc_uniform: ok')
