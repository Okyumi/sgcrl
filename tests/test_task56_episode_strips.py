#!/usr/bin/env python3
"""Tests for the Task-5 / Task-6 episode filmstrip composer."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.episode_strip import (
    compose_strips,
    crop_workspace,
    even_indices,
    pack_episode,
    phase_sequence,
    select_strip_indices,
    success_band,
)
from scripts.record_task56_episode_strips import expand_gif_to_horizon
from contrastive.her_future_phase import PHASE_FAR, PHASE_HOVER, PHASE_SUCCESS


def test_even_indices_includes_ends():
  idx = even_indices(151, 8)
  assert idx[0] == 0
  assert idx[-1] == 150
  assert len(idx) <= 8


def test_select_strip_keeps_first_success():
  phases = [PHASE_FAR] * 40 + [PHASE_HOVER] + [PHASE_SUCCESS] * 110
  idx = select_strip_indices(len(phases), phases, 8)
  assert 0 in idx and (len(phases) - 1) in idx
  assert 41 in idx
  assert np.min(np.diff(idx)) >= 6
  assert len(idx) == 8


def test_crop_and_success_band():
  frame = np.zeros((100, 200, 3), dtype=np.uint8)
  frame[20:70, 40:160] = 9
  cropped = crop_workspace(frame)
  assert cropped.shape[0] < 100 and cropped.shape[1] < 200
  lo, hi = success_band('sawyer_handle_press_side')
  assert abs(lo - 0.05) < 1e-6 and abs(hi - 0.09) < 1e-6
  plo, phi = success_band('sawyer_push')
  assert plo == 0.0 and abs(phi - 0.05) < 1e-6


def test_resize_and_pack(tmp_path: Path):
  frames = np.zeros((12, 48, 64, 3), dtype=np.uint8)
  frames[:, :, :, 0] = 40
  frames[:, 10:30, 15:45, 1] = 80
  states = np.zeros((12, 7), dtype=np.float32)
  states[:, 1] = 0.9
  states[:, 2] = 0.2
  rec = pack_episode(
      frames, states, 'sawyer_push',
      env_steps=250500, success=1.0, attempts=1, checkpoint='x')
  assert rec['frames'].shape[0] == 12
  assert rec['coords'].shape == (12,)
  handle_states = np.zeros((12, 7), dtype=np.float32)
  handle_states[:, 1] = 0.7
  handle_states[:, 2] = 0.15
  handle_states[:, 6] = 0.16
  handle = pack_episode(
      frames, handle_states, 'sawyer_handle_press_side',
      env_steps=150300, success=1.0, attempts=6, checkpoint='y')
  pdf = tmp_path / 'strip.pdf'
  png = tmp_path / 'strip.png'
  compose_strips(handle, rec, pdf, png, n_keep=4)
  assert pdf.exists() and png.exists()


def test_handle_hover_phase():
  state = np.array([-0.07, 0.70, 0.10, 0.3, -0.07, 0.70, 0.12], np.float32)
  assert phase_sequence(state[None, :], 'sawyer_handle_press_side')[0] in (
      PHASE_HOVER, PHASE_FAR, PHASE_SUCCESS)


def test_expand_gif_to_horizon():
  gif = np.stack([np.full((4, 6, 3), i, dtype=np.uint8) for i in (1, 7, 9)], axis=0)
  out = expand_gif_to_horizon(gif, 9)
  assert out.shape == (9, 4, 6, 3)
  assert out[0, 0, 0, 0] == 1
  assert out[-1, 0, 0, 0] == 9


def main():
  test_even_indices_includes_ends()
  test_select_strip_keeps_first_success()
  test_crop_and_success_band()
  test_handle_hover_phase()
  test_expand_gif_to_horizon()
  with tempfile.TemporaryDirectory() as tmp:
    test_resize_and_pack(Path(tmp))
  print('task56 episode-strip tests passed')


if __name__ == '__main__':
  main()
