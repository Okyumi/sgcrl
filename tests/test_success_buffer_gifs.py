#!/usr/bin/env python3
"""Unit checks for D_succ ring unwrap / split (no MuJoCo)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def _load():
  path = REPO_ROOT / 'scripts' / 'record_success_buffer_gifs.py'
  spec = importlib.util.spec_from_file_location('record_success_buffer_gifs', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_unwrap_ring_fifo():
  mod = _load()
  cap = 8
  obs = np.arange(cap * 22, dtype=np.float32).reshape(cap, 22)
  raw = mod.unwrap_ring(obs, size=3, index=3)
  np.testing.assert_array_equal(raw, obs[:3])
  wrapped = mod.unwrap_ring(obs, size=8, index=5)
  np.testing.assert_array_equal(wrapped, np.concatenate([obs[5:], obs[:5]]))


def test_split_episodes_keeps_150():
  mod = _load()
  obs = np.arange(750 * 22, dtype=np.float32).reshape(750, 22)
  segs = mod.split_episodes(obs, episode_len=150)
  assert len(segs) == 5
  assert all(s.shape[0] == 150 for s in segs)
  leftover = np.arange(175 * 22, dtype=np.float32).reshape(175, 22)
  segs = mod.split_episodes(leftover, episode_len=150)
  assert [s.shape[0] for s in segs] == [150, 25]


def test_split_contiguous_object_jumps():
  mod = _load()
  a = np.zeros((40, 22), dtype=np.float32)
  a[:, 4:7] = np.linspace(0.0, 0.04, 40)[:, None]
  b = np.zeros((30, 22), dtype=np.float32)
  b[:, 4:7] = 0.8
  obs = np.concatenate([a, b], axis=0)
  segs = mod.split_contiguous(obs, jump=0.12, min_len=20)
  assert len(segs) == 2
  assert segs[0].shape[0] == 40
  assert segs[1].shape[0] == 30


def test_segment_stats_uses_stored_goal():
  mod = _load()
  obs = np.zeros((10, 22), dtype=np.float32)
  obs[:, 4:7] = np.array([0.02, 0.89, 0.30])
  obs[:, 15:18] = np.array([0.02, 0.89, 0.30])
  stats = mod.segment_stats(obs)
  assert stats['last_in_goal']
  assert stats['frac_in_goal'] == 1.0
  obs[:, 4:7] = np.array([0.4, 0.4, 0.0])
  stats = mod.segment_stats(obs)
  assert not stats['last_in_goal']
  assert stats['any_in_goal'] is False


def test_default_snapshots_exist():
  mod = _load()
  names = {item['name'] for item in mod.default_snapshots()}
  assert 'terminal_s6_first' in names
  assert 'warmup_s6_first' in names
  assert 'window_s6_first' in names
  first = [item for item in mod.default_snapshots()
           if item['name'] == 'terminal_s6_first'][0]
  assert Path(first['path']).is_file()
  window = [item for item in mod.default_snapshots()
            if item['name'] == 'window_s6_first'][0]
  assert Path(window['path']).is_dir()
  launcher = (REPO_ROOT / 'DRAFT_jubail_task7_success_buffer_gifs.sh').read_text(
      encoding='utf-8')
  assert 'record_success_buffer_gifs.py' in launcher
  assert 'tests/test_success_buffer_gifs.py' in launcher
  assert '--episode-len' in launcher


def test_path_frames_covers_every_step():
  mod = _load()
  obs = np.zeros((12, 22), dtype=np.float32)
  obs[:, 4:7] = np.linspace(0.0, 0.05, 12)[:, None]
  obs[:, 15:18] = np.array([0.02, 0.89, 0.30])
  frames = mod.path_frames(obs, size=80)
  assert frames.ndim == 4
  assert frames.shape[0] == 12
  assert frames.dtype == np.uint8


if __name__ == '__main__':
  test_unwrap_ring_fifo()
  test_split_episodes_keeps_150()
  test_split_contiguous_object_jumps()
  test_segment_stats_uses_stored_goal()
  test_default_snapshots_exist()
  test_path_frames_covers_every_step()
  print('test_success_buffer_gifs: ok')
