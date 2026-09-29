#!/usr/bin/env python3
"""Compose the Task-5 / Task-6 episode filmstrip from saved NPZ rolls."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.episode_strip import compose_strips

DEFAULT_DATA = REPO_ROOT / 'results' / 'data' / 'task56_episode_strips'


def _load(path: Path) -> dict:
  data = dict(np.load(path, allow_pickle=True))
  data['phases'] = np.asarray(data['phases']).astype(str)
  return data


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--handle-npz',
      default=str(DEFAULT_DATA / 'handle_strip.npz'))
  parser.add_argument(
      '--push-npz',
      default=str(DEFAULT_DATA / 'push_strip.npz'))
  parser.add_argument(
      '--out-dir',
      default=str(REPO_ROOT / 'results' / 'img' / 'paper'))
  args = parser.parse_args()
  out_dir = Path(args.out_dir)
  compose_strips(
      _load(Path(args.handle_npz)),
      _load(Path(args.push_npz)),
      out_dir / 'fig_task56_episode_strip.pdf',
      out_dir / 'fig_task56_episode_strip.png')
  print('Wrote', out_dir / 'fig_task56_episode_strip.pdf')


if __name__ == '__main__':
  main()
