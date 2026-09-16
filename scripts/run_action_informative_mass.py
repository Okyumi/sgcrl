#!/usr/bin/env python3
"""Run action-informative-mass extraction on exact env config dirs."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.action_informative_mass import TASK_SPECS
from contrastive.feature_shortcut import env_config_dir_matches
import re

STEP_RE = re.compile(r'task_0_step_(\d+)\.pkl$')

PRIMARY_STEPS = {
    'sawyer_handle_press_side': 100000,
    'sawyer_window_close': 250000,
    'sawyer_push': 250000,
}


def list_step_ckpts(checkpoint_dir: Path, env_name: str, seed: int):
  matches = []
  seed_dir = f'seed_{seed}'
  for path in checkpoint_dir.rglob('task_0_step_*.pkl'):
    if path.parent.name != seed_dir:
      continue
    if not env_config_dir_matches(path, env_name):
      continue
    match = STEP_RE.search(path.name)
    if not match:
      continue
    matches.append((int(match.group(1)), path))
  matches.sort()
  return matches


def pick_target(ckpts, want: int):
  if not ckpts:
    return None
  return min(ckpts, key=lambda item: abs(item[0] - want))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint-dir', required=True)
  parser.add_argument('--seed', type=int, default=6)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--episodes', type=int, default=40)
  parser.add_argument(
      '--envs',
      default='sawyer_handle_press_side,sawyer_window_close,sawyer_push')
  args = parser.parse_args()

  probe = REPO_ROOT / 'scripts' / 'measure_action_informative_mass.py'
  ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
  out_dir = Path(args.output_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  npz_paths = []
  for env_name in [e.strip() for e in args.envs.split(',') if e.strip()]:
    if env_name not in TASK_SPECS:
      raise SystemExit(f'Unknown env {env_name}')
    ckpts = list_step_ckpts(ckpt_dir, env_name, args.seed)
    if not ckpts:
      print(f'SKIP {env_name}: no checkpoints under {ckpt_dir}', flush=True)
      continue
    want = PRIMARY_STEPS.get(env_name, ckpts[-1][0])
    step, path = pick_target(ckpts, want)
    tag = f"{env_name.replace('sawyer_', '')}_s{args.seed}_step{step}"
    output = out_dir / f'action_mass_{tag}.json'
    cmd = [
        sys.executable, str(probe),
        '--checkpoint', str(path),
        '--env-name', env_name,
        '--seed', str(args.seed),
        '--episodes', str(args.episodes),
        '--output', str(output),
    ]
    print('Running:', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd)
    npz_paths.append(str(output.with_suffix('.npz')))

  if len(npz_paths) >= 2:
    plot = REPO_ROOT / 'scripts' / 'plot_action_informative_mass.py'
    subprocess.check_call(
        [sys.executable, str(plot), '--npz', *npz_paths,
         '--out-dir', str(REPO_ROOT / 'results' / 'img' / 'paper'),
         '--csv', str(REPO_ROOT / 'results' / 'data' /
                      'action_informative_mass' /
                      'fig_action_informative_mass.csv')])
  return 0 if npz_paths else 1


if __name__ == '__main__':
  raise SystemExit(main())
