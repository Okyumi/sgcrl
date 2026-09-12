#!/usr/bin/env python3
"""Pick mid-task checkpoints and run the same-state action-advice probe.

Used on Jubail after a 1M training cell so we do not need Torch checkpoints.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.feature_shortcut import env_config_dir_matches


STEP_RE = re.compile(r'task_0_step_(\d+)\.pkl$')


def list_step_ckpts(checkpoint_dir: Path, env_name: str, seed: int):
  matches = []
  seed_dir = f'seed_{seed}'
  for path in checkpoint_dir.rglob('task_0_step_*.pkl'):
    if not env_config_dir_matches(path, env_name):
      continue
    if path.parent.name != seed_dir:
      continue
    match = STEP_RE.search(path.name)
    if not match:
      continue
    matches.append((int(match.group(1)), path))
  matches.sort()
  return matches


def pick_targets(ckpts, targets):
  if not ckpts:
    return []
  by_step = {step: path for step, path in ckpts}
  chosen = []
  seen = set()
  for raw in targets:
    token = raw.strip().lower()
    if not token:
      continue
    if token in ('latest', 'last'):
      step, path = ckpts[-1]
    elif token in ('earliest', 'first'):
      step, path = ckpts[0]
    else:
      want = int(token)
      step, path = min(ckpts, key=lambda item: abs(item[0] - want))
    key = str(path)
    if key in seen:
      continue
    seen.add(key)
    chosen.append((step, path))
  return chosen


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint-dir', required=True)
  parser.add_argument('--env-name', required=True)
  parser.add_argument('--seed', type=int, required=True)
  parser.add_argument('--targets', default='latest')
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--episodes', type=int, default=40)
  parser.add_argument('--n-random-actions', type=int, default=32)
  args = parser.parse_args()

  repo = Path(__file__).resolve().parents[1]
  probe = repo / 'scripts' / 'measure_hover_action_advice.py'
  ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
  out_dir = Path(args.output_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  ckpts = list_step_ckpts(ckpt_dir, args.env_name, args.seed)
  if not ckpts:
    print(f'No mid-task checkpoints for {args.env_name} seed={args.seed} '
          f'under {ckpt_dir}', file=sys.stderr)
    return 1

  targets = [t for t in args.targets.split(',') if t.strip()]
  chosen = pick_targets(ckpts, targets)
  print(f'Found {len(ckpts)} mid-task checkpoints; probing {len(chosen)}.')
  for step, path in chosen:
    tag = f"{args.env_name.replace('sawyer_', '')}_s{args.seed}_step{step}"
    output = out_dir / f'action_advice_{tag}.json'
    cmd = [
        sys.executable, str(probe),
        '--checkpoint', str(path),
        '--env-name', args.env_name,
        '--seed', str(args.seed),
        '--episodes', str(args.episodes),
        '--n-random-actions', str(args.n_random_actions),
        '--output', str(output),
    ]
    print('Running:', ' '.join(cmd), flush=True)
    subprocess.check_call(cmd)
    print(f'Wrote {output}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
