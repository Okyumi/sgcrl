#!/usr/bin/env python3
"""Launch, check credentials, or inspect task-boundary progress."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiment_configs_vanilla_crl_gate_bc import runner_arguments


def completed_tasks(root, seed):
  directories = list((root / 'checkpoints').glob(
      f'actor_reset_critic_reset_tid_False_heads_True_success_native_info*/seed_{seed}'))
  if len(directories) > 1:
    raise RuntimeError(f'Ambiguous checkpoints for seed {seed}: {directories}')
  if not directories:
    return 0
  present = {i for i in range(10) if (directories[0] / f'task_{i}.pkl').is_file()}
  count = len(present)
  if present != set(range(count)):
    raise RuntimeError(f'Noncontiguous task checkpoints for seed {seed}: {present}')
  return count


def preflight():
  import wandb
  if os.environ.get('WANDB_MODE') != 'online':
    raise RuntimeError('WANDB_MODE must be online for this experiment.')
  # Accessing viewer performs an authenticated read; never prompt or print keys.
  api = wandb.Api(timeout=30)
  if not api.api_key or not api.viewer:
    raise RuntimeError('W&B login missing. Run wandb login --host https://api.wandb.ai')
  print('W&B authentication OK. Entity:', os.environ.get('WANDB_ENTITY'), flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', choices=('run', 'smoke', 'status', 'complete', 'preflight'), required=True)
  parser.add_argument('--root', type=Path, required=True)
  parser.add_argument('--seed', type=int, choices=range(5, 15), default=5)
  args = parser.parse_args()
  if args.mode == 'preflight':
    preflight()
  elif args.mode == 'complete':
    return 0 if completed_tasks(args.root, args.seed) == 10 else 1
  elif args.mode == 'status':
    for seed in range(5, 15):
      print(f'Seed {seed:2d}: {completed_tasks(args.root, seed)}/10 tasks complete')
  else:
    command = [sys.executable, '-u', str(REPO / 'run_continual_contrastive.py')]
    command += runner_arguments(args.seed, args.root, args.mode == 'smoke')
    print('Launching:', ' '.join(command), flush=True)
    return subprocess.call(command, cwd=REPO)
  return 0


if __name__ == '__main__':
  sys.exit(main())
