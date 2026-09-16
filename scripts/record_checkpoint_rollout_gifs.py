#!/usr/bin/env python3
"""Offline GIF rollouts from decomposed mid-task checkpoints.

Picks ~10 evenly spaced 1M-step snapshots (plus an optional first-success
hunt) and writes compact GIFs. Used for DCC Task-5 / Task-6 visual evidence
when W&B mp4s were not saved to disk.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.feature_shortcut import env_config_dir_matches

STEP_RE = re.compile(r'task_0_step_(\d+)\.pkl$')

FIXED_GOALS = {
    'sawyer_handle_press_side': np.array([-0.07, 0.68, 0.07], dtype=np.float32),
    'sawyer_push': np.array([0.02, 0.89, 0.02], dtype=np.float32),
}

DEFAULT_EVEN_TARGETS = (
    100_000, 200_000, 300_000, 400_000, 500_000,
    600_000, 700_000, 800_000, 900_000, 1_000_000,
)


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
  chosen = []
  seen = set()
  for raw in targets:
    token = str(raw).strip().lower()
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


class _PolicyActor:
  def __init__(self, apply_mode):
    self._apply_mode = apply_mode

  def select_action(self, observation):
    return np.asarray(self._apply_mode(observation), dtype=np.float32)


def even_target_tokens(n: int = 10) -> list:
  if n <= 0:
    return []
  if n >= len(DEFAULT_EVEN_TARGETS):
    return [str(step) for step in DEFAULT_EVEN_TARGETS]
  idx = np.linspace(0, len(DEFAULT_EVEN_TARGETS) - 1, n).round().astype(int)
  seen = []
  for i in idx:
    token = str(DEFAULT_EVEN_TARGETS[int(i)])
    if token not in seen:
      seen.append(token)
  return seen


def _load_actor(ckpt_path: Path, env_name: str, seed: int, args):
  import jax
  from acme import specs

  import contrastive
  from contrastive import utils as contrastive_utils

  with ckpt_path.open('rb') as handle:
    ckpt = pickle.load(handle)
  if 'decomposed_training_state' not in ckpt:
    raise KeyError(f'{ckpt_path} is not a decomposed mid-task checkpoint')

  start_index = int(ckpt.get('goal_start_index', 0))
  end_index = int(ckpt.get('goal_end_index', -1))
  success_mode = ckpt.get('sawyer_success_mode', 'corrected')
  environment, obs_dim = contrastive_utils.make_environment(
      env_name, start_index, end_index, seed,
      fixed_start_end=FIXED_GOALS[env_name],
      sawyer_success_mode=success_mode)
  obs_dim = int(ckpt.get('obs_dim') or obs_dim)
  env_spec = specs.make_environment_spec(environment)
  networks = contrastive.make_networks(
      env_spec, obs_dim=obs_dim,
      hidden_layer_sizes=(args.network_width, args.network_width),
      use_residual=True, network_width=args.network_width,
      critic_depth=args.critic_depth, actor_depth=args.actor_depth)
  policy_params = ckpt.get('decomposed_policy_params') or ckpt.get(
      'composed_policy')
  if policy_params is None:
    raise KeyError(f'{ckpt_path} has no policy parameters')

  def _mode_or_mean(observation):
    dist_params = networks.policy_network.apply(policy_params, observation)
    if hasattr(dist_params, 'mode'):
      return dist_params.mode()
    return dist_params.mean()

  _mode_jit = jax.jit(_mode_or_mean)
  actor = _PolicyActor(lambda obs: np.asarray(_mode_jit(obs)))
  return environment, actor, ckpt


def _record_one(args, step, ckpt_path, output_dir, label, hunt_success):
  from contrastive import eval_video

  environment, actor, ckpt = _load_actor(
      ckpt_path, args.env_name, args.seed, args)
  env_steps = int(ckpt.get('env_steps', step))
  if hunt_success:
    frames, episode_return, success, attempts = eval_video.record_until_success(
        environment, actor, max_episodes=args.hunt_episodes)
    kind = 'first_success' if success >= 1.0 else 'first_success_miss'
  else:
    frames, episode_return, success = eval_video.record_episode_frames(
        environment, actor)
    attempts = 1
    kind = 'paced'

  task = args.env_name.replace('sawyer_', '')
  stem = f'{task}_{label}_s{args.seed}_step{env_steps}_{kind}'
  gif_path = output_dir / f'{stem}.gif'
  eval_video.save_gif(
      frames, gif_path, fps=args.fps, max_side=args.max_side,
      max_frames=args.max_frames)
  meta = {
      'attempts': int(attempts),
      'checkpoint': str(ckpt_path),
      'env_name': args.env_name,
      'env_steps': env_steps,
      'episode_return': float(episode_return),
      'gif': str(gif_path),
      'kind': kind,
      'label': label,
      'seed': int(args.seed),
      'success': float(success),
  }
  json_path = output_dir / f'{stem}.json'
  json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n',
                       encoding='utf-8')
  print(
      f'Wrote {gif_path} success={success:.0f} return={episode_return:.1f} '
      f'attempts={attempts}',
      flush=True)
  return meta


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint-dir', required=True)
  parser.add_argument('--env-name', required=True, choices=sorted(FIXED_GOALS))
  parser.add_argument('--seed', type=int, default=6)
  parser.add_argument('--label', default='dcc')
  parser.add_argument('--even', type=int, default=10)
  parser.add_argument('--first-success-step', type=int, default=0,
                      help='Also hunt a successful eval episode near this step. '
                           '0 disables the extra GIF.')
  parser.add_argument('--hunt-episodes', type=int, default=20)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--network-width', type=int, default=1024)
  parser.add_argument('--critic-depth', type=int, default=4)
  parser.add_argument('--actor-depth', type=int, default=4)
  parser.add_argument('--fps', type=int, default=10)
  parser.add_argument('--max-side', type=int, default=320)
  parser.add_argument('--max-frames', type=int, default=80)
  args = parser.parse_args()

  ckpt_dir = Path(args.checkpoint_dir).expanduser().resolve()
  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  ckpts = list_step_ckpts(ckpt_dir, args.env_name, args.seed)
  if not ckpts:
    print(f'No mid-task checkpoints for {args.env_name} seed={args.seed} '
          f'under {ckpt_dir}', file=sys.stderr)
    return 1

  chosen = pick_targets(ckpts, even_target_tokens(args.even))
  records = []
  for step, path in chosen:
    records.append(_record_one(
        args, step, path, output_dir, args.label, hunt_success=False))

  if args.first_success_step > 0:
    first = pick_targets(ckpts, [str(args.first_success_step)])
    if first:
      step, path = first[0]
      records.append(_record_one(
          args, step, path, output_dir, args.label, hunt_success=True))

  manifest = {
      'checkpoint_dir': str(ckpt_dir),
      'env_name': args.env_name,
      'label': args.label,
      'n_gifs': len(records),
      'records': records,
      'seed': args.seed,
  }
  manifest_path = output_dir / 'manifest.json'
  manifest_path.write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  print(f'Wrote {manifest_path} ({len(records)} gifs)', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
