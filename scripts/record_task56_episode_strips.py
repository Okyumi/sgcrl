#!/usr/bin/env python3
"""Record one Task-5 and one Task-6 eval episode for the appendix filmstrip.

Handle uses the 150.3k first-success snapshot (100.2k eval was 0).
Push uses the 250.5k probe checkpoint that matches the appendix table.
Hunts a successful episode when possible so the latch vs path geometry
is visible. Writes NPZ + the composed PDF/PNG.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.episode_strip import compose_strips, pack_episode
from contrastive.feature_shortcut import env_config_dir_matches
from scripts.record_checkpoint_rollout_gifs import _load_actor

DEFAULT_CKPT_ROOT = (
    REPO_ROOT / 'logs' / 'jubail_task5_action_advice' / 'checkpoints')
GIF_ROOT = REPO_ROOT / 'logs' / 'jubail_task_videos' / 'gifs'
SPECS = (
    {
        'env_name': 'sawyer_handle_press_side',
        'step': 150300,
        'hunt': True,
        'tag': 'handle',
        'gif': GIF_ROOT / 'handle_nobc_gifs'
        / 'handle_press_side_dcc_nobc_s6_step150300_first_success.gif',
    },
    {
        'env_name': 'sawyer_push',
        'step': 200400,
        'hunt': True,
        'tag': 'push',
        'gif': GIF_ROOT / 'push_gifs' / 'push_dcc_s6_step200400_paced.gif',
    },
)


def _ckpt_path(root: Path, env_name: str, seed: int, step: int) -> Path:
  matches = []
  for path in Path(root).rglob(f'task_0_step_{int(step)}.pkl'):
    if path.parent.name != f'seed_{int(seed)}':
      continue
    if env_config_dir_matches(path, env_name):
      matches.append(path)
  if not matches:
    raise FileNotFoundError(
        f'no task_0_step_{step}.pkl for {env_name} seed {seed} under {root}')
  matches.sort(key=lambda p: len(p.parts))
  return matches[0]


def _load_gif_frames(path: Path) -> np.ndarray:
  from PIL import Image, ImageSequence
  image = Image.open(path)
  frames = [np.asarray(frame.convert('RGB')) for frame in ImageSequence.Iterator(image)]
  if not frames:
    raise ValueError(f'empty GIF {path}')
  return np.stack(frames, axis=0)


def expand_gif_to_horizon(gif_frames: np.ndarray, n_steps: int) -> np.ndarray:
  """Nearest-neighbour expand a subsampled GIF onto the full episode length."""
  gif_frames = np.asarray(gif_frames)
  n_steps = int(n_steps)
  n_gif = int(gif_frames.shape[0])
  src = np.linspace(0, n_steps - 1, n_gif).round().astype(np.int32)
  out = np.empty((n_steps,) + gif_frames.shape[1:], dtype=gif_frames.dtype)
  ptr = 0
  for t in range(n_steps):
    while (ptr + 1 < n_gif
           and abs(int(src[ptr + 1]) - t) <= abs(int(src[ptr]) - t)):
      ptr += 1
    out[t] = gif_frames[ptr]
  return out


def _record_with_states(environment, actor, obs_dim: int, hunt: bool,
                        max_episodes: int, render: bool):
  from contrastive import eval_video

  best = None
  n_try = max_episodes if hunt else 1
  for attempt in range(1, n_try + 1):
    frames = []
    states = []
    episode_return = 0.0
    timestep = environment.reset()
    obs = np.asarray(timestep.observation, dtype=np.float32).reshape(-1)
    states.append(obs[:obs_dim].copy())
    if render:
      frames.append(eval_video.render_rgb_array(environment))
    while not timestep.last():
      action = actor.select_action(timestep.observation)
      timestep = environment.step(action)
      episode_return += float(timestep.reward)
      obs = np.asarray(timestep.observation, dtype=np.float32).reshape(-1)
      states.append(obs[:obs_dim].copy())
      if render:
        frames.append(eval_video.render_rgb_array(environment))
    success = float(episode_return >= 1.0)
    frame_arr = np.stack(frames, axis=0) if render else None
    packed = (frame_arr, np.stack(states, axis=0), episode_return, success, attempt)
    if success >= 1.0:
      return packed
    best = packed
  return best


class _Args:
  def __init__(self, ns):
    self.network_width = ns.network_width
    self.critic_depth = ns.critic_depth
    self.actor_depth = ns.actor_depth
    self.sawyer_success_mode = ns.sawyer_success_mode
    self.use_task_id = False
    self.task_id = 0
    self.num_tasks = 1
    self.seed = ns.seed
    self.hunt_episodes = ns.hunt_episodes
    self.checkpoint_root = ns.checkpoint_root
    self.from_gifs = ns.from_gifs


def record_one(args, spec) -> dict:
  step = int(spec['step'])
  ckpt = _ckpt_path(
      Path(args.checkpoint_root), spec['env_name'], args.seed, step)
  environment, actor, blob = _load_actor(
      ckpt, spec['env_name'], args.seed, args)
  obs_spec = environment.observation_spec().shape[0]
  obs_dim = int(blob.get('obs_dim') or obs_spec // 2)
  frames, states, episode_return, success, attempts = _record_with_states(
      environment, actor, obs_dim, spec['hunt'], args.hunt_episodes,
      render=not args.from_gifs)
  if args.from_gifs:
    gif_path = Path(spec['gif'])
    if not gif_path.is_file():
      raise FileNotFoundError(gif_path)
    frames = expand_gif_to_horizon(_load_gif_frames(gif_path), states.shape[0])
  rec = pack_episode(
      frames, states, spec['env_name'],
      env_steps=int(blob.get('env_steps', step)),
      success=success,
      attempts=attempts,
      checkpoint=str(ckpt))
  rec['episode_return'] = np.array([float(episode_return)])
  rec['frames_from_gif'] = np.array([int(args.from_gifs)])
  return rec


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint-root', default=str(DEFAULT_CKPT_ROOT))
  parser.add_argument('--seed', type=int, default=6)
  parser.add_argument('--hunt-episodes', type=int, default=20)
  parser.add_argument('--network-width', type=int, default=1024)
  parser.add_argument('--critic-depth', type=int, default=4)
  parser.add_argument('--actor-depth', type=int, default=4)
  parser.add_argument('--sawyer-success-mode', default='corrected')
  parser.add_argument(
      '--out-dir',
      default=str(REPO_ROOT / 'results' / 'data' / 'task56_episode_strips'))
  parser.add_argument(
      '--fig-dir',
      default=str(REPO_ROOT / 'results' / 'img' / 'paper'))
  parser.add_argument('--npz-only', action='store_true')
  parser.add_argument(
      '--from-gifs', action='store_true',
      help='Skip EGL render; attach audited GIFs to a states-only re-roll.')
  args = parser.parse_args()
  bundle = _Args(args)

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  recorded = {}
  summary = []
  for spec in SPECS:
    rec = record_one(bundle, spec)
    recorded[spec['tag']] = rec
    npz_path = out_dir / f"{spec['tag']}_strip.npz"
    np.savez_compressed(
        npz_path,
        **{k: rec[k] for k in rec if k != 'phases'},
        phases=np.asarray(rec['phases'], dtype='U24'))
    meta = {
        'env_name': spec['env_name'],
        'env_steps': int(rec['env_steps'][0]),
        'success': float(rec['success'][0]),
        'episode_return': float(rec['episode_return'][0]),
        'attempts': int(rec['attempts'][0]),
        'n_frames': int(rec['frames'].shape[0]),
        'npz': str(npz_path),
        'checkpoint': str(rec['checkpoint']),
    }
    summary.append(meta)
    print(json.dumps(meta), flush=True)

  (out_dir / 'manifest.json').write_text(
      json.dumps(summary, indent=2) + '\n', encoding='utf-8')
  if not args.npz_only:
    fig_dir = Path(args.fig_dir)
    compose_strips(
        recorded['handle'], recorded['push'],
        fig_dir / 'fig_task56_episode_strip.pdf',
        fig_dir / 'fig_task56_episode_strip.png')
    print('Wrote filmstrip figure', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
