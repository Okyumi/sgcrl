#!/usr/bin/env python3
"""Render stored D_succ (s,a) rings as GIFs.

The Task-7 policy GIF canvas labeled r0–r9 is independent eval rollouts
(seeds 1006–1015). Those are not the success buffer. Mid-task checkpoints
store the ring in decomposed_training_state. This script unwraps it in
FIFO order, splits contiguous object-space runs, and plays each run back
by writing hand/object poses into SawyerShelfPlace (no policy).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

SHELF_TARGET = np.array([0.02, 0.89, 0.30], dtype=np.float32)
SUCCESS_RADIUS = 0.07
OBS_DIM = 11

TERM_DIR = Path(
    '/scratch/yd2247/sgcrl/logs/terminal_bc_task47/checkpoints/'
    'actor_reset_critic_decomposed_tid_False_heads_True'
    '_success_native_info_dyn0.000_pt256x4'
    '_bridge_4c3aef3792be_ret_7d91520b6c16')
WARM_DIR = Path(
    '/scratch/yd2247/sgcrl/logs/success_bc_warmup/checkpoints/'
    'actor_reset_critic_decomposed_tid_False_heads_True'
    '_success_native_info_dyn0.000_pt256x4'
    '_bridge_52d4c4756d98_ret_a837a9cd0970')
WINDOW_DIR = Path(
    '/scratch/yd2247/sgcrl/logs/first_success_window_bc/checkpoints/'
    'actor_reset_critic_decomposed_tid_False_heads_True'
    '_success_native_info_dyn0.000_pt256x4'
    '_bridge_a3185f93c3be_ret_dc89ef3e83da')


def unwrap_ring(obs, size, index):
  """Oldest-to-newest filled slots of a FIFO ring."""
  obs = np.asarray(obs)
  size = int(size)
  index = int(index)
  capacity = int(obs.shape[0])
  if size <= 0:
    return obs[:0]
  if size < capacity:
    return obs[:size]
  return np.concatenate([obs[index:], obs[:index]], axis=0)


def object_xyz(obs):
  obs = np.asarray(obs)
  if obs.ndim == 1:
    obs = obs[None, :]
  return obs[:, 4:7]


def stored_goal_xyz(obs):
  obs = np.asarray(obs)
  if obs.ndim == 1:
    obs = obs[None, :]
  return obs[:, OBS_DIM + 4:OBS_DIM + 7]


def split_episodes(obs, episode_len=150):
  """FIFO chunks of one MetaWorld episode (150 env steps)."""
  obs = np.asarray(obs)
  episode_len = int(episode_len)
  if obs.shape[0] == 0 or episode_len <= 0:
    return []
  return [
      obs[start:start + episode_len]
      for start in range(0, obs.shape[0], episode_len)
  ]


def split_contiguous(obs, jump=0.12, min_len=20):
  """Split a ring dump wherever the object teleports."""
  obs = np.asarray(obs)
  if obs.shape[0] == 0:
    return []
  obj = object_xyz(obs)
  if obj.shape[0] == 1:
    return [obs]
  jumps = np.linalg.norm(np.diff(obj, axis=0), axis=1)
  cuts = np.flatnonzero(jumps > float(jump)) + 1
  bounds = np.concatenate([[0], cuts, [obs.shape[0]]])
  segments = []
  for start, end in zip(bounds[:-1], bounds[1:]):
    if int(end - start) >= int(min_len):
      segments.append(obs[int(start):int(end)])
  return segments


def segment_stats(obs):
  obj = object_xyz(obs)
  goal = stored_goal_xyz(obs)
  if goal.shape[0] and np.linalg.norm(goal[0]) > 1e-6:
    target = goal
  else:
    target = np.broadcast_to(SHELF_TARGET[None, :], obj.shape)
  dist = np.linalg.norm(obj - target, axis=1)
  in_goal = dist <= SUCCESS_RADIUS
  return {
      'n': int(obs.shape[0]),
      'frac_in_goal': float(np.mean(in_goal)),
      'any_in_goal': bool(np.any(in_goal)),
      'last_in_goal': bool(in_goal[-1]),
      'last_dist': float(dist[-1]),
      'min_dist': float(np.min(dist)),
      'first_in_goal_t':
          int(np.flatnonzero(in_goal)[0]) if np.any(in_goal) else -1,
  }


def first_nonempty_mid(seed_dir: Path, task_id: int = 7):
  steps = []
  for path in Path(seed_dir).glob(f'task_{task_id}_step_*.pkl'):
    step = int(path.name.split('_')[-1].split('.')[0])
    steps.append((step, path))
  steps.sort()
  for step, path in steps:
    with path.open('rb') as handle:
      ckpt = pickle.load(handle)
    state = ckpt.get('decomposed_training_state')
    if state is None:
      continue
    size = int(np.asarray(state.success_buffer_size))
    if size > 0:
      return step, path, size
  return None


def load_ring(path: Path):
  with path.open('rb') as handle:
    ckpt = pickle.load(handle)
  state = ckpt.get('decomposed_training_state')
  if state is None:
    raise KeyError(f'{path} has no decomposed_training_state / D_succ')
  size = int(np.asarray(state.success_buffer_size))
  index = int(np.asarray(state.success_buffer_index))
  obs = np.asarray(state.success_buffer_observation)
  seq = unwrap_ring(obs, size, index)
  return ckpt, seq, size, index


def apply_obs(mujoco_env, obs):  # noqa: D401
  obs = np.asarray(obs, dtype=np.float32).reshape(-1)
  hand = obs[0:3]
  obj = obs[4:7]
  goal = obs[OBS_DIM + 4:OBS_DIM + 7]
  if np.linalg.norm(goal) > 1e-6:
    mujoco_env._target_pos = goal.astype(np.float32)
    if hasattr(mujoco_env, '_goal'):
      mujoco_env._goal = goal.astype(np.float32)
  mujoco_env.data.set_mocap_pos('mocap', hand)
  mujoco_env.data.set_mocap_quat('mocap', np.array([1.0, 0.0, 1.0, 0.0]))
  mujoco_env._set_obj_xyz(obj)
  if hasattr(mujoco_env, 'sim'):
    mujoco_env.sim.forward()


def render_stills(dm_env, mujoco_env, seq, n_each=8):
  from contrastive.eval_video import render_rgb_array
  goal = stored_goal_xyz(seq)
  if np.linalg.norm(goal) < 1e-6:
    goal = np.broadcast_to(SHELF_TARGET[None, :], (seq.shape[0], 3))
  dist = np.linalg.norm(object_xyz(seq) - goal, axis=1)
  in_goal = np.flatnonzero(dist <= SUCCESS_RADIUS)
  far = np.flatnonzero(dist > 0.15)
  picked = []
  for name, idxs in (('in_goal', in_goal), ('far', far)):
    if idxs.size == 0:
      continue
    take = np.linspace(0, idxs.size - 1, num=min(n_each, idxs.size)).round().astype(int)
    for j in take:
      t = int(idxs[j])
      apply_obs(mujoco_env, seq[t])
      picked.append((name, t, float(dist[t]), render_rgb_array(dm_env)))
  return picked


def render_segment(dm_env, mujoco_env, obs_seq, stride=2):
  from contrastive.eval_video import render_rgb_array
  frames = []
  idx = list(range(0, obs_seq.shape[0], max(int(stride), 1)))
  if idx[-1] != obs_seq.shape[0] - 1:
    idx.append(obs_seq.shape[0] - 1)
  for t in idx:
    apply_obs(mujoco_env, obs_seq[t])
    frames.append(render_rgb_array(dm_env))
  return np.stack(frames, axis=0)


def path_frames(obs_seq, size=200):
  """Tiny 3D GIF frames of stored hand/object vs the shelf target."""
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

  obj = object_xyz(obs_seq)
  hand = np.asarray(obs_seq)[:, 0:3]
  goal = stored_goal_xyz(obs_seq)
  if np.linalg.norm(goal[0]) < 1e-6:
    goal = np.broadcast_to(SHELF_TARGET[None, :], obj.shape)
  target = goal[0]
  n = obj.shape[0]
  frames = []
  for t in range(n):
    fig = plt.figure(figsize=(size / 100.0, size / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(obj[:t + 1, 0], obj[:t + 1, 1], obj[:t + 1, 2], color='#d18616', lw=1.4)
    ax.scatter(obj[t, 0], obj[t, 1], obj[t, 2], c='#d18616', s=18)
    ax.plot(hand[:t + 1, 0], hand[:t + 1, 1], hand[:t + 1, 2], color='#6f9cdb', lw=1.0, ls='--')
    ax.scatter(target[0], target[1], target[2], c='#3fa66e', s=36, marker='*')
    ax.set_xlim(-0.2, 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_zlim(0.0, 0.45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f't={t + 1}/{n}')
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    frames.append(np.ascontiguousarray(buf))
    plt.close(fig)
  return np.stack(frames, axis=0)


def to_webp(gif_path: Path, webp_path: Path, quality=40):
  from PIL import Image
  src = Image.open(gif_path)
  frames = []
  try:
    while True:
      frames.append(src.convert('RGB'))
      src.seek(src.tell() + 1)
  except EOFError:
    pass
  if not frames:
    return
  duration = src.info.get('duration', 100)
  frames[0].save(
      webp_path,
      save_all=True,
      append_images=frames[1:],
      duration=duration,
      loop=0,
      quality=int(quality),
      method=6,
  )


def save_path_gif(obs_seq, gif_path, webp_path=None, fps=6):
  from contrastive.eval_video import save_gif
  frames = path_frames(obs_seq)
  save_gif(frames, gif_path, fps=fps, max_side=220, max_frames=150)
  if webp_path is not None:
    to_webp(Path(gif_path), Path(webp_path), quality=45)


def default_snapshots():
  items = [
      {
          'name': 'terminal_s6_first',
          'label': 'terminal-episode',
          'path': TERM_DIR / 'seed_6' / 'task_7_step_1500750.pkl',
      },
      {
          'name': 'warmup_s6_first',
          'label': 'warmup',
          'path': WARM_DIR / 'seed_6' / 'task_7_step_2301150.pkl',
      },
      {
          'name': 'warmup_s5_late',
          'label': 'warmup',
          'path': WARM_DIR / 'seed_5' / 'task_7_step_7903950.pkl',
      },
      {
          'name': 'terminal_s6_late',
          'label': 'terminal-episode',
          'path': TERM_DIR / 'seed_6' / 'task_7_step_7903950.pkl',
      },
      {
          'name': 'window_s6_first',
          'label': 'first-success-window',
          'path': WINDOW_DIR / 'seed_6',
          'first_nonempty': True,
      },
  ]
  return items


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--jump', type=float, default=0.12)
  parser.add_argument('--min-len', type=int, default=20)
  parser.add_argument('--max-segments', type=int, default=40,
                      help='Cap GIFs per snapshot (longest kept). 0 = all.')
  parser.add_argument('--stride', type=int, default=2)
  parser.add_argument('--fps', type=int, default=8)
  parser.add_argument('--max-side', type=int, default=240)
  parser.add_argument('--max-frames', type=int, default=150)
  parser.add_argument('--episode-len', type=int, default=150,
                      help='Chunk D_succ into this many steps (0 = teleport split).')
  parser.add_argument('--webp', action='store_true')
  parser.add_argument('--paths-only', action='store_true',
                      help='3D object/hand paths, no MuJoCo EGL.')
  parser.add_argument('--snapshot', action='append', default=[],
                      help='Restrict to snapshot names.')
  args = parser.parse_args()

  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  snapshots = default_snapshots()
  if args.snapshot:
    keep = set(args.snapshot)
    snapshots = [item for item in snapshots if item['name'] in keep]

  environment = None
  mujoco_env = None
  if not args.paths_only:
    from contrastive.eval_video import _get_base_mujoco_env, save_gif
    from contrastive.utils import make_environment

    environment, _obs_dim = make_environment(
        'sawyer_shelf_place', 0, -1, 6,
        fixed_start_end=SHELF_TARGET,
        sawyer_success_mode='native_info')
    environment.reset()
    mujoco_env = _get_base_mujoco_env(environment)
    if mujoco_env is None:
      raise RuntimeError('Could not unwrap SawyerShelfPlace to a MuJoCo env')

  manifest = []
  for item in snapshots:
    path = Path(item['path'])
    if item.get('first_nonempty'):
      found = first_nonempty_mid(path)
      if found is None:
        print(f'SKIP empty ring under {path}', flush=True)
        continue
      path = Path(found[1])
    if not path.is_file():
      print(f'SKIP missing {path}', flush=True)
      continue
    print(f'Loading {item["name"]} {path}', flush=True)
    ckpt, seq, size, index = load_ring(path)
    if args.episode_len > 0:
      segments = split_episodes(seq, episode_len=args.episode_len)
      order = list(range(len(segments)))
    else:
      segments = split_contiguous(seq, jump=args.jump, min_len=args.min_len)
      order = sorted(range(len(segments)),
                     key=lambda i: segments[i].shape[0], reverse=True)
      if args.max_segments > 0:
        order = order[:args.max_segments]
      order = sorted(order)
    snap_dir = output_dir / item['name']
    if snap_dir.exists():
      for old in snap_dir.glob('*'):
        old.unlink()
    snap_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for local_i, seg_i in enumerate(order):
      obs_seq = segments[seg_i]
      stats = segment_stats(obs_seq)
      stem = f'{item["name"]}_traj{local_i:02d}_n{stats["n"]}'
      gif_path = snap_dir / f'{stem}.gif'
      webp_path = snap_dir / f'{stem}.webp' if args.webp else None
      if args.paths_only:
        save_path_gif(obs_seq, gif_path, webp_path, fps=10)
      else:
        frames = render_segment(
            environment, mujoco_env, obs_seq, stride=args.stride)
        save_gif(
            frames, gif_path, fps=args.fps,
            max_side=args.max_side, max_frames=args.max_frames)
        if webp_path is not None:
          to_webp(gif_path, webp_path)
      rec = {
          **stats,
          'gif': str(gif_path),
          'webp': str(webp_path) if webp_path else None,
          'label': item['label'],
          'snapshot': item['name'],
          'traj_id': local_i,
          'ok': bool(stats['last_in_goal']),
      }
      records.append(rec)
      print(
          f'  traj {local_i:02d} n={stats["n"]} last_dist={stats["last_dist"]:.3f} '
          f'last_in_goal={stats["last_in_goal"]} any={stats["any_in_goal"]} '
          f'-> {gif_path.name}',
          flush=True)
    stills = []
    if not args.paths_only:
      from PIL import Image
      for name, t, dist, frame in render_stills(environment, mujoco_env, seq):
        still_path = snap_dir / f'{item["name"]}_{name}_t{t:04d}.png'
        Image.fromarray(frame).resize((240, 180)).save(still_path)
        stills.append({
            'kind': name, 't': t, 'dist': dist, 'png': str(still_path),
        })
    payload = {
        'checkpoint': str(path),
        'env_steps': ckpt.get('env_steps'),
        'index': index,
        'label': item['label'],
        'n_segments': len(segments),
        'n_gifs': len(records),
        'name': item['name'],
        'records': records,
        'ring_size': size,
        'seq_len': int(seq.shape[0]),
        'stills': stills,
    }
    (snap_dir / 'manifest.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n')
    manifest.append(payload)

  out = output_dir / 'manifest.json'
  out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
  print(f'Wrote {out} snapshots={len(manifest)}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
