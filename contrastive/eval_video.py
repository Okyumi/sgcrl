"""Record deterministic evaluation rollouts as RGB frames, GIFs, or W&B video."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import dm_env
import numpy as np


def _get_base_mujoco_env(env) -> object:
  """Unwrap dm_env/gym wrappers until we find an env with `.sim`."""
  current = env
  seen = set()
  while current is not None and id(current) not in seen:
    seen.add(id(current))
    if hasattr(current, 'sim'):
      return current
    current = (
        getattr(current, '_environment', None)
        or getattr(current, 'env', None)
        or getattr(current, 'environment', None))
  return None


def render_rgb_array(dm_env_env) -> np.ndarray:
  """Best-effort RGB frame from a dm_env-wrapped gym env."""
  base = _get_base_mujoco_env(dm_env_env)
  if base is not None and hasattr(base, 'sim'):
    try:
      import mujoco_py
      sim = base.sim
      if not hasattr(base, '_sgcrl_offscreen_ctx'):
        base._sgcrl_offscreen_ctx = mujoco_py.MjRenderContextOffscreen(
            sim, device_id=-1)
      ctx = base._sgcrl_offscreen_ctx
      width, height = 640, 480
      ctx.render(width, height)
      frame = ctx.read_pixels(width, height, depth=False)
      if frame is not None and len(frame.shape) == 3:
        return _as_uint8_frame(frame[::-1])
    except Exception:
      pass

  for env in (
      getattr(dm_env_env, 'environment', None),
      getattr(dm_env_env, '_environment', None),
      getattr(dm_env_env, 'gym_env', None),
      dm_env_env,
  ):
    if env is None:
      continue
    render = getattr(env, 'render', None)
    if not callable(render):
      continue
    try:
      frame = render(mode='rgb_array')
    except TypeError:
      frame = render()
    if frame is not None:
      return _as_uint8_frame(frame)

  raise RuntimeError(
      'Could not obtain RGB frame from environment for eval video.')


def _as_uint8_frame(frame: np.ndarray) -> np.ndarray:
  array = np.asarray(frame)
  if array.dtype != np.uint8:
    if array.max() <= 1.0:
      array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
  return array


def _maybe_observe_first(actor, timestep) -> None:
  observe_first = getattr(actor, 'observe_first', None)
  if callable(observe_first):
    observe_first(timestep)


def _maybe_observe(actor, action, timestep) -> None:
  observe = getattr(actor, 'observe', None)
  if callable(observe):
    observe(action, next_timestep=timestep)


def record_episode_frames(
    environment: dm_env.Environment,
    actor,
) -> Tuple[np.ndarray, float, float]:
  """Run one deterministic eval episode and return (T,H,W,C) uint8 frames."""
  frames = []
  episode_return = 0.0
  timestep = environment.reset()
  _maybe_observe_first(actor, timestep)
  frames.append(render_rgb_array(environment))

  while not timestep.last():
    action = actor.select_action(timestep.observation)
    timestep = environment.step(action)
    _maybe_observe(actor, action, timestep)
    episode_return += float(timestep.reward)
    frames.append(render_rgb_array(environment))

  success = float(episode_return >= 1.0)
  return np.stack(frames, axis=0), episode_return, success


def record_until_success(
    environment: dm_env.Environment,
    actor,
    *,
    max_episodes: int = 20,
) -> Tuple[np.ndarray, float, float, int]:
  """Roll out up to ``max_episodes`` and keep the first successful one.

  Returns ``(frames, return, success, attempts)``. If no episode succeeds,
  the last attempt is returned with ``success=0``.
  """
  if int(max_episodes) < 1:
    raise ValueError('max_episodes must be >= 1')
  last = None
  for attempt in range(1, int(max_episodes) + 1):
    frames, episode_return, success = record_episode_frames(environment, actor)
    last = (frames, episode_return, success, attempt)
    if success >= 1.0:
      return last
  return last


def downsample_frames(
    frames: np.ndarray,
    *,
    max_side: int = 320,
    max_frames: int = 80,
) -> np.ndarray:
  """Shrink a rollout for a compact GIF (no extra interpolation library)."""
  frames = np.asarray(frames)
  if frames.ndim != 4:
    raise ValueError(f'Expected (T,H,W,C) frames, got {frames.shape}')
  n = frames.shape[0]
  if n > max_frames:
    idx = np.linspace(0, n - 1, num=max_frames).round().astype(int)
    frames = frames[idx]
  height, width = int(frames.shape[1]), int(frames.shape[2])
  scale = min(1.0, float(max_side) / float(max(height, width)))
  if scale < 1.0:
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    ys = np.linspace(0, height - 1, num=new_h).round().astype(int)
    xs = np.linspace(0, width - 1, num=new_w).round().astype(int)
    frames = frames[:, ys][:, :, xs]
  return _as_uint8_frame(frames)


def save_gif(
    frames: np.ndarray,
    path,
    *,
    fps: int = 10,
    max_side: int = 320,
    max_frames: int = 80,
) -> str:
  """Write an animated GIF. Returns the output path as a string."""
  from PIL import Image

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  small = downsample_frames(
      frames, max_side=max_side, max_frames=max_frames)
  images = [Image.fromarray(frame) for frame in small]
  duration_ms = int(round(1000.0 / max(int(fps), 1)))
  images[0].save(
      path,
      save_all=True,
      append_images=images[1:],
      duration=duration_ms,
      loop=0,
      optimize=True,
  )
  return str(path)
