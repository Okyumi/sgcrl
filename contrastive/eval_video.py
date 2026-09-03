"""Record deterministic evaluation rollouts as RGB frame stacks for W&B."""
from __future__ import annotations

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


def record_episode_frames(
    environment: dm_env.Environment,
    actor,
) -> Tuple[np.ndarray, float, float]:
  """Run one deterministic eval episode and return (T,H,W,C) uint8 frames."""
  frames = []
  episode_return = 0.0
  timestep = environment.reset()
  actor.observe_first(timestep)
  frames.append(render_rgb_array(environment))

  while not timestep.last():
    action = actor.select_action(timestep.observation)
    timestep = environment.step(action)
    actor.observe(action, next_timestep=timestep)
    episode_return += float(timestep.reward)
    frames.append(render_rgb_array(environment))

  success = float(episode_return >= 1.0)
  return np.stack(frames, axis=0), episode_return, success
