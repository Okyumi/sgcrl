"""Feature groups and metrics for contrastive critic shortcut probes.

The critic score is s(s,a,g)=φ(s,a)ᵀψ(g). InfoNCE can be solved by a
coordinate that is sufficient for retrieving the matching future among
in-batch negatives, even if that coordinate is useless for ranking actions.

Unified Sawyer layout (``env_utils``):
  0:3  hand xyz
  3:4  gripper opening
  4:7  mechanism / object xyz   (handle on Task 5, cube on push)

Task-5 success is a 2 cm band on handle z (index 6). Push success is a 5 cm
ball on cube xyz, which in this table layout is almost entirely xy.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


HAND = slice(0, 3)
GRIPPER = slice(3, 4)
MECH_XY = slice(4, 6)
MECH_Z = slice(6, 7)
MECH = slice(4, 7)

FEATURE_BLOCKS: Dict[str, slice] = {
    'hand': HAND,
    'gripper': GRIPPER,
    'mech_xy': MECH_XY,
    'mech_z': MECH_Z,
    'mech': MECH,
}

HANDLE_Z_TARGET = 0.07
HANDLE_Z_THRESHOLD = 0.02
PUSH_TARGET = np.array([0.02, 0.89, 0.02], dtype=np.float32)
PUSH_SUCCESS_RADIUS = 0.05
PUSH_PROGRESS_RADIUS = 0.15

FIXED_GOALS = {
    'sawyer_handle_press_side': np.array([-0.07, 0.68, 0.07], dtype=np.float32),
    'sawyer_push': PUSH_TARGET.copy(),
}


def classify_state(state: np.ndarray, env_name: str) -> str:
  """Far / near / hover / progress / success using the hover-advice geometry."""
  state = np.asarray(state, dtype=np.float32).reshape(-1)
  hand = state[:3]
  obj = state[4:7]
  hand_obj = float(np.linalg.norm(hand - obj))
  if env_name == 'sawyer_handle_press_side':
    axis_ok = abs(float(state[6]) - HANDLE_Z_TARGET) <= HANDLE_Z_THRESHOLD
    if axis_ok and hand_obj <= 0.09:
      return 'success'
    if hand_obj <= 0.09:
      return 'hover_contact'
    if hand_obj <= 0.15:
      return 'near_approach'
    return 'far'
  obj_dist = float(np.linalg.norm(obj - PUSH_TARGET))
  if obj_dist <= PUSH_SUCCESS_RADIUS:
    return 'success'
  if obj_dist <= PUSH_PROGRESS_RADIUS:
    return 'object_progress'
  if hand_obj <= 0.09:
    return 'hover_contact'
  if hand_obj <= 0.15:
    return 'near_approach'
  return 'far'


def biased_action(pi_action: np.ndarray, env_name: str) -> np.ndarray:
  a = np.array(pi_action, dtype=np.float32, copy=True)
  if env_name == 'sawyer_handle_press_side':
    a[2] = -1.0
  else:
    a[1] = 1.0
  return np.clip(a, -1.0, 1.0)


def transplant_indices(
    dst: np.ndarray, src: np.ndarray, slc: slice) -> np.ndarray:
  out = np.array(dst, dtype=np.float32, copy=True)
  src = np.asarray(src, dtype=np.float32).reshape(-1)
  out[slc] = src[slc]
  return out


def synthetic_progress_state(state: np.ndarray, env_name: str) -> np.ndarray:
  """Copy ``state`` and write the task-progress coordinate to its success value.

  Handle: only index 6 (handle z) is set to 0.07.
  Push: object xyz is set to the table target. Hand / gripper / action are
  left unchanged, so this is a pure object-pose intervention.
  """
  out = np.array(state, dtype=np.float32, copy=True)
  if env_name == 'sawyer_handle_press_side':
    out[6] = HANDLE_Z_TARGET
  else:
    out[4:7] = PUSH_TARGET
  return out


def recovery_fraction(
    base: float, treated: float, full: float, eps: float = 1e-6) -> float:
  """How much of the base→full gap a treatment recovers."""
  denom = float(full) - float(base)
  if abs(denom) < eps:
    return float('nan')
  return (float(treated) - float(base)) / denom


def pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
  x = np.asarray(x, dtype=np.float64).reshape(-1)
  y = np.asarray(y, dtype=np.float64).reshape(-1)
  if x.size < 3 or y.size < 3:
    return float('nan')
  x = x - np.mean(x)
  y = y - np.mean(y)
  denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
  if denom < eps:
    return float('nan')
  return float(np.sum(x * y) / denom)


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
  """Ordinary-least-squares residual of y after a linear fit on x (with bias)."""
  y = np.asarray(y, dtype=np.float64).reshape(-1)
  x = np.asarray(x, dtype=np.float64).reshape(-1)
  if y.size < 3:
    return y.copy()
  a = np.stack([np.ones_like(x), x], axis=1)
  coef, *_ = np.linalg.lstsq(a, y, rcond=None)
  return y - a.dot(coef)


def categorical_accuracy(logits: np.ndarray) -> float:
  logits = np.asarray(logits)
  pred = np.argmax(logits, axis=1)
  labels = np.arange(logits.shape[0])
  return float(np.mean(pred == labels))


def shuffle_block(
    values: np.ndarray, slc: slice, rng: np.random.Generator) -> np.ndarray:
  out = np.array(values, dtype=np.float32, copy=True)
  perm = rng.permutation(out.shape[0])
  out[:, slc] = out[perm][:, slc]
  return out


def shuffle_rows(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
  out = np.array(values, dtype=np.float32, copy=True)
  return out[rng.permutation(out.shape[0])]


def occupancy_stats(
    values: np.ndarray, low: float, high: float) -> Dict[str, float]:
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  if values.size == 0:
    return {
        'n': 0.0,
        'mean': float('nan'),
        'std': float('nan'),
        'frac_in_band': float('nan'),
        'bernoulli_entropy': float('nan'),
    }
  frac = float(np.mean((values >= low) & (values <= high)))
  p = min(max(frac, 1e-8), 1.0 - 1e-8)
  entropy = float(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))
  return {
      'n': float(values.size),
      'mean': float(np.mean(values)),
      'std': float(np.std(values)),
      'frac_in_band': frac,
      'bernoulli_entropy': entropy,
  }


def mechanism_density(states: np.ndarray, env_name: str) -> Dict[str, float]:
  """How peaked the task-progress coordinate is in collected states."""
  states = np.asarray(states, dtype=np.float32)
  if states.size == 0:
    return occupancy_stats(np.array([]), 0.0, 1.0)
  if env_name == 'sawyer_handle_press_side':
    z = states[:, 6]
    stats = occupancy_stats(
        z, HANDLE_Z_TARGET - HANDLE_Z_THRESHOLD,
        HANDLE_Z_TARGET + HANDLE_Z_THRESHOLD)
    stats['coord'] = 6.0
    stats['xy_std'] = float(np.std(states[:, 4:6]))
    stats['z_std'] = float(np.std(z))
    return stats
  dist = np.linalg.norm(states[:, 4:7] - PUSH_TARGET[None, :], axis=-1)
  stats = occupancy_stats(dist, 0.0, PUSH_SUCCESS_RADIUS)
  stats['coord'] = -1.0
  stats['xy_std'] = float(np.std(states[:, 4:6]))
  stats['z_std'] = float(np.std(states[:, 6]))
  stats['frac_in_progress_band'] = float(np.mean(
      (dist > PUSH_SUCCESS_RADIUS) & (dist <= PUSH_PROGRESS_RADIUS)))
  return stats


def make_her_pairs(
    episodes: Sequence[Mapping[str, np.ndarray]],
    rng: np.random.Generator,
    *,
    max_pairs: int = 256,
    discount: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Sample (s_t, a_t, s_{t+k}) pairs with discounted-future k.

  Returns state, action, future_state arrays of shape (N, ·).
  """
  pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
  for episode in episodes:
    states = np.asarray(episode['states'], dtype=np.float32)
    actions = np.asarray(episode['actions'], dtype=np.float32)
    t_len = int(states.shape[0])
    if t_len < 2:
      continue
    for t in range(t_len - 1):
      remaining = np.arange(t + 1, t_len)
      weights = np.power(discount, remaining - t - 1)
      weights = weights / np.sum(weights)
      k = int(rng.choice(remaining, p=weights))
      pairs.append((states[t], actions[t], states[k]))
  if not pairs:
    empty = np.zeros((0, 11), dtype=np.float32)
    return empty, np.zeros((0, 4), dtype=np.float32), empty
  rng.shuffle(pairs)
  pairs = pairs[: int(max_pairs)]
  state = np.stack([p[0] for p in pairs], axis=0)
  action = np.stack([p[1] for p in pairs], axis=0)
  future = np.stack([p[2] for p in pairs], axis=0)
  return state, action, future


def summarize(values: Iterable[float]) -> Dict[str, float]:
  arr = np.asarray(list(values), dtype=np.float64)
  if arr.size == 0:
    return {'n': 0.0, 'mean': float('nan'), 'std': float('nan')}
  return {
      'n': float(arr.size),
      'mean': float(np.mean(arr)),
      'std': float(np.std(arr)),
      'median': float(np.median(arr)),
  }


def env_config_dir_matches(path, env_name: str) -> bool:
  """True when a checkpoint lives in the exact ``_env_{name}`` config dir.

  The inject cell appends ``_ret_<hash>``, which is a prefix match on the
  env token but not an exact config-dir match.
  """
  marker = f'_env_{env_name}'
  current = getattr(path, 'parents', None)
  if current is None:
    return False
  for parent in path.parents:
    if parent.name.startswith('seed_'):
      return parent.parent.name.endswith(marker)
  return False
