"""Proxies for critic action sensitivity and action-informative replay mass.

This module does **not** estimate the theoretical total-variation term

    D_TV[p(F|s,a), p(F|s,a')].

It computes critic-score ranges over counterfactual actions, survival
curves of those ranges, and (when the caller supplies them) environment
branching ranges of a task coordinate. Callers must keep that distinction
in figure captions and reports.

Continual-sequence indices (``contrastive.continual_config``):
  5  sawyer_handle_press_side
  6  sawyer_push
  8  sawyer_window_close
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from contrastive.her_future_phase import (
    PHASE_HOVER,
    PHASE_PROGRESS,
    PHASE_SUCCESS,
    classify_her_goal,
)


GAUSS_SCALES: Tuple[float, ...] = (0.10, 0.25)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
EPISODE_LEN = 150

TASK_SPECS: Dict[str, dict] = {
    'sawyer_handle_press_side': {
        'index': 5,
        'short': 'Handle press (Task 5)',
        'fixed_goal': np.array([-0.07, 0.68, 0.07], dtype=np.float32),
        'coord_name': 'handle z',
        'coord_index': 6,
        'coord_target': 0.07,
        'bias_dim': 2,
        'bias_value': -1.0,
        'color': '#9d174d',
    },
    'sawyer_window_close': {
        'index': 8,
        'short': 'Window close (Task 8)',
        'fixed_goal': np.array([0.0, 0.80, 0.20], dtype=np.float32),
        'coord_name': 'window y',
        'coord_index': 5,
        'coord_target': 0.80,
        'bias_dim': 1,
        'bias_value': 1.0,
        'color': '#c2410c',
    },
    'sawyer_push': {
        'index': 6,
        'short': 'Push (Task 6)',
        'fixed_goal': np.array([0.02, 0.89, 0.02], dtype=np.float32),
        'coord_name': 'object–target dist',
        'coord_index': None,
        'coord_target': None,
        'bias_dim': 1,
        'bias_value': 1.0,
        'color': '#0f766e',
    },
}


def clip_action(action: np.ndarray) -> np.ndarray:
  return np.clip(np.asarray(action, dtype=np.float32), ACTION_LOW, ACTION_HIGH)


def biased_action(pi_action: np.ndarray, env_name: str) -> np.ndarray:
  spec = TASK_SPECS[env_name]
  a = np.array(pi_action, dtype=np.float32, copy=True)
  a[int(spec['bias_dim'])] = float(spec['bias_value'])
  return clip_action(a)


def task_coordinate(state: np.ndarray, env_name: str) -> float:
  """Scalar overlay used in trajectory plots; not a success predicate."""
  state = np.asarray(state, dtype=np.float32).reshape(-1)
  spec = TASK_SPECS[env_name]
  if env_name == 'sawyer_push':
    obj = state[4:7]
    target = spec['fixed_goal']
    return float(np.linalg.norm(obj - target))
  return float(state[int(spec['coord_index'])])


def contact_flag(state: np.ndarray, env_name: str) -> bool:
  """Hand within the audited 9 cm interaction radius of the mechanism/object."""
  return classify_her_goal(state, env_name) in (
      PHASE_HOVER, PHASE_SUCCESS, PHASE_PROGRESS)


def make_candidate_actions(
    pi_action: np.ndarray,
    replay_actions: np.ndarray,
    rng: np.random.Generator,
    *,
    gauss_scales: Sequence[float] = GAUSS_SCALES,
    n_gauss: int = 8,
    n_shuffle: int = 8,
    n_uniform: int = 8,
    axis_delta: float = 0.50,
) -> Tuple[np.ndarray, np.ndarray]:
  """Counterfactual actions at a fixed state.

  Families: policy, zero, axis ±Δ, Gaussian around π at each scale,
  shuffled replay actions, uniform in [-1, 1].
  """
  pi_action = clip_action(pi_action).reshape(-1)
  action_dim = int(pi_action.shape[0])
  replay_actions = np.asarray(replay_actions, dtype=np.float32)
  if replay_actions.ndim == 1:
    replay_actions = replay_actions.reshape(1, -1)

  rows: List[np.ndarray] = []
  families: List[str] = []

  def _add(action, family: str) -> None:
    rows.append(clip_action(action))
    families.append(family)

  _add(pi_action, 'pi')
  _add(np.zeros(action_dim, dtype=np.float32), 'zero')
  for dim in range(action_dim):
    for sign in (-1.0, 1.0):
      a = pi_action.copy()
      a[dim] = float(np.clip(a[dim] + sign * axis_delta, ACTION_LOW, ACTION_HIGH))
      _add(a, 'axis')
  for scale in gauss_scales:
    noise = rng.normal(0.0, float(scale), size=(int(n_gauss), action_dim))
    for row in pi_action[None, :] + noise.astype(np.float32):
      _add(row, f'gauss_{scale:.2f}')
  n_rep = int(replay_actions.shape[0])
  if n_rep:
    pick = rng.integers(0, n_rep, size=int(n_shuffle))
    for row in replay_actions[pick]:
      _add(row, 'shuffle')
  uniform = rng.uniform(
      ACTION_LOW, ACTION_HIGH, size=(int(n_uniform), action_dim)
  ).astype(np.float32)
  for row in uniform:
    _add(row, 'uniform')
  return np.stack(rows, axis=0), np.asarray(families)


def score_range(scores: np.ndarray) -> float:
  scores = np.asarray(scores, dtype=np.float64).reshape(-1)
  if scores.size == 0:
    return float('nan')
  return float(np.max(scores) - np.min(scores))


def score_std(scores: np.ndarray) -> float:
  scores = np.asarray(scores, dtype=np.float64).reshape(-1)
  if scores.size < 2:
    return float('nan')
  return float(np.std(scores))


def family_range(scores: np.ndarray, families: np.ndarray, prefix: str) -> float:
  scores = np.asarray(scores, dtype=np.float64).reshape(-1)
  families = np.asarray(families)
  mask = np.array([str(f).startswith(prefix) or str(f) == 'pi' for f in families])
  return score_range(scores[mask])


def survival_curve(values: np.ndarray, eps_grid: np.ndarray) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  values = values[np.isfinite(values)]
  eps_grid = np.asarray(eps_grid, dtype=np.float64).reshape(-1)
  if values.size == 0:
    return np.full(eps_grid.shape, np.nan, dtype=np.float64)
  return np.array(
      [float(np.mean(values > float(eps))) for eps in eps_grid],
      dtype=np.float64)


def bootstrap_survival(
    values: np.ndarray,
    eps_grid: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return mean, 16th, and 84th percentile survival curves."""
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  values = values[np.isfinite(values)]
  mean = survival_curve(values, eps_grid)
  if values.size < 8:
    nan = np.full_like(mean, np.nan)
    return mean, nan, nan
  draws = []
  n = values.size
  for _ in range(int(n_boot)):
    sample = values[rng.integers(0, n, size=n)]
    draws.append(survival_curve(sample, eps_grid))
  stacked = np.stack(draws, axis=0)
  lo = np.percentile(stacked, 16, axis=0)
  hi = np.percentile(stacked, 84, axis=0)
  return mean, lo, hi


def time_bin_mean(
    t_frac: np.ndarray,
    values: np.ndarray,
    n_bins: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Bin a trajectory quantity vs normalized time in ``[0, 1]``."""
  t_frac = np.asarray(t_frac, dtype=np.float64).reshape(-1)
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
  centers = 0.5 * (edges[:-1] + edges[1:])
  means = np.full(int(n_bins), np.nan)
  ses = np.full(int(n_bins), np.nan)
  counts = np.zeros(int(n_bins), dtype=np.float64)
  for i in range(int(n_bins)):
    if i == int(n_bins) - 1:
      mask = (t_frac >= edges[i]) & (t_frac <= edges[i + 1])
    else:
      mask = (t_frac >= edges[i]) & (t_frac < edges[i + 1])
    chunk = values[mask]
    chunk = chunk[np.isfinite(chunk)]
    counts[i] = float(chunk.size)
    if chunk.size:
      means[i] = float(np.mean(chunk))
      ses[i] = float(np.std(chunk) / np.sqrt(chunk.size))
  return centers, means, ses, counts


def default_eps_grid(max_value: float, n: int = 61) -> np.ndarray:
  hi = max(float(max_value), 1e-6)
  return np.linspace(0.0, hi, int(n), dtype=np.float64)


def state_score_scale(f_pi: np.ndarray) -> float:
  """On-policy std of f(s, a_π, g). Used to compare Δ_f across tasks."""
  values = np.asarray(f_pi, dtype=np.float64).reshape(-1)
  values = values[np.isfinite(values)]
  if values.size < 2:
    return float('nan')
  return float(np.std(values))


def relative_delta(delta: np.ndarray, f_pi: np.ndarray) -> np.ndarray:
  """Scale-free action sensitivity: Δ_f / std_s[f(s, a_π, g)]."""
  scale = state_score_scale(f_pi)
  delta = np.asarray(delta, dtype=np.float64)
  if not np.isfinite(scale) or scale <= 1e-12:
    return np.full(delta.shape, np.nan, dtype=np.float64)
  return delta / scale


def summarize_run(npz: Mapping[str, np.ndarray]) -> Dict[str, float]:
  delta = np.asarray(npz['delta_f_task'], dtype=np.float64)
  f_pi = np.asarray(npz['f_pi'], dtype=np.float64)
  rel = relative_delta(delta, f_pi)
  out = {
      'n_states': float(delta.size),
      'delta_f_task_mean': float(np.nanmean(delta)),
      'delta_f_task_median': float(np.nanmedian(delta)),
      'delta_f_gauss010_mean': float(np.nanmean(npz['delta_f_gauss_0.10'])),
      'delta_f_gauss025_mean': float(np.nanmean(npz['delta_f_gauss_0.25'])),
      'sigma_s_f_pi': state_score_scale(f_pi),
      'rel_delta_mean': float(np.nanmean(rel)),
      'rel_delta_median': float(np.nanmedian(rel)),
      'std_a_over_std_s': float(
          np.nanmean(npz['delta_f_std']) / max(state_score_scale(f_pi), 1e-12)),
      'grad_norm_mean': float(np.nanmean(npz['grad_norm'])),
      'contact_mass': float(np.nanmean(npz['contact'])),
      'categorical_accuracy': float(npz['categorical_accuracy'][0]),
      'action_shuffle_accuracy': float(npz['action_shuffle_accuracy'][0]),
  }
  if 'env_delta_taskcoord' in npz and np.asarray(npz['env_delta_taskcoord']).size:
    out['env_delta_taskcoord_mean'] = float(
        np.nanmean(npz['env_delta_taskcoord']))
  return out
