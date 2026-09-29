"""Helpers for Task-5 / Task-6 episode filmstrips.

The strip is a visualization of occupancy, not of critic scores.
Task 5 is a latch: the handle pose is nearly constant until a brief
press. Task 6 is a path: the cube translates over a large fraction of
the horizon.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from contrastive.action_informative_mass import task_coordinate
from contrastive.her_future_phase import (
    PHASE_FAR,
    PHASE_HOVER,
    PHASE_OTHER,
    PHASE_PROGRESS,
    PHASE_SUCCESS,
    TASK_GEOMETRY,
    classify_her_goal,
)

INK = '#111827'
INK_MUTED = '#6B7280'
SPINE = '#D1D5DB'
GRID = '#EEF2F6'
HANDLE_LINE = '#D55E00'
PUSH_LINE = '#0072B2'
BAND = '#DEEAF6'

# Tight tabletop crop in relative image coordinates. The default Sawyer
# camera wastes most of the frame on floor tiles; the latch vs path
# difference lives on the table.
CROP_Y0, CROP_Y1 = 0.12, 0.73
CROP_X0, CROP_X1 = 0.17, 0.83

PHASE_COLORS = {
    PHASE_FAR: '#9CA3AF',
    PHASE_OTHER: '#9CA3AF',
    PHASE_HOVER: '#E69F00',
    PHASE_PROGRESS: '#56B4E9',
    PHASE_SUCCESS: '#0072B2',
}
PHASE_LABELS = {
    PHASE_FAR: 'Far',
    PHASE_OTHER: 'Near',
    PHASE_HOVER: 'Hover',
    PHASE_PROGRESS: 'Progress',
    PHASE_SUCCESS: 'Success',
}

mpl.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.labelcolor': INK,
    'axes.titlecolor': INK,
    'axes.edgecolor': SPINE,
    'axes.linewidth': 0.7,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'xtick.labelsize': 7.0,
    'ytick.labelsize': 7.0,
    'legend.fontsize': 7.0,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def even_indices(n_steps: int, n_keep: int = 8) -> np.ndarray:
  n_steps = int(n_steps)
  n_keep = min(int(n_keep), max(n_steps, 1))
  if n_steps <= 1:
    return np.zeros((1,), dtype=np.int32)
  return np.unique(np.linspace(0, n_steps - 1, n_keep).round().astype(np.int32))


def select_strip_indices(
    n_steps: int,
    phases: Sequence[str],
    n_keep: int = 8,
) -> np.ndarray:
  """Keyframe + even samples, with a minimum gap so frames are not twins."""
  n_steps = int(n_steps)
  n_keep = min(int(n_keep), max(n_steps, 1))
  if n_steps <= n_keep:
    return np.arange(n_steps, dtype=np.int32)
  phases = [str(p) for p in np.asarray(phases).reshape(-1).tolist()]
  ranked = [0, n_steps - 1]
  for label in (PHASE_SUCCESS, PHASE_HOVER, PHASE_PROGRESS):
    for i, phase in enumerate(phases):
      if phase == label:
        ranked.append(i)
        break
  ranked.extend(int(i) for i in even_indices(n_steps, n_keep))
  min_gap = max(n_steps // (2 * n_keep), 6)
  chosen = []
  for idx in ranked:
    if idx < 0 or idx >= n_steps:
      continue
    if chosen and min(abs(idx - j) for j in chosen) < min_gap:
      continue
    chosen.append(idx)
    if len(chosen) == n_keep:
      break
  if len(chosen) < n_keep:
    for idx in even_indices(n_steps, n_keep * 3):
      if len(chosen) >= n_keep:
        break
      if any(abs(int(idx) - j) < min_gap for j in chosen):
        continue
      chosen.append(int(idx))
  return np.asarray(sorted(set(int(i) for i in chosen)), dtype=np.int32)


def crop_workspace(frame: np.ndarray) -> np.ndarray:
  frame = np.asarray(frame)
  if frame.ndim != 3:
    raise ValueError(f'frame must be HWC; got {frame.shape}')
  height, width = frame.shape[:2]
  y0 = int(round(CROP_Y0 * height))
  y1 = int(round(CROP_Y1 * height))
  x0 = int(round(CROP_X0 * width))
  x1 = int(round(CROP_X1 * width))
  y1 = min(max(y1, y0 + 1), height)
  x1 = min(max(x1, x0 + 1), width)
  return frame[y0:y1, x0:x1]


def success_band(env_name: str):
  spec = TASK_GEOMETRY[str(env_name)]
  if 'axis_index' in spec:
    target = float(spec['axis_target'])
    thr = float(spec['axis_threshold'])
    return target - thr, target + thr
  return 0.0, float(spec['success_threshold'])


def phase_sequence(states: np.ndarray, env_name: str) -> np.ndarray:
  return np.array([classify_her_goal(s, env_name) for s in states])


def coordinate_sequence(states: np.ndarray, env_name: str) -> np.ndarray:
  return np.array([task_coordinate(s, env_name) for s in states], dtype=np.float64)


def resize_frame(frame: np.ndarray, height: int = 120) -> np.ndarray:
  frame = np.asarray(frame)
  if frame.ndim != 3:
    raise ValueError(f'frame must be HWC; got {frame.shape}')
  h, w = frame.shape[:2]
  width = max(int(round(w * height / float(h))), 1)
  try:
    from PIL import Image
  except ImportError:
    ys = np.linspace(0, h - 1, height).round().astype(int)
    xs = np.linspace(0, w - 1, width).round().astype(int)
    return frame[ys][:, xs]
  img = Image.fromarray(frame)
  return np.asarray(img.resize((width, height), Image.BILINEAR))


def _phase_rgb(phases: Sequence[str]) -> np.ndarray:
  rgb = np.zeros((1, len(phases), 3), dtype=np.float64)
  for i, phase in enumerate(phases):
    hex_color = PHASE_COLORS.get(str(phase), '#D1D5DB')
    rgb[0, i] = mpl.colors.to_rgb(hex_color)
  return rgb


def _draw_row(fig, gs_row, rec: Mapping, *, letter: str, title: str,
              coord_name: str, line_color: str, n_keep: int = 8):
  frames = rec['frames']
  phases = np.asarray(rec['phases'])
  coords = np.asarray(rec['coords'], dtype=np.float64)
  n = int(frames.shape[0])
  idx = select_strip_indices(n, phases, n_keep)
  n_cols = int(idx.size)
  inner = gs_row.subgridspec(
      4, 1, height_ratios=[0.14, 1.15, 0.12, 0.40], hspace=0.05)

  ax_t = fig.add_subplot(inner[0, 0])
  ax_t.set_axis_off()
  ax_t.text(
      0.0, 0.35, letter, fontsize=11, fontweight='bold', color=INK,
      ha='left', va='center', transform=ax_t.transAxes)
  ax_t.text(
      0.045, 0.35, title, fontsize=10, color=INK,
      ha='left', va='center', transform=ax_t.transAxes)

  frame_gs = inner[1, 0].subgridspec(1, n_cols, wspace=0.028)
  for j, t in enumerate(idx):
    ax = fig.add_subplot(frame_gs[0, j])
    ax.imshow(resize_frame(crop_workspace(frames[int(t)])))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_color(PHASE_COLORS.get(str(phases[int(t)]), SPINE))
      spine.set_linewidth(1.6)
    ax.set_xlabel(f'{int(t)}', fontsize=6.8, labelpad=1, color=INK_MUTED)

  ax_p = fig.add_subplot(inner[2, 0])
  ax_p.imshow(
      _phase_rgb(phases), aspect='auto', interpolation='nearest',
      extent=(0, max(n - 1, 1), 0, 1))
  ax_p.set_yticks([])
  ax_p.set_xlim(0, max(n - 1, 1))
  ax_p.tick_params(axis='x', labelbottom=False, length=0)
  for t in idx:
    ax_p.axvline(float(t), color=INK, lw=0.5, alpha=0.45)
  ax_p.spines['top'].set_visible(False)
  ax_p.spines['right'].set_visible(False)

  ax_c = fig.add_subplot(inner[3, 0], sharex=ax_p)
  env_name = str(np.asarray(rec['env_name']).reshape(-1)[0])
  lo, hi = success_band(env_name)
  ax_c.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
  t_axis = np.arange(n, dtype=np.float64)
  ax_c.plot(t_axis, coords, color=line_color, lw=1.2, zorder=2)
  ax_c.scatter(
      idx, coords[idx], s=16, color=line_color, zorder=3,
      edgecolors='white', linewidths=0.4)
  ax_c.set_xlim(0, max(n - 1, 1))
  ax_c.set_xlabel('Episode step')
  ax_c.set_ylabel(coord_name, fontsize=7.5)
  ax_c.spines['top'].set_visible(False)
  ax_c.spines['right'].set_visible(False)
  ax_c.yaxis.grid(True, color=GRID, lw=0.6)
  ax_c.set_axisbelow(True)


def compose_strips(
    handle: Mapping,
    push: Mapping,
    out_pdf: Path,
    out_png: Path,
    *,
    n_keep: int = 8,
) -> None:
  fig = plt.figure(figsize=(7.16, 5.05))
  outer = fig.add_gridspec(
      2, 1, height_ratios=[1.0, 1.0], hspace=0.30,
      left=0.09, right=0.995, top=0.91, bottom=0.08)
  handle_step = int(np.asarray(handle['env_steps']).reshape(-1)[0])
  push_step = int(np.asarray(push['env_steps']).reshape(-1)[0])
  _draw_row(
      fig, outer[0], handle,
      letter='a',
      title=(
          f'Task 5  Handle press  (latch, {handle_step // 1000}k ckpt)'
      ),
      coord_name='Handle $z$ (m)',
      line_color=HANDLE_LINE,
      n_keep=n_keep)
  _draw_row(
      fig, outer[1], push,
      letter='b',
      title=(
          f'Task 6  Push  (path, {push_step // 1000}k ckpt)'
      ),
      coord_name='Object–target (m)',
      line_color=PUSH_LINE,
      n_keep=n_keep)

  handles = [
      mpl.patches.Patch(color=PHASE_COLORS[p], label=PHASE_LABELS[p])
      for p in (PHASE_FAR, PHASE_HOVER, PHASE_PROGRESS, PHASE_SUCCESS)
  ]
  fig.legend(
      handles=handles, loc='upper center', ncol=4, frameon=False,
      handlelength=1.1, columnspacing=1.3,
      bbox_to_anchor=(0.54, 1.01), bbox_transform=fig.transFigure)

  out_pdf.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_pdf)
  fig.savefig(out_png)
  plt.close(fig)


def pack_episode(
    frames: np.ndarray,
    states: np.ndarray,
    env_name: str,
    *,
    env_steps: int,
    success: float,
    attempts: int,
    checkpoint: str,
) -> Dict[str, np.ndarray]:
  phases = phase_sequence(states, env_name)
  coords = coordinate_sequence(states, env_name)
  return {
      'frames': np.asarray(frames),
      'states': np.asarray(states),
      'phases': phases.astype(object),
      'coords': coords,
      'env_name': np.array(env_name),
      'env_steps': np.array([int(env_steps)]),
      'success': np.array([float(success)]),
      'attempts': np.array([int(attempts)]),
      'checkpoint': np.array(checkpoint),
  }
