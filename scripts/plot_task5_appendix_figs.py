#!/usr/bin/env python3
"""Appendix figures for the Task-5 contrastive-critic failure.

Figures (measured seed-6 probes only; no invented CIs):

  1. Hover action advice: press rank, gradient compass, std_a vs std_s
  2. State-similarity probe: shuffle heatmap, retrieval slope, occupancy
  3. 20% success-mass inject: eval success and HER success over training

Sources:
  logs/jubail_task5_action_advice/runs/action_advice_*.json
  logs/jubail_task5_action_advice/runs/17901349_0.out  (handle-late probe)
  logs/jubail_task5_feature_shortcut/runs/feature_shortcut_*.json
  logs/jubail_task5_action_advice/runs/17901349_{0_0,1_1,2_2}.out
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

INK = '#111827'
INK_MUTED = '#6B7280'
SPINE = '#D1D5DB'
GRID = '#EEF2F6'
REF = '#9CA3AF'
HANDLE = '#D55E00'
HANDLE_LATE = '#E69F00'
PUSH = '#0072B2'
INJECT = '#009E73'

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
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 8.0,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'dejavusans',
    'lines.solid_capstyle': 'round',
})

ADVICE_DIR = (
    REPO_ROOT / 'logs' / 'jubail_task5_action_advice' / 'runs')
SHORTCUT_DIR = (
    REPO_ROOT / 'logs' / 'jubail_task5_feature_shortcut' / 'runs')
DEFAULT_OUT = REPO_ROOT / 'results' / 'img' / 'paper'
DEFAULT_CSV = REPO_ROOT / 'results' / 'data' / 'task5_appendix'


def _style_ax(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.yaxis.grid(True, color=GRID, lw=0.65, zorder=0)
  ax.set_axisbelow(True)
  ax.tick_params(length=2.6, width=0.6, pad=1.8, direction='out')


def _panel_label(ax, letter):
  ax.text(
      0.0, 1.04, letter, transform=ax.transAxes, fontsize=11,
      fontweight='bold', color=INK, ha='left', va='bottom', clip_on=False)


def _annotate_bar(ax, x, y, text, *, va='bottom', color=INK, dy=None):
  if dy is None:
    dy = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    if va == 'top':
      dy = -dy
  ax.text(
      x, y + dy, text, ha='center', va=va, fontsize=7.2,
      color=color, fontweight='semibold', clip_on=False, zorder=6)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
  return json.loads(Path(path).read_text(encoding='utf-8'))


def probe_row_from_advice(blob: dict, label: str, color: str) -> dict:
  hover = blob['hover_summary']
  sens = blob['sensitivity']
  return {
      'label': label,
      'color': color,
      'env_steps': int(blob['env_steps']),
      'n_hover': int(hover['n']),
      'gap': float(hover['gap_press_minus_pi_task_mean']),
      'rank': float(hover['press_rank_task_mean']),
      'cos': float(hover['grad_cos_press_dir_mean']),
      'std_a': float(sens['hover_mean_action_std_task']),
      'std_s': float(sens['hover_state_std_of_pi_score_task']),
      'ratio': float(sens['action_over_state_std_ratio_task']),
  }


def parse_handle_late_from_stdout(path: Path) -> dict:
  """The inject cell overwrote the late handle JSON; recover from stdout."""
  text = Path(path).read_text(encoding='utf-8')
  summaries = text.split('"hover_summary"')
  if len(summaries) < 3:
    raise ValueError(f'Expected two hover_summary dumps in {path}')
  chunk = summaries[2]
  sens = text.split('"sensitivity"')[2]

  def grab(block, key):
    match = re.search(rf'"{key}":\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', block)
    if not match:
      raise KeyError(key)
    return float(match.group(1))

  return {
      'label': 'Handle 952k',
      'color': HANDLE_LATE,
      'env_steps': 951900,
      'n_hover': int(grab(chunk, 'n')),
      'gap': grab(chunk, 'gap_press_minus_pi_task_mean'),
      'rank': grab(chunk, 'press_rank_task_mean'),
      'cos': grab(chunk, 'grad_cos_press_dir_mean'),
      'std_a': grab(sens, 'hover_mean_action_std_task'),
      'std_s': grab(sens, 'hover_state_std_of_pi_score_task'),
      'ratio': grab(sens, 'action_over_state_std_ratio_task'),
  }


def load_action_advice_rows(
    advice_dir: Path | None = None,
    handle_late_stdout: Path | None = None,
) -> list:
  advice_dir = Path(advice_dir or ADVICE_DIR)
  handle_late_stdout = Path(
      handle_late_stdout or (ADVICE_DIR / '17901349_0.out'))
  handle = load_json(
      advice_dir / 'action_advice_handle_press_side_s6_step100200.json')
  push = load_json(advice_dir / 'action_advice_push_s6_step250500.json')
  return [
      probe_row_from_advice(handle, 'Handle 100k', HANDLE),
      parse_handle_late_from_stdout(handle_late_stdout),
      probe_row_from_advice(push, 'Push 250k', PUSH),
  ]


def shuffle_drops(blob: dict) -> dict:
  retrieval = blob['retrieval']
  density = blob['density']['all_states']
  return {
      'baseline': float(retrieval['baseline_accuracy']),
      'mech_xy': float(retrieval['state_mech_xy_drop']),
      'mech_z': float(retrieval['state_mech_z_drop']),
      'hand': float(retrieval['state_hand_drop']),
      'action': float(retrieval['action_drop']),
      'acc_xy': float(retrieval['state_mech_xy_accuracy']),
      'acc_z': float(retrieval['state_mech_z_accuracy']),
      'acc_hand': float(retrieval['state_hand_accuracy']),
      'acc_action': float(retrieval['action_accuracy']),
      'n_pairs': float(retrieval['n_pairs']),
      'frac_success_band': float(density['frac_in_band']),
      'frac_progress': float(density.get('frac_in_progress_band', 0.0)),
      'xy_std': float(density['xy_std']),
      'z_std': float(density['z_std']),
      'd_z_pi': float(
          blob['one_step_mechanism_delta']['pi']['d_mech_z']['mean']),
      'd_z_press': float(
          blob['one_step_mechanism_delta']['press']['d_mech_z']['mean']),
      'd_xy_pi': float(
          blob['one_step_mechanism_delta']['pi']['d_mech_xy']['mean']),
      'd_xy_press': float(
          blob['one_step_mechanism_delta']['press']['d_mech_xy']['mean']),
  }


def load_feature_rows(shortcut_dir: Path | None = None) -> dict:
  shortcut_dir = Path(shortcut_dir or SHORTCUT_DIR)
  handle = load_json(
      shortcut_dir / 'feature_shortcut_handle_press_side_s6_step100200.json')
  push = load_json(
      shortcut_dir / 'feature_shortcut_push_s6_step250500.json')
  return {
      'Handle 100k': {'color': HANDLE, **shuffle_drops(handle)},
      'Push 250k': {'color': PUSH, **shuffle_drops(push)},
  }


EVAL_RE = re.compile(
    r'\[eval @ (\d+)\] success=([0-9.]+)%')
HER_RE = re.compile(
    r'\[her phase @ (\d+)\] succ=([0-9.]+) hover=([0-9.]+) '
    r'prog=([0-9.]+) far=([0-9.]+) succ\|prog=([0-9.]+)')
INJECT_RE = re.compile(r'\[success inject @ (\d+)\]')
FINAL_RE = re.compile(r'Task 0 \[sawyer_[^\]]+\]: ([0-9.]+)%')


def parse_eval_log(text: str) -> tuple[np.ndarray, np.ndarray, int | None]:
  steps, vals = [], []
  for match in EVAL_RE.finditer(text):
    steps.append(int(match.group(1)))
    vals.append(float(match.group(2)) / 100.0)
  inject_at = None
  found = INJECT_RE.search(text)
  if found:
    inject_at = int(found.group(1))
  finals = list(FINAL_RE.finditer(text))
  if finals and steps:
    final_step = 990000
    if steps[-1] != final_step:
      steps.append(final_step)
      vals.append(float(finals[-1].group(1)) / 100.0)
    else:
      vals[-1] = float(finals[-1].group(1)) / 100.0
  return np.asarray(steps, dtype=np.int64), np.asarray(vals, dtype=np.float64), inject_at


def parse_her_log(text: str) -> dict[str, np.ndarray]:
  steps, succ, prog, both = [], [], [], []
  for match in HER_RE.finditer(text):
    steps.append(int(match.group(1)))
    succ.append(float(match.group(2)))
    prog.append(float(match.group(4)))
    both.append(float(match.group(6)))
  return {
      'steps': np.asarray(steps, dtype=np.int64),
      'succ': np.asarray(succ, dtype=np.float64),
      'prog': np.asarray(prog, dtype=np.float64),
      'succ_or_prog': np.asarray(both, dtype=np.float64),
  }


def load_inject_series(log_dir: Path | None = None) -> dict:
  log_dir = Path(log_dir or ADVICE_DIR)
  specs = (
      ('handle', log_dir / '17901349_0_0.out', HANDLE),
      ('inject', log_dir / '17901349_2_2.out', INJECT),
      ('push', log_dir / '17901349_1_1.out', PUSH),
  )
  out = {}
  for key, path, color in specs:
    text = path.read_text(encoding='utf-8')
    steps, vals, inject_at = parse_eval_log(text)
    her = parse_her_log(text)
    out[key] = {
        'color': color,
        'eval_steps': steps,
        'eval_success': vals,
        'her': her,
        'inject_at': inject_at,
    }
  return out


# ---------------------------------------------------------------------------
# Figure 1 — hover action advice (rank, gradient, variance)
# ---------------------------------------------------------------------------

def _draw_compass(ax, cos_val, color):
  """Unit circle: press axis along +x, critic gradient at arccos(cos)."""
  theta = float(np.arccos(np.clip(cos_val, -1.0, 1.0)))
  ring = plt.Circle((0.0, 0.0), 1.0, fill=False, color=SPINE, lw=0.8, zorder=1)
  ax.add_patch(ring)
  ax.plot([-1.08, 1.08], [0, 0], color=SPINE, lw=0.6, zorder=1)
  ax.plot([0, 0], [-0.18, 1.08], color=SPINE, lw=0.6, zorder=1)
  ax.annotate(
      '', xy=(1.0, 0.0), xytext=(0.0, 0.0),
      arrowprops=dict(
          arrowstyle='-|>', color=REF, lw=1.3,
          mutation_scale=9),
      zorder=2)
  ax.text(1.18, 0.0, 'press', fontsize=6.6, color=INK_MUTED, va='center', ha='left')
  gx, gy = np.cos(theta), np.sin(theta)
  ax.annotate(
      '', xy=(gx, gy), xytext=(0.0, 0.0),
      arrowprops=dict(
          arrowstyle='-|>', color=color, lw=1.8,
          mutation_scale=11),
      zorder=3)
  ax.plot(gx, gy, 'o', color=color, ms=4.0, zorder=4,
          markeredgecolor='white', markeredgewidth=0.5)
  ax.set_xlim(-1.35, 1.55)
  ax.set_ylim(-0.35, 1.25)
  ax.set_aspect('equal')
  ax.axis('off')


def fig_state_vs_action(rows, out_pdf: Path, out_png: Path, csv_rows: list):
  """Match the appendix table: Task 5 at 100k vs Push at 250k."""
  shown = [r for r in rows if r['label'] in ('Handle 100k', 'Push 250k')]
  fig = plt.figure(figsize=(7.16, 2.55))
  gs = fig.add_gridspec(
      1, 3, width_ratios=[1.05, 1.15, 1.15],
      left=0.07, right=0.99, top=0.80, bottom=0.24, wspace=0.38)
  ax_r, ax_c, ax_v = (fig.add_subplot(gs[0, i]) for i in range(3))

  # A — press rank among 34 candidate actions
  _panel_label(ax_r, 'A')
  ax_r.axhspan(16.5, 18.5, color='#EEF2F6', zorder=0)
  ax_r.axhline(17.5, color=REF, lw=0.7, ls=(0, (3.2, 2.4)), zorder=1)
  ax_r.text(
      -0.38, 17.5, 'chance', fontsize=6.6, color=INK_MUTED,
      ha='right', va='center')
  for i, row in enumerate(shown):
    ax_r.plot([i, i], [1, 34], color=SPINE, lw=1.1, zorder=1)
    ax_r.plot(
        i, row['rank'], 'o', color=row['color'], ms=9.5, zorder=4,
        markeredgecolor='white', markeredgewidth=0.7)
    rank_txt = f"{row['rank']:.1f}" if row['rank'] >= 2 else f"{row['rank']:.2f}"
    ax_r.annotate(
        rank_txt, xy=(i, row['rank']),
        xytext=(8, 0), textcoords='offset points',
        fontsize=8.0, fontweight='semibold', color=row['color'], va='center')
    ax_r.text(
        i, 36.2, row['label'].replace(' 100k', '').replace(' 250k', ''),
        ha='center', va='bottom', fontsize=7.5, color=INK)
  ax_r.set_ylim(36.5, 0.4)
  ax_r.set_xlim(-0.85, 1.55)
  ax_r.set_xticks([])
  ax_r.set_ylabel('Press rank  (1 = best of 34)')
  ax_r.set_yticks([1, 10, 17.5, 25, 34])
  ax_r.set_yticklabels(['1', '10', '17.5', '25', '34'])
  _style_ax(ax_r)
  ax_r.spines['bottom'].set_visible(False)
  ax_r.tick_params(axis='x', length=0)

  # B — two gradient compasses
  _panel_label(ax_c, 'B')
  ax_c.axis('off')
  inner = ax_c.inset_axes([0.0, 0.02, 1.0, 0.96])
  inner.axis('off')
  left = inner.inset_axes([0.0, 0.08, 0.48, 0.84])
  right = inner.inset_axes([0.52, 0.08, 0.48, 0.84])
  _draw_compass(left, shown[0]['cos'], shown[0]['color'])
  _draw_compass(right, shown[1]['cos'], shown[1]['color'])
  left.set_title(
      f"Handle\ncos = {shown[0]['cos']:+.2f}",
      fontsize=7.2, color=shown[0]['color'], pad=2)
  right.set_title(
      f"Push\ncos = {shown[1]['cos']:+.2f}",
      fontsize=7.2, color=shown[1]['color'], pad=2)
  ax_c.text(
      0.5, -0.08,
      r'Alignment of $\nabla_a f$ with the press/push axis',
      transform=ax_c.transAxes, ha='center', va='top',
      fontsize=8.0, color=INK)

  # C — std across states vs across actions
  _panel_label(ax_v, 'C')
  for i, row in enumerate(shown):
    y = 1 - i
    ax_v.plot(
        [row['std_a'], row['std_s']], [y, y],
        color=row['color'], lw=2.0, zorder=2, solid_capstyle='round')
    ax_v.plot(
        row['std_s'], y, 's', color='#4B4B4B', ms=7.5, zorder=4,
        markeredgecolor='white', markeredgewidth=0.5)
    ax_v.plot(
        row['std_a'], y, 'o', color=row['color'], ms=8.0, zorder=4,
        markeredgecolor='white', markeredgewidth=0.5)
    ax_v.text(
        0.72, y + 0.32, row['label'].replace(' 100k', '').replace(' 250k', ''),
        fontsize=7.5, color=INK, va='bottom')
    ratio_x = max(row['std_a'], row['std_s']) * 1.18
    ax_v.text(
        ratio_x, y,
        rf"{row['ratio']:.3f}",
        fontsize=7.8, fontweight='semibold', color=row['color'], va='center')
  ax_v.set_xscale('log')
  ax_v.set_xlim(0.7, 90)
  ax_v.set_ylim(-0.45, 1.55)
  ax_v.set_yticks([])
  ax_v.set_xlabel('Critic-score scale at hover')
  ax_v.text(
      0.0, -0.28, r'$\blacksquare$  across states    $\bullet$  across actions',
      transform=ax_v.transAxes, fontsize=6.6, color=INK_MUTED, ha='left')
  _style_ax(ax_v)
  ax_v.spines['left'].set_visible(False)
  ax_v.tick_params(axis='y', length=0)
  ax_v.text(
      0.98, 1.02, r'$\mathrm{std}_a/\mathrm{std}_s$',
      transform=ax_v.transAxes, ha='right', va='bottom',
      fontsize=7.0, color=INK_MUTED)

  for row in rows:
    csv_rows.append({
        'figure': 'state_vs_action',
        'series': row['label'],
        'env_steps': row['env_steps'],
        'n_hover': row['n_hover'],
        'press_minus_pi': row['gap'],
        'press_rank': row['rank'],
        'grad_cos': row['cos'],
        'std_a': row['std_a'],
        'std_s': row['std_s'],
        'std_a_over_std_s': row['ratio'],
    })
  fig.savefig(out_pdf)
  fig.savefig(out_png)
  plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — state similarity / feature shuffle
# ---------------------------------------------------------------------------

def fig_feature_shuffle(feature_rows, out_pdf: Path, out_png: Path, csv_rows: list):
  fig = plt.figure(figsize=(7.16, 2.55))
  gs = fig.add_gridspec(
      1, 3, width_ratios=[1.05, 1.2, 1.05],
      left=0.06, right=0.99, top=0.80, bottom=0.28, wspace=0.32)
  ax_h, ax_s, ax_o = (fig.add_subplot(gs[0, i]) for i in range(3))

  handle = feature_rows['Handle 100k']
  push = feature_rows['Push 250k']
  feat_keys = ['mech_xy', 'mech_z', 'hand', 'action']
  feat_labels = [r'$xy$', r'$z$', 'Hand', 'Action']
  mat = np.array([
      [handle[k] for k in feat_keys],
      [push[k] for k in feat_keys],
  ], dtype=np.float64)
  cmap = mpl.colors.LinearSegmentedColormap.from_list(
      'drop', ['#F8FAFC', '#FDE68A', '#D55E00'])
  im = ax_h.imshow(
      mat, cmap=cmap, vmin=0.0, vmax=0.22, aspect='auto', zorder=1)
  ax_h.set_xticks(np.arange(4))
  ax_h.set_xticklabels(feat_labels, fontsize=7.5)
  ax_h.set_yticks([0, 1])
  ax_h.set_yticklabels(['Task 5', 'Push'], fontsize=8.0)
  for i in range(2):
    for j in range(4):
      val = mat[i, j]
      ax_h.text(
          j, i, f'{val:.2f}', ha='center', va='center',
          fontsize=7.6, fontweight='semibold',
          color=INK if val < 0.12 else 'white')
  for spine in ax_h.spines.values():
    spine.set_visible(False)
  ax_h.tick_params(length=0, pad=3)
  ax_h.set_xlabel('Shuffled block of $(s,a)$')
  _panel_label(ax_h, 'A')
  cax = ax_h.inset_axes([1.03, 0.15, 0.04, 0.7])
  cb = fig.colorbar(im, cax=cax)
  cb.set_ticks([0.0, 0.10, 0.20])
  cb.ax.tick_params(labelsize=6.4, length=2)
  cb.set_label('Accuracy drop', fontsize=6.6, labelpad=2)

  chance = 1.0 / 256.0
  xs = np.arange(4)
  xlabels = ['Intact', r'Shuffle $xy$', r'Shuffle $z$', 'Shuffle\naction']
  for rec, lab in ((handle, 'Task 5'), (push, 'Push')):
    ys = [rec['baseline'], rec['acc_xy'], rec['acc_z'], rec['acc_action']]
    ax_s.plot(xs, ys, color=rec['color'], lw=1.9, zorder=3, solid_capstyle='round')
    ax_s.scatter(
        xs, ys, s=28, color=rec['color'], zorder=4,
        edgecolors='white', linewidths=0.6)
    ax_s.text(
        -0.12, ys[0] + (0.012 if lab == 'Push' else -0.012),
        lab, color=rec['color'], fontsize=7.2, ha='right', va='center',
        fontweight='semibold')
  ax_s.axhline(chance, color=REF, lw=0.7, ls=(0, (3.2, 2.4)), zorder=1)
  ax_s.text(
      3.05, chance + 0.006, 'chance', fontsize=6.4, color=INK_MUTED,
      ha='right', va='bottom')
  ax_s.set_xticks(xs)
  ax_s.set_xticklabels(xlabels, fontsize=7.0)
  ax_s.set_ylim(-0.01, 0.24)
  ax_s.set_ylabel('256-way HER accuracy')
  ax_s.set_xlim(-0.55, 3.25)
  _style_ax(ax_s)
  _panel_label(ax_s, 'B')

  occ_colors = {
      'success': '#0072B2',
      'progress': '#56B4E9',
      'other': '#E5E7EB',
  }
  specs = [
      ('Task 5', handle, HANDLE),
      ('Push', push, PUSH),
  ]
  for i, (name, rec, _) in enumerate(specs):
    y = 1 - i
    succ = rec['frac_success_band']
    prog = rec['frac_progress']
    other = max(0.0, 1.0 - succ - prog)
    ax_o.barh(y, succ, height=0.42, color=occ_colors['success'], zorder=3)
    ax_o.barh(
        y, prog, left=succ, height=0.42, color=occ_colors['progress'], zorder=3)
    ax_o.barh(
        y, other, left=succ + prog, height=0.42, color=occ_colors['other'],
        zorder=3)
    if succ >= 0.08:
      ax_o.text(
          succ / 2.0, y, f'{100 * succ:.0f}%', ha='center', va='center',
          fontsize=6.6, color='white', fontweight='semibold')
    elif succ > 0.01:
      ax_o.text(
          succ + 0.02, y, f'{100 * succ:.0f}% success',
          ha='left', va='center', fontsize=6.5, color=INK)
    if prog > 0.08:
      ax_o.text(
          succ + prog / 2.0, y, f'{100 * prog:.0f}%', ha='center', va='center',
          fontsize=6.6, color=INK, fontweight='semibold')
    ax_o.text(-0.03, y, name, ha='right', va='center', fontsize=8.0, color=INK)
  ax_o.set_xlim(0, 1.0)
  ax_o.set_ylim(-0.55, 1.55)
  ax_o.set_yticks([])
  ax_o.set_xlabel('On-policy occupancy')
  ax_o.set_xticks([0.0, 0.5, 1.0])
  _style_ax(ax_o)
  ax_o.spines['left'].set_visible(False)
  _panel_label(ax_o, 'C')
  ax_o.plot([], [], color=occ_colors['success'], lw=6, label='Success band')
  ax_o.plot([], [], color=occ_colors['progress'], lw=6, label='Progress band')
  ax_o.plot([], [], color=occ_colors['other'], lw=6, label='Other')
  ax_o.legend(
      loc='upper center', fontsize=6.4, handlelength=1.1, ncol=3,
      bbox_to_anchor=(0.55, -0.22), borderaxespad=0.0, columnspacing=1.1)

  for name, rec in feature_rows.items():
    for key, lab in (
        ('mech_xy', 'xy'), ('mech_z', 'z'), ('hand', 'hand'),
        ('action', 'action')):
      csv_rows.append({
          'figure': 'feature_shuffle',
          'panel': 'drop',
          'series': name,
          'feature': key,
          'accuracy_drop': rec[key],
          'baseline_accuracy': rec['baseline'],
          'n_pairs': rec['n_pairs'],
      })
    csv_rows.append({
        'figure': 'feature_shuffle',
        'panel': 'occupancy',
        'series': name,
        'frac_success_band': rec['frac_success_band'],
        'frac_progress': rec['frac_progress'],
    })
    for key in ('d_xy_pi', 'd_xy_press', 'd_z_pi', 'd_z_press'):
      csv_rows.append({
          'figure': 'feature_shuffle',
          'panel': 'one_step_mm',
          'series': name,
          'feature': key,
          'delta_mm': 1000.0 * rec[key],
      })

  fig.savefig(out_pdf)
  fig.savefig(out_png)
  plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — success-mass inject
# ---------------------------------------------------------------------------

def fig_inject(series, out_pdf: Path, out_png: Path, csv_rows: list):
  fig, axes = plt.subplots(2, 1, figsize=(7.16, 3.85), sharex=True)
  fig.subplots_adjust(
      left=0.10, right=0.99, top=0.86, bottom=0.13, hspace=0.16)
  inject_at = series['inject']['inject_at'] or 250500
  names = [
      ('handle', 'Handle', HANDLE, 1.6),
      ('inject', 'Handle + 20% inject', INJECT, 1.85),
      ('push', 'Push', PUSH, 1.6),
  ]

  ax = axes[0]
  ax.axvline(inject_at / 1000.0, color=REF, lw=0.85, ls=(0, (3.2, 2.4)), zorder=2)
  for key, lab, color, lw in names:
    rec = series[key]
    x = rec['eval_steps'] / 1000.0
    y = rec['eval_success']
    ax.plot(x, y, color=color, label=lab, zorder=3, lw=lw)
    ax.scatter(x, y, s=14, color=color, zorder=4, linewidths=0)
    for step, val in zip(rec['eval_steps'], rec['eval_success']):
      csv_rows.append({
          'figure': 'inject',
          'panel': 'eval',
          'series': key,
          'env_steps': int(step),
          'success': float(val),
      })
  inj = dict(zip(
      series['inject']['eval_steps'].tolist(),
      series['inject']['eval_success'].tolist()))
  callouts = (
      (250500, inj.get(250500, 0.20), '20%', (8, 10)),
      (300600, inj.get(300600, 0.10), '10%', (10, -14)),
      (990000, inj.get(990000, 0.0), '0%', (-18, 12)),
  )
  for step, val, text, offset in callouts:
    ax.annotate(
        text, xy=(step / 1000.0, val), xytext=offset,
        textcoords='offset points', color=INJECT, fontsize=7.4,
        fontweight='semibold',
        arrowprops=dict(arrowstyle='-', color=INJECT, lw=0.6))
  _style_ax(ax)
  ax.set_ylim(-0.06, 1.12)
  ax.set_yticks([0.0, 0.5, 1.0])
  ax.set_ylabel('Eval success')
  _panel_label(ax, 'A')
  ax.text(
      inject_at / 1000.0 - 8, 1.05, '20% success\ncloned into replay',
      fontsize=6.6, color=INK_MUTED, ha='right', va='top')

  ax = axes[1]
  ax.axvline(inject_at / 1000.0, color=REF, lw=0.85, ls=(0, (3.2, 2.4)), zorder=2)
  ax.axhline(0.20, color=INJECT, lw=0.7, ls=(0, (1.6, 1.6)), alpha=0.75, zorder=1)
  for key, lab, color, lw in names:
    rec = series[key]
    her = rec['her']
    ax.plot(
        her['steps'] / 1000.0, her['succ'],
        color=color, zorder=3, lw=lw)
    for step, succ, prog, both in zip(
        her['steps'], her['succ'], her['prog'], her['succ_or_prog']):
      csv_rows.append({
          'figure': 'inject',
          'panel': 'her',
          'series': key,
          'env_steps': int(step),
          'her_succ': float(succ),
          'her_prog': float(prog),
          'her_succ_or_prog': float(both),
      })
  _style_ax(ax)
  ax.set_ylim(-0.02, 0.55)
  ax.set_yticks([0.0, 0.2, 0.4])
  ax.set_ylabel('HER success mass')
  ax.set_xlabel('Environment steps ($\\times 10^3$)')
  ax.set_xlim(0, 1000)
  _panel_label(ax, 'B')
  ax.text(
      992, 0.215, 'inject target',
      fontsize=6.6, color=INJECT, ha='right', va='bottom')

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
      handles, labels, loc='upper center', ncol=3, frameon=False,
      handlelength=1.7, handletextpad=0.45, columnspacing=1.6,
      bbox_to_anchor=(0.55, 1.01), bbox_transform=fig.transFigure)

  fig.savefig(out_pdf)
  fig.savefig(out_png)
  plt.close(fig)


def _write_csv(path: Path, rows: list):
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = []
  for row in rows:
    for key in row:
      if key not in keys:
        keys.append(key)
  with path.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--out-dir', default=str(DEFAULT_OUT))
  parser.add_argument('--csv-dir', default=str(DEFAULT_CSV))
  parser.add_argument('--advice-dir', default=str(ADVICE_DIR))
  parser.add_argument('--shortcut-dir', default=str(SHORTCUT_DIR))
  args = parser.parse_args()
  out_dir = Path(args.out_dir)
  csv_dir = Path(args.csv_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  csv_dir.mkdir(parents=True, exist_ok=True)

  advice_rows = load_action_advice_rows(Path(args.advice_dir))
  feature_rows = load_feature_rows(Path(args.shortcut_dir))
  inject_series = load_inject_series(Path(args.advice_dir))

  csv_rows = []
  fig_state_vs_action(
      advice_rows,
      out_dir / 'fig_task5_state_vs_action.pdf',
      out_dir / 'fig_task5_state_vs_action.png',
      csv_rows)
  fig_feature_shuffle(
      feature_rows,
      out_dir / 'fig_task5_feature_shuffle.pdf',
      out_dir / 'fig_task5_feature_shuffle.png',
      csv_rows)
  fig_feature_shuffle(
      feature_rows,
      out_dir / 'fig_task5_state_similarity.pdf',
      out_dir / 'fig_task5_state_similarity.png',
      [])
  fig_inject(
      inject_series,
      out_dir / 'fig_task5_success_inject.pdf',
      out_dir / 'fig_task5_success_inject.png',
      csv_rows)
  _write_csv(csv_dir / 'fig_task5_appendix.csv', csv_rows)
  print('Wrote appendix figures to', out_dir)


if __name__ == '__main__':
  main()
