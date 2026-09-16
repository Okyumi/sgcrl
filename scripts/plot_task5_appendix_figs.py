#!/usr/bin/env python3
"""Appendix figures for the Task-5 contrastive-critic failure.

Three figures, measured values only (seed 6, no invented CIs):

  1. Same-state action advice vs state-induced score variation
  2. Feature-shuffle retrieval drops (mechanism xy / z / hand / action)
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
  return {
      'baseline': float(blob['retrieval']['baseline_accuracy']),
      'mech_xy': float(blob['retrieval']['state_mech_xy_drop']),
      'mech_z': float(blob['retrieval']['state_mech_z_drop']),
      'hand': float(blob['retrieval']['state_hand_drop']),
      'action': float(blob['retrieval']['action_drop']),
      'n_pairs': float(blob['retrieval']['n_pairs']),
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
# Figure 1
# ---------------------------------------------------------------------------

def fig_state_vs_action(rows, out_pdf: Path, out_png: Path, csv_rows: list):
  fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.55))
  fig.subplots_adjust(
      left=0.09, right=0.99, top=0.90, bottom=0.14, wspace=0.38, hspace=0.52)
  xs = np.arange(len(rows))
  colors = [r['color'] for r in rows]
  labels = [r['label'] for r in rows]
  width = 0.62

  ax = axes[0, 0]
  gaps = [r['gap'] for r in rows]
  ax.axhline(0.0, color=REF, lw=0.7, ls=(0, (3.2, 2.4)), zorder=1)
  ax.bar(xs, gaps, width=width, color=colors, zorder=3)
  _style_ax(ax)
  ax.set_xticks(xs)
  ax.set_xticklabels(labels, fontsize=7.2)
  ax.set_ylabel('Press $-$ $\\pi$ score')
  ax.set_ylim(-0.55, 2.45)
  _panel_label(ax, 'A')
  for x, y in zip(xs, gaps):
    va = 'bottom' if y >= 0 else 'top'
    _annotate_bar(ax, x, y, f'{y:+.2f}', va=va)

  ax = axes[0, 1]
  ranks = [r['rank'] for r in rows]
  ax.axhline(17.5, color=REF, lw=0.7, ls=(0, (3.2, 2.4)), zorder=1)
  for x, y, color in zip(xs, ranks, colors):
    ax.plot([x, x], [17.5, y], color=color, lw=2.0, zorder=3, solid_capstyle='round')
    ax.plot(
        x, y, 'o', color=color, ms=7.5, zorder=4,
        markeredgecolor='white', markeredgewidth=0.6)
  _style_ax(ax)
  ax.set_xticks(xs)
  ax.set_xticklabels(labels, fontsize=7.2)
  ax.set_ylabel('Press rank (of 34)')
  ax.set_ylim(34.5, 0.2)
  ax.set_xlim(-0.72, 2.55)
  ax.text(
      -0.48, 17.5, 'chance', ha='right', va='center',
      fontsize=6.8, color=INK_MUTED)
  _panel_label(ax, 'B')
  offsets = {0: (6, 5), 1: (6, 5), 2: (6, 5)}
  for i, (x, y) in enumerate(zip(xs, ranks)):
    ax.annotate(
        f'{y:.1f}', xy=(x, y), xytext=offsets[i],
        textcoords='offset points', fontsize=7.2, fontweight='semibold',
        color=INK, ha='left', va='center')

  ax = axes[1, 0]
  coss = [r['cos'] for r in rows]
  ax.axhline(0.0, color=REF, lw=0.7, ls=(0, (3.2, 2.4)), zorder=1)
  ax.bar(xs, coss, width=width, color=colors, zorder=3)
  _style_ax(ax)
  ax.set_xticks(xs)
  ax.set_xticklabels(labels, fontsize=7.2)
  ax.set_ylabel(r'$\cos(\nabla_a s,\, d_{\mathrm{press}})$')
  ax.set_ylim(-1.05, 1.15)
  _panel_label(ax, 'C')
  for x, y in zip(xs, coss):
    va = 'bottom' if y >= 0 else 'top'
    _annotate_bar(ax, x, y, f'{y:+.2f}', va=va)

  ax = axes[1, 1]
  w = 0.34
  std_s = [r['std_s'] for r in rows]
  std_a = [r['std_a'] for r in rows]
  h_state = ax.bar(
      xs - w / 2, std_s, width=w, color='#4B4B4B', zorder=3)
  h_act = ax.bar(
      xs + w / 2, std_a, width=w, color=colors, zorder=3)
  _style_ax(ax)
  ax.set_xticks(xs)
  ax.set_xticklabels(
      [f"{r['label']}\n{r['ratio']:.3f}" for r in rows], fontsize=7.2)
  ax.set_ylabel('Score std. dev.')
  ax.set_yscale('log')
  ax.set_ylim(0.9, 80)
  _panel_label(ax, 'D')
  ax.text(
      xs[0] - w / 2, std_s[0] * 1.08, r'$\mathrm{std}_s$',
      ha='center', va='bottom', fontsize=7.0, color='#4B4B4B')
  ax.text(
      xs[0] + w / 2, std_a[0] * 1.12, r'$\mathrm{std}_a$',
      ha='center', va='bottom', fontsize=7.0, color=colors[0])
  ax.set_xlabel(r'$\mathrm{std}_a/\mathrm{std}_s$', labelpad=2, fontsize=8)

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
# Figure 2
# ---------------------------------------------------------------------------

def fig_feature_shuffle(feature_rows, out_pdf: Path, out_png: Path, csv_rows: list):
  fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
  fig.subplots_adjust(
      left=0.08, right=0.99, top=0.82, bottom=0.26, wspace=0.36)

  features = [
      ('mech_xy', r'Mechanism $xy$'),
      ('mech_z', r'Mechanism $z$'),
      ('hand', 'Hand'),
      ('action', 'Action'),
  ]
  series = list(feature_rows.items())
  n_feat = len(features)
  xs = np.arange(n_feat)
  width = 0.36
  ax = axes[0]
  for i, (name, rec) in enumerate(series):
    vals = [rec[key] for key, _ in features]
    ax.bar(
        xs + (i - 0.5) * width, vals, width=width * 0.92,
        color=rec['color'], zorder=3, label=name)
    for key, val in zip((k for k, _ in features), vals):
      csv_rows.append({
          'figure': 'feature_shuffle',
          'panel': 'drop',
          'series': name,
          'feature': key,
          'accuracy_drop': val,
          'baseline_accuracy': rec['baseline'],
          'n_pairs': rec['n_pairs'],
      })
  ax.axhline(0.0, color=REF, lw=0.6, zorder=1)
  _style_ax(ax)
  ax.set_xticks(xs)
  ax.set_xticklabels(
      [r'Mechanism' + '\n' + r'$xy$',
       r'Mechanism' + '\n' + r'$z$',
       'Hand',
       'Action'],
      fontsize=7.2)
  ax.set_ylabel('Retrieval accuracy drop')
  ax.set_ylim(0.0, 0.27)
  ax.legend(loc='upper right', handlelength=1.3, borderaxespad=0.15)
  _panel_label(ax, 'A')
  for i, (name, rec) in enumerate(series):
    x = 0 + (i - 0.5) * width
    y = rec['mech_xy']
    _annotate_bar(ax, x, y, f'{y:.2f}')

  ax = axes[1]
  # Group: Δxy π, Δxy press, Δz π, Δz press — two tasks
  cats = [
      r'$\Delta xy$  $\pi$',
      r'$\Delta xy$  press',
      r'$\Delta z$  $\pi$',
      r'$\Delta z$  press',
  ]
  keys = ['d_xy_pi', 'd_xy_press', 'd_z_pi', 'd_z_press']
  xs = np.arange(len(cats))
  for i, (name, rec) in enumerate(series):
    vals = [1000.0 * rec[k] for k in keys]  # metres → mm
    xpos = xs + (i - 0.5) * width
    ax.bar(
        xpos, vals, width=width * 0.92,
        color=rec['color'], zorder=3, label=name)
    for x, val in zip(xpos, vals):
      if val < 0.15:
        ax.text(
            x, 0.12, '0', ha='center', va='bottom', fontsize=6.6,
            color=rec['color'], fontweight='semibold')
    for key, val in zip(keys, vals):
      csv_rows.append({
          'figure': 'feature_shuffle',
          'panel': 'one_step_mm',
          'series': name,
          'feature': key,
          'delta_mm': val,
      })
  _style_ax(ax)
  ax.set_xticks(xs)
  ax.set_xticklabels(cats, fontsize=7.0)
  ax.set_ylabel('One-step |Δ| at hover (mm)')
  ax.set_ylim(0.0, 5.6)
  _panel_label(ax, 'B')
  ax.legend(loc='upper right', handlelength=1.3, borderaxespad=0.15)

  fig.savefig(out_pdf)
  fig.savefig(out_png)
  plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

def fig_inject(series, out_pdf: Path, out_png: Path, csv_rows: list):
  fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.35), sharex=True)
  fig.subplots_adjust(
      left=0.10, right=0.99, top=0.88, bottom=0.12, hspace=0.18)
  inject_at = series['inject']['inject_at'] or 250500
  names = [
      ('handle', 'Handle', HANDLE, dict(lw=1.5, ls='-')),
      ('inject', 'Handle + 20% inject', INJECT, dict(lw=1.7, ls='-')),
      ('push', 'Push', PUSH, dict(lw=1.5, ls='-')),
  ]

  ax = axes[0]
  ax.axvline(inject_at / 1000.0, color=REF, lw=0.8, ls=(0, (3.2, 2.4)), zorder=1)
  for key, lab, color, style in names:
    rec = series[key]
    x = rec['eval_steps'] / 1000.0
    y = rec['eval_success']
    ax.plot(x, y, color=color, label=lab, zorder=3, **style)
    ax.scatter(x, y, s=12, color=color, zorder=4, linewidths=0)
    for step, val in zip(rec['eval_steps'], rec['eval_success']):
      csv_rows.append({
          'figure': 'inject',
          'panel': 'eval',
          'series': key,
          'env_steps': int(step),
          'success': float(val),
      })
  _style_ax(ax)
  ax.set_ylim(-0.04, 1.08)
  ax.set_yticks([0.0, 0.5, 1.0])
  ax.set_ylabel('Eval success')
  _panel_label(ax, 'A')
  ax.text(
      inject_at / 1000.0, 1.0, '  20% inject',
      fontsize=7.0, color=INK_MUTED, ha='left', va='bottom',
      rotation=90, clip_on=False)

  ax = axes[1]
  ax.axvline(inject_at / 1000.0, color=REF, lw=0.8, ls=(0, (3.2, 2.4)), zorder=1)
  ax.axhline(0.20, color=INJECT, lw=0.7, ls=(0, (1.6, 1.6)), alpha=0.7, zorder=1)
  for key, lab, color, style in names:
    rec = series[key]
    her = rec['her']
    x = her['steps'] / 1000.0
    y = her['succ']
    ax.plot(x, y, color=color, label=lab, zorder=3, lw=style['lw'], ls=style['ls'])
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
      990, 0.205, 'inject target',
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
  fig_inject(
      inject_series,
      out_dir / 'fig_task5_success_inject.pdf',
      out_dir / 'fig_task5_success_inject.png',
      csv_rows)
  _write_csv(csv_dir / 'fig_task5_appendix.csv', csv_rows)
  print('Wrote three appendix figures to', out_dir)


if __name__ == '__main__':
  main()
