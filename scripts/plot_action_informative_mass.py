#!/usr/bin/env python3
"""Publication figure: action-informative occupancy for Tasks 5, 8, 6.

Raw critic Δ_f is not comparable across tasks: the on-policy scale of
f(s, a_π, g) differs by an order of magnitude. The plotted critic proxy is
therefore the scale-free ratio

    Δ_f(s, g) / σ_s,    σ_s = std of f(s, a_π, g) on the same rollouts.

Neither this ratio nor the 9 cm contact flag is D_TV of futures.
Mathematical definitions belong in the caption, not on the axes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from contrastive.action_informative_mass import (  # noqa: E402
    bootstrap_survival,
    default_eps_grid,
    relative_delta,
    state_score_scale,
    survival_curve,
    time_bin_mean,
)

# Okabe–Ito (colorblind-safe). Warm hues for the two latch failures,
# blue for the solved continuous-push comparison.
FIGURE_STYLE = {
    'sawyer_handle_press_side': {
        'color': '#D55E00',
        'label': 'Handle press',
    },
    'sawyer_window_close': {
        'color': '#CC79A7',
        'label': 'Window close',
    },
    'sawyer_push': {
        'color': '#0072B2',
        'label': 'Push',
    },
}
TASK_ORDER = [
    'sawyer_handle_press_side',
    'sawyer_window_close',
    'sawyer_push',
]

INK = '#111827'
INK_MUTED = '#6B7280'
SPINE = '#D1D5DB'
GRID = '#EEF2F6'
REF = '#9CA3AF'

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
    'legend.fontsize': 8.5,
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
    'lines.solid_joinstyle': 'round',
})


def _load_records(npz_paths):
  records = []
  for path in npz_paths:
    path = Path(path)
    data = dict(np.load(path, allow_pickle=True))
    meta_path = path.with_suffix('.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    env_name = meta.get('env_name')
    if env_name is None:
      for cand in FIGURE_STYLE:
        if cand.replace('sawyer_', '') in path.name:
          env_name = cand
          break
    if env_name not in FIGURE_STYLE:
      raise ValueError(f'Cannot infer env from {path}')
    steps = int(meta.get('env_steps', data['env_steps'][0]))
    records.append({
        'path': path,
        'env_name': env_name,
        'meta': meta,
        'data': data,
        'steps': steps,
    })
  return records


def _prepare(rec):
  data = rec['data']
  f_pi = np.asarray(data['f_pi'], dtype=np.float64)
  delta = np.asarray(data['delta_f_task'], dtype=np.float64)
  rec = dict(rec)
  rec['sigma_s'] = state_score_scale(f_pi)
  rec['rel_delta'] = relative_delta(delta, f_pi)
  rec['rel_g010'] = relative_delta(
      np.asarray(data['delta_f_gauss_0.10'], dtype=np.float64), f_pi)
  rec['rel_g025'] = relative_delta(
      np.asarray(data['delta_f_gauss_0.25'], dtype=np.float64), f_pi)
  rec['delta'] = delta
  rec['t_frac'] = np.asarray(data['t_frac'], dtype=np.float64)
  rec['contact'] = np.asarray(data['contact'], dtype=np.float64)
  rec['f_pi'] = f_pi
  return rec


def _shared_rel_eps(records):
  hi = 1e-6
  for rec in records:
    vals = rec['rel_delta']
    if vals.size:
      hi = max(hi, float(np.nanpercentile(vals, 99)))
  return default_eps_grid(hi, n=81)


def _style_ax(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.yaxis.grid(True, color=GRID, lw=0.65, zorder=0)
  ax.set_axisbelow(True)
  ax.tick_params(length=2.6, width=0.6, pad=1.8, direction='out')
  ax.xaxis.set_tick_params(which='both', top=False)
  ax.yaxis.set_tick_params(which='both', right=False)


def _panel_label(ax, letter):
  ax.text(
      0.0, 1.04, letter, transform=ax.transAxes, fontsize=11,
      fontweight='bold', color=INK, ha='left', va='bottom', clip_on=False)


def _survival_at(eps, surv, threshold=1.0):
  eps = np.asarray(eps, dtype=np.float64)
  surv = np.asarray(surv, dtype=np.float64)
  if eps.size == 0 or not np.isfinite(surv).any():
    return float('nan')
  if threshold < float(eps.min()) or threshold > float(eps.max()):
    return float('nan')
  return float(np.interp(threshold, eps, surv))


def plot(records, out_pdf: Path, out_png: Path, out_csv: Path):
  by_env = {}
  for rec in records:
    env = rec['env_name']
    prev = by_env.get(env)
    if prev is None or rec['steps'] < prev['steps']:
      by_env[env] = rec
  chosen = [_prepare(by_env[e]) for e in TASK_ORDER if e in by_env]
  if len(chosen) < 2:
    raise ValueError('Need at least two tasks to plot')

  eps = _shared_rel_eps(chosen)
  rng = np.random.default_rng(0)

  fig = plt.figure(figsize=(7.16, 2.22))
  gs = fig.add_gridspec(
      1, 3, width_ratios=[1.0, 1.0, 1.08],
      left=0.052, right=0.995, top=0.78, bottom=0.22,
      wspace=0.28)
  ax_a = fig.add_subplot(gs[0, 0])
  ax_b = fig.add_subplot(gs[0, 1])
  ax_c = fig.add_subplot(gs[0, 2])

  csv_rows = []
  plotted = {}
  task_handles = []
  c_marks = []
  for rec in chosen:
    env = rec['env_name']
    style = FIGURE_STYLE[env]
    color = style['color']
    t_frac = rec['t_frac']
    centers, c_mean, c_se, _ = time_bin_mean(t_frac, rec['contact'], n_bins=15)
    _, r_mean, r_se, _ = time_bin_mean(t_frac, rec['rel_delta'], n_bins=15)
    _, d_mean, d_se, _ = time_bin_mean(t_frac, rec['delta'], n_bins=15)

    if env == 'sawyer_push':
      ax_a.fill_between(
          centers, 0.0, c_mean, color=color, alpha=0.08, lw=0, zorder=1)
    h_a, = ax_a.plot(centers, c_mean, color=color, lw=1.95, zorder=3)
    task_handles.append(h_a)

    ax_b.plot(centers, r_mean, color=color, lw=1.95, zorder=3)
    ax_b.fill_between(
        centers, r_mean - r_se, r_mean + r_se, color=color, alpha=0.14, lw=0,
        zorder=2)

    surv, lo, hi = bootstrap_survival(rec['rel_delta'], eps, rng)
    ax_c.plot(eps, surv, color=color, lw=2.0, zorder=3)
    ax_c.fill_between(eps, lo, hi, color=color, alpha=0.12, lw=0, zorder=2)
    surv025 = survival_curve(rec['rel_g025'], eps)
    ax_c.plot(
        eps, surv025, color=color, lw=1.15, ls=(0, (1.15, 1.35)),
        alpha=0.95, zorder=3)
    y_at_one = _survival_at(eps, surv, 1.0)
    c_marks.append((env, color, y_at_one))

    plotted[env] = {
        't_centers': centers,
        'contact_mean': c_mean,
        'contact_se': c_se,
        'rel_mean': r_mean,
        'rel_se': r_se,
        'raw_mean': d_mean,
        'raw_se': d_se,
        'eps': eps,
        'survival_rel': surv,
        'survival_rel_lo': lo,
        'survival_rel_hi': hi,
        'survival_rel_gauss025': surv025,
        'survival_rel_gauss010': survival_curve(rec['rel_g010'], eps),
        'sigma_s': rec['sigma_s'],
        'contact_mass': float(np.nanmean(rec['contact'])),
        'rel_mean_all': float(np.nanmean(rec['rel_delta'])),
        'raw_mean_all': float(np.nanmean(rec['delta'])),
        'survival_at_eps1': y_at_one,
    }
    for t, cm, cse, rm, rse, dm, dse in zip(
        centers, c_mean, c_se, r_mean, r_se, d_mean, d_se):
      csv_rows.append({
          'panel': 'A', 'env': env, 't_frac': float(t),
          'contact_mean': float(cm), 'contact_se': float(cse),
          'rel_delta_mean': float(rm), 'rel_delta_se': float(rse),
          'delta_f_mean': float(dm), 'delta_f_se': float(dse),
          'sigma_s': rec['sigma_s'], 'env_steps': rec['steps'],
      })
    for e, s, slo, shi, s10, s25 in zip(
        eps, surv, lo, hi,
        plotted[env]['survival_rel_gauss010'],
        surv025):
      csv_rows.append({
          'panel': 'B', 'env': env, 'epsilon': float(e),
          'survival_rel': float(s), 'survival_rel_lo': float(slo),
          'survival_rel_hi': float(shi),
          'survival_rel_gauss010': float(s10),
          'survival_rel_gauss025': float(s25),
          'sigma_s': rec['sigma_s'], 'env_steps': rec['steps'],
      })

  _panel_label(ax_a, 'A')
  ax_a.set_ylabel('Interaction occupancy')
  ax_a.set_xlabel('Episode progress')
  ax_a.set_xlim(0, 1)
  ax_a.set_ylim(0.0, 1.05)
  ax_a.set_xticks([0.0, 0.5, 1.0])
  ax_a.set_yticks([0.0, 0.5, 1.0])
  _style_ax(ax_a)

  _panel_label(ax_b, 'B')
  ax_b.axhline(1.0, color=REF, lw=0.7, ls=(0, (3.2, 2.4)), zorder=1)
  ax_b.set_ylabel('Normalized critic\nsensitivity')
  ax_b.set_xlabel('Episode progress')
  ax_b.set_xlim(0, 1)
  ax_b.set_ylim(0.0, 3.45)
  ax_b.set_xticks([0.0, 0.5, 1.0])
  ax_b.set_yticks([0, 1, 2, 3])
  _style_ax(ax_b)

  _panel_label(ax_c, 'C')
  ax_c.axvline(1.0, color=REF, lw=0.7, ls=(0, (3.2, 2.4)), zorder=1)
  ax_c.set_ylabel('Fraction of states\nabove threshold')
  ax_c.set_xlabel('Sensitivity threshold')
  ax_c.set_xlim(0, max(float(eps[-1]), 1.05))
  ax_c.set_ylim(0, 1.12)
  ax_c.set_xticks([0, 1, 2, 3])
  ax_c.set_yticks([0.0, 0.5, 1.0])
  _style_ax(ax_c)

  # Place the ε=1 percentages in empty regions: handle left of the
  # threshold, window/push to the right of their markers.
  text_off = {
      'sawyer_handle_press_side': dict(xytext=(-8, 9), ha='right', va='bottom'),
      'sawyer_window_close': dict(xytext=(8, 10), ha='left', va='bottom'),
      'sawyer_push': dict(xytext=(8, 5), ha='left', va='bottom'),
  }
  halo = [pe.withStroke(linewidth=2.4, foreground='white')]
  for env, color, y_at_one in c_marks:
    if not np.isfinite(y_at_one):
      continue
    ax_c.plot(
        1.0, y_at_one, 'o', ms=4.2, color=color,
        markeredgecolor='white', markeredgewidth=0.55,
        zorder=6, clip_on=False)
    style = text_off.get(env, dict(xytext=(7, 4), ha='left', va='center'))
    ax_c.annotate(
        f'{int(round(100.0 * y_at_one))}%',
        xy=(1.0, y_at_one), xytext=style['xytext'],
        textcoords='offset points', color=color, fontsize=7.5,
        fontweight='semibold', ha=style['ha'], va=style['va'],
        clip_on=False, zorder=7, path_effects=halo)

  task_labels = [FIGURE_STYLE[rec['env_name']]['label'] for rec in chosen]
  fig.legend(
      task_handles, task_labels,
      loc='upper center', ncol=3, frameon=False,
      handlelength=1.7, handletextpad=0.45, columnspacing=1.7,
      borderaxespad=0.0, fontsize=8.5,
      bbox_to_anchor=(0.53, 1.02), bbox_transform=fig.transFigure)

  out_pdf.parent.mkdir(parents=True, exist_ok=True)
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_pdf)
  fig.savefig(out_png)
  plt.close(fig)

  keys = [
      'panel', 'env', 'env_steps', 'sigma_s', 't_frac', 'contact_mean',
      'contact_se', 'rel_delta_mean', 'rel_delta_se', 'delta_f_mean',
      'delta_f_se', 'epsilon', 'survival_rel', 'survival_rel_lo',
      'survival_rel_hi', 'survival_rel_gauss010', 'survival_rel_gauss025',
  ]
  with out_csv.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(csv_rows)

  npz_path = out_csv.with_suffix('.npz')
  packed = {}
  for env, item in plotted.items():
    tag = env.replace('sawyer_', '')
    for key, val in item.items():
      if np.isscalar(val):
        packed[f'{tag}/{key}'] = np.asarray([val], dtype=np.float64)
      else:
        packed[f'{tag}/{key}'] = np.asarray(val)
  np.savez_compressed(npz_path, **packed)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--npz', nargs='+', required=True)
  parser.add_argument(
      '--out-dir',
      default=str(REPO_ROOT / 'results' / 'img' / 'paper'))
  parser.add_argument(
      '--csv',
      default=str(REPO_ROOT / 'results' / 'data' /
                  'action_informative_mass' /
                  'fig_action_informative_mass.csv'))
  args = parser.parse_args()
  records = _load_records(args.npz)
  plot(
      records,
      Path(args.out_dir) / 'fig_action_informative_mass.pdf',
      Path(args.out_dir) / 'fig_action_informative_mass.png',
      Path(args.csv),
  )
  print('Wrote figure and CSV', flush=True)


if __name__ == '__main__':
  main()
