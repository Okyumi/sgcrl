#!/usr/bin/env python3
"""Fetch and plot paper DCC vs Success-BC learning curves for Tasks 4 and 7."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / 'logs' / 'paper_task47_curves'

GROUPS = (
    ('PAPER-DCC-NODYN-10SEED-ablation', 'plain_dcc'),
    ('PAPER-DCC-NODYN-10SEED-success-bc', 'success_bc'),
)
TASKS = (
    (4, 'sawyer_stick_pull'),
    (7, 'sawyer_shelf_place'),
)
HISTORY_KEYS = (
    'evaluator/env_steps',
    'evaluator/success_rate',
    'retention/bc_loss',
    'retention/bc_to_dcc_loss_ratio',
    'retention/weighted_bc_loss',
    'retention/buffer_size',
    'retention/bc_active',
)


def _load_wandb():
  keyfile = Path.home() / '.wandb_api_key'
  if keyfile.exists() and not os.environ.get('WANDB_API_KEY'):
    os.environ['WANDB_API_KEY'] = keyfile.read_text().strip()
  import wandb
  return wandb


def _run_matches(run, task_idx, env_name):
  name = str(run.name or '')
  prefix = f'task{task_idx}_{env_name}_'
  if name.startswith(prefix):
    return True
  cfg = run.config or {}
  if int(cfg.get('task_id', -1)) == task_idx:
    return True
  single = str(cfg.get('single_task') or cfg.get('env_name') or '')
  return single == env_name


def _history_rows(run, want_bc=False):
  rows = []
  print(f'    history {run.name} ...', flush=True)
  try:
    samples = run.history(
        keys=['evaluator/env_steps', 'evaluator/success_rate'],
        pandas=False, samples=20000)
  except Exception as exc:
    print(f'    history failed {run.name}: {exc}', flush=True)
    return rows
  extras = {}
  if want_bc:
    try:
      extra_samples = run.history(
          keys=['evaluator/env_steps', 'retention/bc_loss',
                'retention/bc_to_dcc_loss_ratio'],
          pandas=False, samples=20000)
      for sample in extra_samples or []:
        steps = sample.get('evaluator/env_steps')
        if steps is None:
          continue
        extras[float(steps)] = sample
    except Exception as exc:
      print(f'    bc history skipped {run.name}: {exc}', flush=True)
  for sample in samples or []:
    steps = sample.get('evaluator/env_steps')
    success = sample.get('evaluator/success_rate')
    if steps is None or success is None:
      continue
    extra = extras.get(float(steps), {})
    rows.append({
        'env_steps': float(steps),
        'success_rate': float(success),
        'bc_loss': _maybe_float(extra.get('retention/bc_loss')),
        'bc_to_dcc_ratio': _maybe_float(
            extra.get('retention/bc_to_dcc_loss_ratio')),
        'weighted_bc_loss': None,
        'buffer_size': None,
    })
  rows.sort(key=lambda row: row['env_steps'])
  return rows


def _maybe_float(value):
  if value is None:
    return None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _seed_of(run):
  cfg = run.config or {}
  if 'seed' in cfg:
    return int(cfg['seed'])
  name = str(run.name or '')
  if '_s' in name:
    try:
      return int(name.rsplit('_s', 1)[-1])
    except ValueError:
      return -1
  return -1


def _aggregate(seed_series, bin_size=100_000):
  buckets = defaultdict(list)
  for rows in seed_series:
    for row in rows:
      bucket = int(round(row['env_steps'] / bin_size) * bin_size)
      buckets[bucket].append(row)
  steps = sorted(buckets)
  mean = []
  std = []
  n = []
  bc_ratio_mean = []
  bc_loss_mean = []
  for step in steps:
    vals = np.array([row['success_rate'] for row in buckets[step]],
                    dtype=np.float64)
    mean.append(float(vals.mean()))
    std.append(float(vals.std(ddof=0)) if len(vals) > 1 else 0.0)
    n.append(int(len(vals)))
    ratios = [row['bc_to_dcc_ratio'] for row in buckets[step]
              if row['bc_to_dcc_ratio'] is not None]
    losses = [row['bc_loss'] for row in buckets[step]
              if row['bc_loss'] is not None]
    bc_ratio_mean.append(float(np.mean(ratios)) if ratios else None)
    bc_loss_mean.append(float(np.mean(losses)) if losses else None)
  return {
      'env_steps': steps,
      'mean': mean,
      'std': std,
      'n': n,
      'bc_to_dcc_ratio_mean': bc_ratio_mean,
      'bc_loss_mean': bc_loss_mean,
  }


def fetch(project: str, out_dir: Path):
  wandb = _load_wandb()
  api = wandb.Api(timeout=120)
  payload = {
      'project': project,
      'groups': [g for g, _ in GROUPS],
      'tasks': [{'task_idx': t, 'env': e} for t, e in TASKS],
      'series': {},
  }
  for group, variant in GROUPS:
    print(f'Fetching {group} ...', flush=True)
    runs = list(api.runs(
        project,
        filters={
            'group': group,
            'displayName': {'$regex': r'task[47]_'},
        }))
    matched = []
    for run in runs:
      for task_idx, env_name in TASKS:
        if _run_matches(run, task_idx, env_name):
          matched.append((task_idx, env_name, run))
          break
    print(f'  matched {len(matched)} runs', flush=True)
    best = {}
    for task_idx, env_name, run in matched:
      rows = _history_rows(run, want_bc=False)
      key = (task_idx, env_name, _seed_of(run))
      prev = best.get(key)
      if prev is None or len(rows) > len(prev['rows']):
        best[key] = {
            'run_id': run.id,
            'run_name': run.name,
            'state': run.state,
            'seed': key[2],
            'rows': rows,
        }
      print(
          f'  {run.name} seed={key[2]} state={run.state} '
          f'points={len(rows)}',
          flush=True)
    for task_idx, env_name in TASKS:
      seed_rows = [
          item['rows'] for (t, e, seed), item in sorted(best.items())
          if t == task_idx and e == env_name and item['rows']
      ]
      agg = _aggregate(seed_rows)
      final = []
      for item in best.values():
        if not item['rows']:
          continue
        if item['rows'][-1]['env_steps'] and True:
          pass
      finals = []
      for (t, e, seed), item in sorted(best.items()):
        if t != task_idx or e != env_name or not item['rows']:
          continue
        finals.append(item['rows'][-1]['success_rate'])
      payload['series'][f'{variant}/task{task_idx}'] = {
          'variant': variant,
          'group': group,
          'task_idx': task_idx,
          'env_name': env_name,
          'n_seeds': len(seed_rows),
          'final_mean': float(np.mean(finals)) if finals else None,
          'final_std': float(np.std(finals)) if len(finals) > 1 else 0.0,
          'aggregate': agg,
          'seeds': [
              {
                  'seed': seed,
                  'run_id': item['run_id'],
                  'run_name': item['run_name'],
                  'state': item['state'],
                  'n_points': len(item['rows']),
                  'final_success': (
                      item['rows'][-1]['success_rate'] if item['rows']
                      else None),
              }
              for (t, e, seed), item in sorted(best.items())
              if t == task_idx and e == env_name
          ],
      }
  out_dir.mkdir(parents=True, exist_ok=True)
  json_path = out_dir / 'task47_curves.json'
  json_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
  print(f'Wrote {json_path}', flush=True)
  _write_csv(payload, out_dir / 'task47_curves.csv')
  _write_png(payload, out_dir / 'task47_curves.png')
  return payload


def _write_csv(payload, path: Path):
  lines = ['variant,task_idx,env_name,env_steps,mean_success,std_success,n,'
           'bc_to_dcc_ratio_mean,bc_loss_mean']
  for key, series in payload['series'].items():
    agg = series['aggregate']
    for i, step in enumerate(agg['env_steps']):
      ratio = agg['bc_to_dcc_ratio_mean'][i]
      loss = agg['bc_loss_mean'][i]
      lines.append(
          f"{series['variant']},{series['task_idx']},{series['env_name']},"
          f"{step},{agg['mean'][i]:.6f},{agg['std'][i]:.6f},{agg['n'][i]},"
          f"{'' if ratio is None else f'{ratio:.6f}'},"
          f"{'' if loss is None else f'{loss:.6f}'}")
  path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
  print(f'Wrote {path}', flush=True)


def _write_png(payload, path: Path):
  try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
  except ImportError:
    print('matplotlib not available; skipping PNG', flush=True)
    return
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
  colors = {'plain_dcc': '#2563eb', 'success_bc': '#dc2626'}
  labels = {'plain_dcc': 'plain DCC', 'success_bc': 'DCC + Success-BC'}
  for ax, (task_idx, env_name) in zip(axes, TASKS):
    for variant in ('plain_dcc', 'success_bc'):
      series = payload['series'].get(f'{variant}/task{task_idx}')
      if not series:
        continue
      agg = series['aggregate']
      if not agg['env_steps']:
        continue
      x = np.array(agg['env_steps']) / 1e6
      y = np.array(agg['mean'])
      s = np.array(agg['std'])
      ax.plot(x, y, color=colors[variant], label=labels[variant], lw=2)
      ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1),
                      color=colors[variant], alpha=0.18)
    ax.set_title(f'Task {task_idx}: {env_name.replace("sawyer_", "")}')
    ax.set_xlabel('env steps (M)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
  axes[0].set_ylabel('eval success (mean ± std)')
  axes[1].legend(frameon=False, loc='upper left')
  fig.suptitle(
      'Paper 10-seed DCC vs Success-BC (λ=0.1, native_info, dyn=0)',
      fontsize=11)
  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)
  print(f'Wrote {path}', flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--project', default='nyuad_mmvc/continual_gcrl_paper')
  parser.add_argument('--out-dir', default=str(DEFAULT_OUT))
  args = parser.parse_args()
  fetch(args.project, Path(args.out_dir))


if __name__ == '__main__':
  main()
