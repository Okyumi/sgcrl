#!/usr/bin/env python3
"""Tests for Task-5 appendix figure loaders and render."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def _load_plot_module():
  path = REPO_ROOT / 'scripts' / 'plot_task5_appendix_figs.py'
  spec = importlib.util.spec_from_file_location('plot_task5_appendix_figs', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


_plot = _load_plot_module()
load_action_advice_rows = _plot.load_action_advice_rows
load_feature_rows = _plot.load_feature_rows
load_inject_series = _plot.load_inject_series
parse_eval_log = _plot.parse_eval_log
parse_her_log = _plot.parse_her_log
parse_handle_late_from_stdout = _plot.parse_handle_late_from_stdout


def test_parse_eval_and_her_snippet():
  text = """
  [eval @ 50100] success=0.0% return=0.0
  [her phase @ 1500] succ=0.00 hover=0.11 prog=0.00 far=0.48 succ|prog=0.00
  [eval @ 250500] success=20.0% return=23.1
  [success inject @ 250500] n=50100 target=50100
  [eval @ 300600] success=10.0% return=10.3
  Task 0 training complete (990000 env steps, 6600 episodes).
    Task 0 [sawyer_handle_press_side]: 0.0%
    Mean success: 0.0%
"""
  steps, vals, inject_at = parse_eval_log(text)
  assert inject_at == 250500
  assert list(steps) == [50100, 250500, 300600, 990000]
  np.testing.assert_allclose(vals, [0.0, 0.20, 0.10, 0.0])
  her = parse_her_log(text)
  assert her['steps'][0] == 1500
  assert her['succ'][0] == 0.0
  assert her['succ_or_prog'][0] == 0.0


def test_probe_row_reads_measured_json():
  rows = load_action_advice_rows()
  by_label = {r['label']: r for r in rows}
  h = by_label['Handle 100k']
  p = by_label['Push 250k']
  late = by_label['Handle 952k']
  assert abs(h['gap'] - (-0.1304658282119616)) < 1e-9
  assert abs(h['ratio'] - 0.07541861600152262) < 1e-9
  assert abs(p['gap'] - 2.03742316365242) < 1e-9
  assert abs(p['ratio'] - 1.9498053081888769) < 1e-9
  assert late['n_hover'] == 32
  assert abs(late['gap'] - 0.3809514045715332) < 1e-9
  assert abs(late['rank'] - 3.03125) < 1e-9


def test_feature_drops_match_json():
  rows = load_feature_rows()
  h = rows['Handle 100k']
  p = rows['Push 250k']
  assert abs(h['mech_xy'] - 0.171875) < 1e-12
  assert abs(h['mech_z'] - 0.02734375) < 1e-12
  assert abs(h['action'] - 0.015625) < 1e-12
  assert abs(p['mech_xy'] - 0.19921875) < 1e-12
  assert h['d_xy_pi'] == 0.0
  assert h['d_xy_press'] == 0.0


def test_inject_series_from_logs():
  series = load_inject_series()
  assert series['inject']['inject_at'] == 250500
  inj_eval = dict(zip(
      series['inject']['eval_steps'].tolist(),
      series['inject']['eval_success'].tolist()))
  assert inj_eval[250500] == 0.20
  assert inj_eval[300600] == 0.10
  assert inj_eval[990000] == 0.0
  assert series['push']['eval_success'][-1] == 0.90
  her = series['inject']['her']
  at_249 = her['succ'][her['steps'] == 249000][0]
  at_298 = her['succ'][her['steps'] == 298500][0]
  assert abs(at_249 - 0.03) < 1e-9
  assert abs(at_298 - 0.15) < 1e-9


def test_handle_late_parser_direct():
  row = parse_handle_late_from_stdout(
      REPO_ROOT / 'logs' / 'jubail_task5_action_advice' / 'runs' /
      '17901349_0.out')
  assert row['std_s'] > row['std_a']
  assert row['ratio'] < 0.05


def main():
  test_parse_eval_and_her_snippet()
  test_probe_row_reads_measured_json()
  test_feature_drops_match_json()
  test_inject_series_from_logs()
  test_handle_late_parser_direct()
  print('task5 appendix figure tests passed')


if __name__ == '__main__':
  main()
