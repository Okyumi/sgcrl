"""Dependency-light checks for DCC-SAC and AC-DCC wiring."""
import ast
from pathlib import Path

import numpy as np

from contrastive.dcc_sac_math import (
    normalized_action_correction,
    shortcut_retention,
    stable_q_gate,
)


ROOT = Path(__file__).resolve().parents[1]


def test_normalized_action_correction_is_positive_affine_invariant():
  candidates = np.array([[1.0, 10.0], [2.0, 12.0], [3.0, 14.0]])
  selected = np.array([2.5, 13.0])
  expected = normalized_action_correction(selected, candidates)
  transformed = normalized_action_correction(
      7.0 * selected + 4.0, 7.0 * candidates + 4.0)
  np.testing.assert_allclose(expected, transformed)


def test_stable_q_gate_is_closed_until_all_conditions_hold():
  kwargs = dict(
      warmup_updates=100,
      ramp_updates=100,
      td_error_threshold=0.5,
      twin_disagreement_threshold=0.1,
  )
  assert stable_q_gate(99, 0.1, 0.01, **kwargs) == 0.0
  assert stable_q_gate(150, 0.6, 0.01, **kwargs) == 0.0
  assert stable_q_gate(150, 0.1, 0.2, **kwargs) == 0.0
  assert stable_q_gate(150, 0.1, 0.01, **kwargs) == 0.5
  assert stable_q_gate(250, 0.1, 0.01, **kwargs) == 1.0


def test_stable_q_gate_rejects_nonfinite_statistics():
  kwargs = dict(
      warmup_updates=0,
      ramp_updates=1,
      td_error_threshold=0.5,
      twin_disagreement_threshold=0.1,
  )
  assert stable_q_gate(10, np.nan, 0.01, **kwargs) == 0.0
  assert stable_q_gate(10, 0.01, np.inf, **kwargs) == 0.0


def test_shortcut_retention_definition():
  assert shortcut_retention(0.8, 0.6) == 0.75


def test_new_python_files_parse():
  for relative in (
      'contrastive/continual_learning_dcc_sac.py',
      'contrastive/shortcut_diagnostics.py',
      'contrastive/dcc_sac_math.py',
      'contrastive/decomposed_networks.py',
      'run_continual_contrastive.py',
  ):
    ast.parse((ROOT / relative).read_text(), filename=relative)


def test_actor_fusion_and_gradient_separation_are_explicit():
  source = (
      ROOT / 'contrastive/continual_learning_dcc_sac.py').read_text()
  assert "beta_effective = beta_max * jax.lax.stop_gradient(gate)" in source
  assert "center = jax.lax.stop_gradient" in source
  assert "scale = jax.lax.stop_gradient" in source
  assert "dcc_grads" in source
  assert "q_grad" in source
  assert "TD gradients never enter DCC" in source


def test_action_dcc_and_ablation_modes_are_wired():
  source = (
      ROOT / 'contrastive/continual_learning_dcc_sac.py').read_text()
  for mode in (
      'dcc_sac',
      'dcc_sac_separate',
      'action_dcc',
      'action_dcc_sac',
  ):
    assert repr(mode) in source
  assert "self._use_q = hybrid_mode != 'action_dcc'" in source
  assert "q_params = None" in source
  assert "hold (s, achieved_goal(s')) fixed" in source
  runner = (ROOT / 'run_continual_contrastive.py').read_text()
  assert "if critic_mode != 'action_dcc':\n      hybrid_sac_nets =" in runner


def test_runner_uses_canonical_her_for_q_modes():
  source = (ROOT / 'run_continual_contrastive.py').read_text()
  assert "_HER_CRITIC_MODES" in source
  assert "sac_her.her_reward_and_discount" in source
  assert "critic_mode in _HER_CRITIC_MODES" in source


def test_paired_score_avoids_actor_bxb_matrix():
  source = (ROOT / 'contrastive/decomposed_networks.py').read_text()
  assert 'def apply_paired_score(' in source
  learner = (
      ROOT / 'contrastive/continual_learning_dcc_sac.py').read_text()
  assert 'decomp.apply_paired_score(' in learner


def test_evaluation_scope_controls_present():
  source = (ROOT / 'run_continual_contrastive.py').read_text()
  assert "'post_task_eval_scope'" in source
  assert "['all_seen', 'current', 'none']" in source
