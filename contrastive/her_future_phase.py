"""Classify HER-sampled future goals into geometric phase buckets.

Used to test the \"lucky spike\" story: after early success, do actor/critic
updates again see mostly near-object unsolved futures (hover) rather than
success-like futures?

The goal half of a HER-relabelled observation is ``obs_to_goal(future_state)``.
Under ``goal_conditioning_mode=full_state`` this mirrors the state layout, so
the goal vector can be classified with the same hand / object / axis rules as
a state.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


PHASE_SUCCESS = 'success'
PHASE_HOVER = 'hover_unsolved'
PHASE_PROGRESS = 'object_progress'
PHASE_FAR = 'far_reach'
PHASE_OTHER = 'other'
PHASES = (
    PHASE_SUCCESS,
    PHASE_HOVER,
    PHASE_PROGRESS,
    PHASE_FAR,
    PHASE_OTHER,
)


# Geometry families for the continual Sawyer sequence.
# short_contact_mechanism: success needs a brief contact that moves a
#   constrained mechanism; most near-object futures are still unsolved.
# continuous_object_progress: object pose moves smoothly toward a fixed
#   3-D target; futures often encode partial progress.
# multi_phase_contact: grasp / insert / strike with several contacts.
TASK_GEOMETRY: Dict[str, dict] = {
    'sawyer_hammer': {
        'family': 'multi_phase_contact',
        'object_slice': (7, 10),  # nail
        'fixed_object_target': np.array([0.24, 0.74, 0.11], dtype=np.float32),
        'success_threshold': 0.05,
        'interaction_threshold': 0.12,
    },
    'sawyer_push_wall': {
        'family': 'continuous_object_progress',
        'object_slice': (4, 7),
        'fixed_object_target': np.array([0.05, 0.85, 0.015], dtype=np.float32),
        'success_threshold': 0.07,
        'interaction_threshold': 0.09,
        'progress_threshold': 0.15,
    },
    'sawyer_faucet_close': {
        'family': 'short_contact_mechanism',
        'object_slice': (4, 7),
        'fixed_object_target': np.array([-0.14, 0.82, 0.13], dtype=np.float32),
        'success_threshold': 0.07,
        'interaction_threshold': 0.09,
    },
    'sawyer_push_back': {
        'family': 'continuous_object_progress',
        'object_slice': (4, 7),
        'fixed_object_target': np.array([0.06, 0.62, 0.02], dtype=np.float32),
        'success_threshold': 0.07,
        'interaction_threshold': 0.09,
        'progress_threshold': 0.15,
    },
    'sawyer_stick_pull': {
        'family': 'multi_phase_contact',
        'object_slice': (7, 10),  # handle being pulled
        'fixed_object_target': np.array([0.41, 0.54, 0.02], dtype=np.float32),
        'success_threshold': 0.12,
        'interaction_threshold': 0.12,
    },
    'sawyer_handle_press_side': {
        'family': 'short_contact_mechanism',
        'object_slice': (4, 7),
        'axis_index': 6,
        'axis_target': 0.07,
        'axis_threshold': 0.02,
        'interaction_threshold': 0.09,
    },
    'sawyer_push': {
        'family': 'continuous_object_progress',
        'object_slice': (4, 7),
        'fixed_object_target': np.array([0.02, 0.89, 0.02], dtype=np.float32),
        'success_threshold': 0.05,
        'interaction_threshold': 0.09,
        'progress_threshold': 0.15,
    },
    'sawyer_shelf_place': {
        'family': 'multi_phase_contact',
        'object_slice': (4, 7),
        'fixed_object_target': np.array([0.02, 0.89, 0.30], dtype=np.float32),
        'success_threshold': 0.07,
        'interaction_threshold': 0.09,
    },
    'sawyer_window_close': {
        'family': 'short_contact_mechanism',
        'object_slice': (4, 7),
        'axis_index': 5,  # y toward closed
        'axis_target': 0.80,
        'axis_threshold': 0.05,
        'interaction_threshold': 0.09,
    },
    'sawyer_peg_unplug_side': {
        'family': 'short_contact_mechanism',
        'object_slice': (4, 7),
        'fixed_object_target': np.array([0.01, 0.66, 0.13], dtype=np.float32),
        'success_threshold': 0.07,
        'interaction_threshold': 0.09,
    },
}


def geometry_family(env_name: str) -> str:
  if env_name not in TASK_GEOMETRY:
    raise ValueError(f'No HER-phase geometry for {env_name!r}')
  return str(TASK_GEOMETRY[env_name]['family'])


def classify_her_goal(
    goal_or_state: np.ndarray,
    env_name: str,
) -> str:
  """Assign one phase label to a HER future used as a goal."""
  spec = TASK_GEOMETRY[env_name]
  vec = np.asarray(goal_or_state, dtype=np.float32).reshape(-1)
  if vec.shape[0] < 7:
    raise ValueError(f'Expected at least 7-D Sawyer goal/state; got {vec.shape}')
  hand = vec[:3]
  obj_slice = spec['object_slice']
  obj = vec[obj_slice[0]:obj_slice[1]]
  hand_obj = float(np.linalg.norm(hand - obj[:3]))
  interaction_thr = float(spec.get('interaction_threshold', 0.09))

  if 'axis_index' in spec:
    axis_ok = abs(float(vec[int(spec['axis_index'])]) - float(spec['axis_target'])) <= float(
        spec['axis_threshold'])
    if axis_ok and hand_obj <= interaction_thr:
      return PHASE_SUCCESS
    if hand_obj <= interaction_thr and (not axis_ok):
      return PHASE_HOVER
    if hand_obj >= 0.15:
      return PHASE_FAR
    return PHASE_OTHER

  target = np.asarray(spec['fixed_object_target'], dtype=np.float32).reshape(-1)
  obj_dist = float(np.linalg.norm(obj[: target.shape[0]] - target))
  success_thr = float(spec['success_threshold'])
  if obj_dist <= success_thr:
    return PHASE_SUCCESS
  if (spec['family'] == 'continuous_object_progress'
      and obj_dist <= float(spec.get('progress_threshold', 0.15))):
    return PHASE_PROGRESS
  if hand_obj <= interaction_thr:
    return PHASE_HOVER
  if hand_obj >= 0.15:
    return PHASE_FAR
  return PHASE_OTHER


def summarize_her_goal_batch(
    goals: np.ndarray,
    env_name: str,
) -> Dict[str, float]:
  """Return phase fractions for a batch of HER goal vectors."""
  goals = np.asarray(goals, dtype=np.float32)
  if goals.ndim != 2:
    raise ValueError(f'goals must be [B, D]; got {goals.shape}')
  counts = {phase: 0 for phase in PHASES}
  for row in goals:
    counts[classify_her_goal(row, env_name)] += 1
  total = max(int(goals.shape[0]), 1)
  metrics = {
      f'her_phase/frac_{phase}': counts[phase] / total for phase in PHASES}
  metrics['her_phase/n'] = float(goals.shape[0])
  metrics['her_phase/frac_success_or_progress'] = (
      counts[PHASE_SUCCESS] + counts[PHASE_PROGRESS]) / total
  metrics['her_phase/frac_hover_or_far'] = (
      counts[PHASE_HOVER] + counts[PHASE_FAR]) / total
  metrics['her_phase/geometry_family_code'] = float({
      'continuous_object_progress': 0.0,
      'short_contact_mechanism': 1.0,
      'multi_phase_contact': 2.0,
  }[geometry_family(env_name)])
  return metrics


def ema_update(
    ema: Mapping[str, float],
    batch_metrics: Mapping[str, float],
    *,
    decay: float = 0.99,
) -> Dict[str, float]:
  """Exponential moving average over learner-step HER phase fractions."""
  out = dict(ema)
  for key, value in batch_metrics.items():
    if key == 'her_phase/n':
      out[key] = float(value)
      continue
    prev = float(out.get(key, value))
    out[key] = float(decay) * prev + (1.0 - float(decay)) * float(value)
  return out


def continual_sequence_taxonomy() -> Tuple[dict, ...]:
  """Return the audited geometry table for CONTINUAL_TASK_SEQUENCE."""
  rows = []
  for index, name in enumerate((
      'sawyer_hammer',
      'sawyer_push_wall',
      'sawyer_faucet_close',
      'sawyer_push_back',
      'sawyer_stick_pull',
      'sawyer_handle_press_side',
      'sawyer_push',
      'sawyer_shelf_place',
      'sawyer_window_close',
      'sawyer_peg_unplug_side',
  )):
    spec = TASK_GEOMETRY[name]
    rows.append({
        'index': index,
        'env_name': name,
        'family': spec['family'],
        'short_contact_plus_jump_risk': spec['family'] != 'continuous_object_progress',
    })
  return tuple(rows)
