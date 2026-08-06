"""Dependency-light reference math for DCC-SAC tests and analysis."""
from __future__ import annotations

import numpy as np


def normalized_action_correction(q_value, candidate_values, eps=1e-3,
                                 clip=5.0):
  """Normalize one action value against candidates for the same state/goal."""
  q_value = np.asarray(q_value, dtype=np.float64)
  candidate_values = np.asarray(candidate_values, dtype=np.float64)
  center = np.mean(candidate_values, axis=0)
  scale = np.std(candidate_values, axis=0)
  return np.clip((q_value - center) / np.maximum(scale, eps), -clip, clip)


def stable_q_gate(update_count, td_error_ema, twin_disagreement_ema, *,
                  warmup_updates, ramp_updates, td_error_threshold,
                  twin_disagreement_threshold):
  """Return the scalar DCC-SAC gate used by the actor correction."""
  if not np.isfinite(td_error_ema) or not np.isfinite(
      twin_disagreement_ema):
    return 0.0
  if update_count < warmup_updates:
    return 0.0
  if td_error_ema > td_error_threshold:
    return 0.0
  if twin_disagreement_ema > twin_disagreement_threshold:
    return 0.0
  ramp = max(int(ramp_updates), 1)
  return float(np.clip(
      (update_count - warmup_updates) / ramp, 0.0, 1.0))


def shortcut_retention(reference_accuracy, ablated_accuracy, eps=1e-6):
  """Accuracy retained after shuffling or zeroing the action."""
  return float(ablated_accuracy / max(reference_accuracy, eps))
