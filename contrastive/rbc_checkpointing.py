"""Pure checkpoint-identity helpers for RBC-DCC."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Any


RBC_IDENTITY_FIELDS = (
    'dyn_aux_weight',
    'dyn_aux_after_task0',
    'phi_task_width',
    'phi_task_depth',
    'combine_mode',
    'goal_encoder_mode',
    'bellman_loss_weight',
    'bellman_residual_l2_weight',
    'bellman_discount',
    'bellman_tau',
    'bellman_hidden_dim',
    'her_reward_threshold',
    'step_penalty_reward',
)


def identity_payload(config: Mapping[str, Any]) -> dict:
  """Return the canonical subset that defines an RBC-DCC checkpoint."""
  missing = [field for field in RBC_IDENTITY_FIELDS if field not in config]
  if missing:
    raise ValueError(f'Missing RBC checkpoint identity fields: {missing}')
  return {field: config[field] for field in RBC_IDENTITY_FIELDS}


def fingerprint_payload(payload: Mapping[str, Any], length: int = 12) -> str:
  """Return a stable short hash of an arbitrary identity mapping.

  Used by hybrid modes (DCC-SAC / AC-DCC) whose identity fields are not
  the RBC Bellman set. RBC itself should keep using ``config_fingerprint``.
  """
  encoded = json.dumps(
      dict(payload), sort_keys=True, separators=(',', ':'), default=str)
  return hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:length]


def config_fingerprint(config: Mapping[str, Any], length: int = 12) -> str:
  """Return a stable short hash of every RBC-defining setting."""
  return fingerprint_payload(identity_payload(config), length=length)
