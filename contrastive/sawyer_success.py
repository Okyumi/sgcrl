"""Dependency-light sparse-success adapter for custom Sawyer wrappers."""
from __future__ import annotations


SUCCESS_MODES = ('legacy_distance', 'native_info')


def set_success_mode(environment, success_mode: str) -> None:
  """Validate and attach a versioned Sawyer success contract."""
  if success_mode not in SUCCESS_MODES:
    raise ValueError(
        f'Unknown Sawyer success mode {success_mode!r}; expected one of '
        f'{SUCCESS_MODES}.')
  environment._sawyer_success_mode = success_mode


def native_sparse_transition(environment, native_result):
  """Return a native sparse transition, or ``None`` for legacy semantics."""
  mode = getattr(environment, '_sawyer_success_mode', 'legacy_distance')
  if mode == 'legacy_distance':
    return None
  if mode != 'native_info':
    raise ValueError(f'Invalid Sawyer success mode {mode!r}.')
  if not isinstance(native_result, tuple) or len(native_result) != 4:
    raise RuntimeError(
        'MetaWorld step must return (observation, reward, done, info).')
  _, native_reward, _, native_info = native_result
  native_info = dict(native_info or {})
  if 'success' not in native_info:
    raise RuntimeError(
        'MetaWorld step info has no success key; cannot use native_info mode.')
  success = float(native_info['success'])
  if success not in (0.0, 1.0):
    raise RuntimeError(f'MetaWorld success must be binary; got {success!r}.')
  native_info['native_reward'] = float(native_reward)
  native_info['wrapper_success_mode'] = 'native_info'
  # The outer StepLimitWrapper remains the sole episode-boundary authority.
  return environment._get_obs(), success, False, native_info
