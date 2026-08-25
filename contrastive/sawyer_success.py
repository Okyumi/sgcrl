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
  """Return a native sparse transition, or ``None`` for legacy semantics.

  MetaWorld releases in this project exist behind both the legacy Gym
  four-item API and the Gymnasium-style five-item API.  The custom wrapper
  continues to expose the four-item API expected by Acme, while retaining the
  native termination fields in ``info`` for auditing.
  """
  mode = getattr(environment, '_sawyer_success_mode', 'legacy_distance')
  if mode == 'legacy_distance':
    return None
  if mode != 'native_info':
    raise ValueError(f'Invalid Sawyer success mode {mode!r}.')
  if not isinstance(native_result, tuple) or len(native_result) not in (4, 5):
    raise RuntimeError(
        'MetaWorld step must return either (observation, reward, done, info) '
        'or (observation, reward, terminated, truncated, info).')
  if len(native_result) == 4:
    _, native_reward, native_done, native_info = native_result
    native_terminated = bool(native_done)
    native_truncated = False
    native_step_api = 'gym_4tuple'
  else:
    (_, native_reward, native_terminated, native_truncated,
     native_info) = native_result
    native_terminated = bool(native_terminated)
    native_truncated = bool(native_truncated)
    native_step_api = 'gymnasium_5tuple'
  native_info = dict(native_info or {})
  if 'success' not in native_info:
    raise RuntimeError(
        'MetaWorld step info has no success key; cannot use native_info mode.')
  success = float(native_info['success'])
  if success not in (0.0, 1.0):
    raise RuntimeError(f'MetaWorld success must be binary; got {success!r}.')
  native_info['native_reward'] = float(native_reward)
  native_info['native_terminated'] = native_terminated
  native_info['native_truncated'] = native_truncated
  native_info['native_step_api'] = native_step_api
  native_info['wrapper_success_mode'] = 'native_info'
  # The outer StepLimitWrapper remains the sole episode-boundary authority.
  return environment._get_obs(), success, False, native_info
