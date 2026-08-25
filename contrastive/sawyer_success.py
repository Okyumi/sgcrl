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


def _evaluate_state_after_step(environment, action, native_result):
  """Recover MetaWorld reward/info when its parent ``step`` returns nothing."""
  if action is None:
    raise RuntimeError(
        'MetaWorld returned a non-standard step result and no action was '
        'provided for the evaluate_state fallback.')
  evaluate_state = getattr(environment, 'evaluate_state', None)
  if not callable(evaluate_state):
    raise RuntimeError(
        'MetaWorld returned a non-standard step result and the environment '
        'does not expose evaluate_state().')
  evaluation = evaluate_state(environment._get_obs(), action)
  if not isinstance(evaluation, tuple) or len(evaluation) != 2:
    raise RuntimeError(
        'MetaWorld evaluate_state() must return (reward, info); got '
        f'{type(evaluation).__name__}.')
  native_reward, native_info = evaluation
  result_type = type(native_result).__name__
  result_length = (len(native_result)
                   if hasattr(native_result, '__len__') else -1)
  return (native_reward, native_info, False, False,
          f'evaluate_state_fallback:{result_type}:{result_length}')


def native_sparse_transition(environment, native_result, action=None):
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
  if isinstance(native_result, tuple) and len(native_result) == 4:
    _, native_reward, native_done, native_info = native_result
    native_terminated = bool(native_done)
    native_truncated = False
    native_step_api = 'gym_4tuple'
  elif isinstance(native_result, tuple) and len(native_result) == 5:
    (_, native_reward, native_terminated, native_truncated,
     native_info) = native_result
    native_terminated = bool(native_terminated)
    native_truncated = bool(native_truncated)
    native_step_api = 'gymnasium_5tuple'
  elif (isinstance(native_result, tuple) and len(native_result) == 2
        and isinstance(native_result[1], dict)):
    # Some MetaWorld forks expose the result of evaluate_state directly.
    native_reward, native_info = native_result
    native_terminated = False
    native_truncated = False
    native_step_api = 'metaworld_reward_info_2tuple'
  else:
    (native_reward, native_info, native_terminated, native_truncated,
     native_step_api) = _evaluate_state_after_step(
         environment, action, native_result)
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
