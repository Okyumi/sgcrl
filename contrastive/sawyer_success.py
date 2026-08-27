"""Dependency-light sparse-success adapter for custom Sawyer wrappers."""
from __future__ import annotations

import math


SUCCESS_MODES = ('corrected', 'legacy_distance', 'task_axis', 'native_info')


def synchronize_simulator_after_reset(environment) -> None:
  """Refresh site/body positions after a reset mutates MuJoCo joint state.

  MetaWorld's Window-Close V2 reset writes ``window_slide`` directly and then
  returns an observation without forwarding the simulator. Under mujoco-py,
  site positions therefore still describe the pre-write state until the first
  environment step. This helper makes a reset observation and the first
  post-reset transition refer to the same physical state.

  The failure is explicit because silently skipping the forward pass would
  recreate the observation defect on an unsupported simulator adapter.
  """
  simulator = getattr(environment, 'sim', None)
  forward = getattr(simulator, 'forward', None)
  if not callable(forward):
    raise RuntimeError(
        'Sawyer reset synchronization requires environment.sim.forward().')
  forward()


def set_success_mode(environment, success_mode: str) -> None:
  """Validate and attach a versioned Sawyer success contract."""
  if success_mode not in SUCCESS_MODES:
    raise ValueError(
        f'Unknown Sawyer success mode {success_mode!r}; expected one of '
        f'{SUCCESS_MODES}.')
  environment._sawyer_success_mode = success_mode


def task_axis_sparse_reward(
    environment,
    mechanism_position,
    goal,
    *,
    axis: int,
    threshold: float,
):
  """Return the fast task-axis reward, or ``None`` in other modes.

  The Task-5/Task-8 wrappers already expose the mechanism position and goal,
  so this is one scalar comparison with no native-reward query, observation
  reconstruction, or additional simulator step.
  """
  mode = getattr(environment, '_sawyer_success_mode', 'corrected')
  if mode not in ('corrected', 'task_axis'):
    return None
  distance = abs(
      float(mechanism_position[axis]) - float(goal[axis]))
  success = float(distance <= float(threshold))
  return success, {
      'success': success,
      'success_axis_distance': distance,
      'success_axis_index': int(axis),
      'success_threshold': float(threshold),
      'wrapper_success_mode': mode,
  }


def stick_pull_sparse_reward(
    environment,
    handle_position,
    goal,
    stick_end_position,
    *,
    threshold: float = 0.12,
):
  """Return corrected Stick-Pull success, or ``None`` in other modes.

  MetaWorld requires both the handle to reach the target and the far end of
  the stick to remain inserted through the handle.  The historical wrapper
  checked only handle distance, allowing a policy to receive success by
  pushing the object directly.  This helper mirrors the official geometric
  insertion predicate without changing the observation or taking a simulator
  step.
  """
  mode = getattr(environment, '_sawyer_success_mode', 'corrected')
  if mode != 'corrected':
    return None
  handle = tuple(float(value) for value in handle_position)
  target = tuple(float(value) for value in goal)
  stick_end = tuple(float(value) for value in stick_end_position)
  if not (len(handle) == len(target) == len(stick_end) == 3):
    raise ValueError(
        'Stick-Pull success requires 3-D handle, target, and stick-end '
        'positions.')
  target_distance = math.dist(handle, target)
  inserted = (
      stick_end[0] >= handle[0]
      and abs(stick_end[1] - handle[1]) <= 0.040
      and abs(stick_end[2] - handle[2]) <= 0.060)
  success = float(target_distance <= float(threshold) and inserted)
  return success, {
      'success': success,
      'handle_target_distance': target_distance,
      'stick_is_inserted': float(inserted),
      'success_threshold': float(threshold),
      'wrapper_success_mode': mode,
  }


def _metaworld_parent(environment):
  """Return the native MetaWorld task class below a local subclass."""
  for parent in type(environment).__mro__[1:]:
    if (parent.__module__.startswith('metaworld.')
        and hasattr(parent, '_get_obs')
        and hasattr(parent, 'evaluate_state')):
      return parent
  return None


def _copy_if_possible(value):
  return value.copy() if hasattr(value, 'copy') else value


def native_observation(environment, native_parent=None):
  """Return MetaWorld's V2 observation without changing wrapper semantics.

  The MetaWorld release used by this project infers ``isV2`` from the runtime
  class name. Local subclasses such as ``SawyerWindowClose`` therefore inherit
  a V2 task but are incorrectly marked as V1. Calling the parent ``_get_obs``
  directly would then construct the V1 layout. Build the V2 observation under
  temporary bookkeeping values and retain a separate native frame-stack state
  instead of changing the historical wrapper's transition path.
  """
  native_parent = native_parent or _metaworld_parent(environment)
  if native_parent is None:
    return environment._get_obs()
  if 'V2' not in native_parent.__name__:
    return native_parent._get_obs(environment)
  if not hasattr(environment, 'isV2') or bool(environment.isV2):
    return native_parent._get_obs(environment)

  required = ('_obs_obj_max_len', '_obs_obj_possible_lens', '_prev_obs')
  if not all(hasattr(environment, name) for name in required):
    return native_parent._get_obs(environment)

  old_is_v2 = environment.isV2
  old_max_len = environment._obs_obj_max_len
  old_possible_lens = environment._obs_obj_possible_lens
  old_prev_obs = environment._prev_obs
  try:
    environment.isV2 = True
    environment._obs_obj_max_len = 14
    environment._obs_obj_possible_lens = (6, 14)
    current = environment._get_curr_obs_combined_no_goal()
    native_prev = getattr(environment, '_sawyer_native_v2_prev_obs', None)
    path_length = int(getattr(environment, 'curr_path_length', 0))
    previous_path_length = int(getattr(
        environment, '_sawyer_native_v2_path_length', -1))
    if (native_prev is None
        or len(native_prev) != len(current)
        or path_length < previous_path_length
        or path_length == 0):
      native_prev = _copy_if_possible(current)
    environment._prev_obs = native_prev
    observation = native_parent._get_obs(environment)
    environment._sawyer_native_v2_prev_obs = _copy_if_possible(
        environment._prev_obs)
    environment._sawyer_native_v2_path_length = path_length
    return _copy_if_possible(observation)
  finally:
    environment.isV2 = old_is_v2
    environment._obs_obj_max_len = old_max_len
    environment._obs_obj_possible_lens = old_possible_lens
    environment._prev_obs = old_prev_obs


def _evaluate_state_after_step(environment, action, native_result):
  """Recover MetaWorld reward/info when its parent ``step`` returns nothing."""
  if action is None:
    raise RuntimeError(
        'MetaWorld returned a non-standard step result and no action was '
        'provided for the evaluate_state fallback.')
  native_parent = _metaworld_parent(environment)
  if native_parent is not None:
    observation = native_observation(environment, native_parent)
    evaluation = native_parent.evaluate_state(
        environment, observation, action)
    evaluation_source = native_parent.__name__
  else:
    evaluate_state = getattr(environment, 'evaluate_state', None)
    if not callable(evaluate_state):
      raise RuntimeError(
          'MetaWorld returned a non-standard step result and the environment '
          'does not expose evaluate_state().')
    observation = environment._get_obs()
    evaluation = evaluate_state(observation, action)
    evaluation_source = type(environment).__name__
  if not isinstance(evaluation, tuple) or len(evaluation) != 2:
    raise RuntimeError(
        f'{evaluation_source}.evaluate_state() must return (reward, info); got '
        f'{type(evaluation).__name__}.')
  native_reward, native_info = evaluation
  result_type = type(native_result).__name__
  result_length = (len(native_result)
                   if hasattr(native_result, '__len__') else -1)
  return (native_reward, native_info, False, False,
          f'evaluate_state_fallback:{evaluation_source}:'
          f'{result_type}:{result_length}')


def native_sparse_transition(environment, native_result, action=None):
  """Return a native sparse transition, or ``None`` for local semantics.

  MetaWorld releases in this project exist behind both the legacy Gym
  four-item API and the Gymnasium-style five-item API.  The custom wrapper
  continues to expose the four-item API expected by Acme, while retaining the
  native termination fields in ``info`` for auditing.
  """
  mode = getattr(environment, '_sawyer_success_mode', 'corrected')
  if mode in ('corrected', 'legacy_distance', 'task_axis'):
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
  native_parent = _metaworld_parent(environment)
  needs_v2_reevaluation = (
      native_parent is not None
      and 'V2' in native_parent.__name__
      and hasattr(environment, 'isV2')
      and not bool(environment.isV2)
      and not native_step_api.startswith('evaluate_state_fallback:'))
  if needs_v2_reevaluation:
    if action is None:
      raise RuntimeError(
          'A misclassified MetaWorld V2 wrapper requires the executed action '
          'to reconstruct authoritative reward and success.')
    observation = native_observation(environment, native_parent)
    native_reward, native_info = native_parent.evaluate_state(
        environment, observation, action)
    native_step_api += ':native_v2_reevaluated'
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
