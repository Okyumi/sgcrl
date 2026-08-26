"""Utility for loading the goal-conditioned environments.

Continual RL (10-task Meta-World): All Sawyer task wrappers expose a unified
observation space so that state_dim and goal_dim are the same across tasks.
Smaller envs are padded with zeros to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED.
Goal has the exact same semantic meaning as state: goal[i] is the desired
value for the quantity at state[i] (same index = same quantity). See
docs/2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md for index semantics and padding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import gym
import metaworld
import numpy as np
import point_env
from contrastive import sawyer_success

os.environ['SDL_VIDEODRIVER'] = 'dummy'

# -----------------------------------------------------------------------------
# Unified observation dimensions for continual RL (Option A: pad to max).
# All 10 continual Sawyer tasks (sawyer_box excluded per user) return
# observation = [state_padded, goal_padded] with these sizes.
# STATE_DIM_UNIFIED = 11, GOAL_DIM_UNIFIED = 11 (max over tasks; sawyer_box
# would be 11/11; we use 11 for all so a single policy/critic works).
# -----------------------------------------------------------------------------
STATE_DIM_UNIFIED = 11
GOAL_DIM_UNIFIED = 11
FULL_OBS_DIM = STATE_DIM_UNIFIED + GOAL_DIM_UNIFIED  # 22
SAWYER_SUCCESS_MODES = sawyer_success.SUCCESS_MODES


def _set_sawyer_success_mode(environment, success_mode):
  """Select sparse-reward semantics without changing historical defaults."""
  sawyer_success.set_success_mode(environment, success_mode)


def _native_sparse_transition(environment, native_result, action):
  """Convert MetaWorld's authoritative success signal to a sparse reward.

  Returns ``None`` in legacy mode so each wrapper can execute its historical
  hand-written distance predicate unchanged.  In ``native_info`` mode this
  reuses the already-computed parent transition, so it adds no simulator step
  or extra ``evaluate_state`` call.
  """
  return sawyer_success.native_sparse_transition(
      environment, native_result, action=action)


class TaskIDGymWrapper(gym.Wrapper):
  """Appends a one-hot task identifier to both state and goal.

  The raw observation is [state (STATE_DIM_UNIFIED), goal (GOAL_DIM_UNIFIED)].
  After wrapping: [state, task_one_hot, goal, task_one_hot].
  This gives state and goal the same dimensionality (STATE_DIM_UNIFIED + num_tasks)
  so the persistent contrastive critic can condition on the current task.
  """

  def __init__(self, env, task_id, num_tasks):
    super().__init__(env)
    self._task_one_hot = np.zeros(num_tasks, dtype=np.float32)
    self._task_one_hot[task_id] = 1.0
    self._state_dim = STATE_DIM_UNIFIED
    new_dim = FULL_OBS_DIM + 2 * num_tasks
    self.observation_space = gym.spaces.Box(
        low=np.full(new_dim, -np.inf),
        high=np.full(new_dim, np.inf),
        dtype=np.float32)

  def _augment_obs(self, obs):
    state = obs[:self._state_dim]
    goal = obs[self._state_dim:]
    return np.concatenate([state, self._task_one_hot,
                           goal, self._task_one_hot]).astype(np.float32)

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    return self._augment_obs(obs)

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    return self._augment_obs(obs), reward, done, info


def _pad_to_len(arr, length, pad_value=0.0):
  """Pad a 1D array to `length` with `pad_value`. No-op if len(arr) >= length."""
  n = arr.size
  if n >= length:
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr
  return np.concatenate([
      arr.astype(np.float32) if arr.dtype != np.float32 else arr,
      np.full(length - n, pad_value, dtype=np.float32),
  ])


def euler2quat(euler):
  """Convert Euler angles to quaternions."""
  euler = np.asarray(euler, dtype=np.float64)
  assert euler.shape[-1] == 3, 'Invalid shape euler {}'.format(euler)

  ai, aj, ak = euler[Ellipsis, 2] / 2, -euler[Ellipsis, 1] / 2, euler[Ellipsis, 0] / 2
  si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
  ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
  cc, cs = ci * ck, ci * sk
  sc, ss = si * ck, si * sk

  quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
  quat[Ellipsis, 0] = cj * cc + sj * ss
  quat[Ellipsis, 3] = cj * sc - sj * cs
  quat[Ellipsis, 2] = -(cj * ss + sj * cc)
  quat[Ellipsis, 1] = cj * cs - sj * sc
  return quat


def load(env_name, fixed_start_end=None, task_id=None, num_tasks=None,
         sawyer_success_mode='corrected'):
  """Loads the train and eval environments, as well as the obs_dim.

  If task_id and num_tasks are provided, wraps the environment with
  TaskIDGymWrapper to append a one-hot task identifier to both state
  and goal vectors.  obs_dim is then STATE_DIM_UNIFIED + num_tasks.

  ``sawyer_success_mode='corrected'`` uses the task axis for Tasks 5/8 and
  keeps the existing local predicate for every other Sawyer task.
  ``'legacy_distance'`` reproduces historical rewards, while ``'task_axis'``
  is the old Task-5/Task-8-only spelling of the corrected predicate.
  ``'native_info'`` uses the success predicate returned by MetaWorld itself.
  """
  # pylint: disable=invalid-name
  if (sawyer_success_mode == 'task_axis'
      and env_name not in (
          'sawyer_handle_press_side', 'sawyer_window_close')):
    raise ValueError(
        'task_axis success is defined only for sawyer_handle_press_side '
        'and sawyer_window_close.')
  kwargs = {}
  if env_name == 'sawyer_bin':
    CLASS = SawyerBin
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_box':
    CLASS = SawyerBox
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_peg':
    CLASS = SawyerPeg
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_push_back':
    CLASS = SawyerPushBack
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_hammer':
    CLASS = SawyerHammer
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_push_wall':
    CLASS = SawyerPushWall
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_faucet_close':
    CLASS = SawyerFaucetClose
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_stick_pull':
    CLASS = SawyerStickPull
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_handle_press_side':
    CLASS = SawyerHandlePressSide
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_push':
    CLASS = SawyerPush
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_shelf_place':
    CLASS = SawyerShelfPlace
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_window_close':
    CLASS = SawyerWindowClose
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_peg_unplug_side':
    CLASS = SawyerPegUnplugSide
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name.startswith('point_'):
    CLASS = point_env.PointEnv
    kwargs['walls'] = env_name.split('_')[-1]
    kwargs['fixed_start_end'] = fixed_start_end
    if '11x11' in env_name:
      max_episode_steps = 100
    else:
      max_episode_steps = 50
  else:
    raise NotImplementedError('Unsupported environment: %s' % env_name)

  # Disable type checking in line below because different environments have
  # different kwargs, which pytype doesn't reason about.
  gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
  # For continual RL, all Sawyer wrappers use unified obs space (state + goal
  # padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED). obs_dim = state size.
  if env_name.startswith('sawyer_'):
    _set_sawyer_success_mode(gym_env, sawyer_success_mode)
    obs_dim = STATE_DIM_UNIFIED
  else:
    obs_dim = gym_env.observation_space.shape[0] // 2

  # Optionally wrap with task_id one-hot for continual RL.
  if task_id is not None and num_tasks is not None:
    gym_env = TaskIDGymWrapper(gym_env, task_id, num_tasks)
    obs_dim = obs_dim + num_tasks

  return gym_env, obs_dim, max_episode_steps


class SawyerBin(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']):
  """Wrapper for the SawyerBin environment."""

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerBin, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end=fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerBin, self).reset()
    body_id = self.model.body_name2id('bin_goal')
    pos1 = self.sim.data.body_xpos[body_id].copy()
    pos1 += np.random.uniform(-0.05, 0.05, 3)
    pos2 = self._get_pos_objects().copy()
    
    if self._fixed_start_end is not None:
        # Set the goal to be a fixed location
        self._goal = self._fixed_start_end 
    else:
        t = np.random.random()
        # Set the goal to be a uniformly sampled location
        # between the starting and end point
        self._goal = t * pos1 + (1 - t) * pos2
        self._goal[2] = np.random.uniform(0.03, 0.12)
    self._target_pos = self._goal
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerBin, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    obj_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - obj_pos)
    obs = self._get_obs()
    r = float(dist < 0.05)  # Taken from metaworld
    done = False
    info = {}
        
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), block_pos(4-6). Goal [7:14]: hand_above(7-9), gripper(10), block_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    state = np.concatenate((pos_hand, [gripper_distance_apart],
                            self._get_pos_objects()))
    goal = np.concatenate([self._goal + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerBox(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['box-close-v2']):
  """Wrapper for the SawyerBox environment."""

  def __init__(self, fixed_start_end=None):
    self._goal_pos = np.zeros(3)
    self._goal_quat = np.zeros(4)
    super(SawyerBox, self).__init__()
    self._fixed_start_end=fixed_start_end
    self._set_task_called = True
    self._partially_observable = False
    self._freeze_rand_vec = False
    self.reset()

  def reset(self):
    super(SawyerBox, self).reset()
    pos1 = self._target_pos.copy()
    pos2 = self._get_pos_objects().copy()
    
    if self._fixed_start_end is not None:
        # Set the goal to be a fixed location
        self._goal_pos = pos1 
    else:
        # Set the goal to be a uniformly sampled location
        # between the starting and end point
        t = np.random.random()
        self._goal_pos = t * pos1 + (1 - t) * pos2
        
    self._goal_quat = np.array([0.707, 0, 0, 0.707]) # ideal orientation of lid
    self._target_pos = self._goal_pos    
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerBox, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    obj_pos = self._get_pos_objects()
    obj_quat = self._get_quat_objects()
    
    dist_pos = np.linalg.norm(self._goal_pos - obj_pos)
    dist_quat = np.linalg.norm(self._goal_quat - obj_quat)
    
    obs = self._get_obs()
    r = float(dist_pos < 0.08 and dist_quat < 0.08)  # Taken from metaworld
    done = False
    info = {}
    
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:11]: hand(0-2), gripper(3), lid_pos(4-6), lid_quat(7-10). Goal [11:22]: same layout.
    # Box already has state_dim=11, goal_dim=11; padding is no-op (unified dims match).
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    obj_pos = self._get_pos_objects()
    obj_quat = self._get_quat_objects()
    state = np.concatenate((pos_hand, [gripper_distance_apart],
                            obj_pos, obj_quat))
    goal = np.concatenate([self._goal_pos + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal_pos, self._goal_quat])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)

class SawyerPeg(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['peg-insert-side-v2']):
  """Wrapper for the SawyerPeg environment."""

  def __init__(self, fixed_start_end=None):
    self._goal_pos = np.zeros(3)
    super(SawyerPeg, self).__init__()
    self._fixed_start_end=fixed_start_end
    self._set_task_called = True
    self._partially_observable = False
    self._freeze_rand_vec = False
    self.reset()

  def reset(self):
    super(SawyerPeg, self).reset()
    pos1 = self._target_pos.copy()
    pos2 = self._get_site_pos("pegHead")
    
    if self._fixed_start_end is not None:
        # Set the goal to be a fixed location
        self._goal_pos = pos1 
    else:
        # Set the goal to be a uniformly sampled location
        # between the starting and end point
        t = np.random.random()
        self._goal_pos = t * pos1 + (1 - t) * pos2
    self._target_pos = self._goal_pos    
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerPeg, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    obj_head = self._get_site_pos("pegHead")
    
    scale = np.array([1.0, 2.0, 2.0])
    dist_pos = float(np.linalg.norm((obj_head - self._goal_pos) * scale))
       
    r = float(dist_pos < 0.07)  # Taken from metaworld
    done = False
    info = {}
    return self._get_obs(), r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), peg_head_pos(4-6). Goal [7:14]: hand_above(7-9), gripper(10), peg_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    obj_pos_head = self._get_site_pos("pegHead")
    state = np.concatenate((pos_hand, [gripper_distance_apart], obj_pos_head))
    goal = np.concatenate([self._goal_pos + np.array([0.13, 0.0, 0.03]),
                           [0.4], self._goal_pos])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerPushBack(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['push-back-v2']):
  """Wrapper for PushBack: push object to a target position.

  Goal semantics: the desired state is the object at the target position.
  State = hand (3), gripper (1), object position (3) = 7 dims.
  Goal = same structure as state: hand slightly above target (3), gripper (1),
  target position (3). Success when object within 0.07 of target (Meta-World).
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerPushBack, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerPushBack, self).reset()
    # Parent sets _target_pos to the goal position for the object
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerPushBack, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    obj_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - obj_pos)
    obs = self._get_obs()
    r = float(dist < 0.07)  # Meta-World push-back success threshold
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), object_pos(4-6).
    # Goal  [7:14]: hand_above_target(7-9), gripper(10), object_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    obj_pos = self._get_pos_objects()
    state = np.concatenate((
        pos_hand,
        [gripper_distance_apart],
        obj_pos,
    ))
    goal = np.concatenate([
        self._goal + np.array([0.0, 0.0, 0.03]),
        [0.4],
        self._goal,
    ])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerHammer(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['hammer-v2']):
  """Wrapper for Hammer: drive nail to target position.

  Meta-World _get_pos_objects() returns hammer (3) + nail (3) = 6 dims. We include
  both so state is not truncated. State = hand (3), gripper (1), hammer (3),
  nail (3) = 10 dims. Goal = hand above target (3), gripper (1), nail target (3) = 7.
  Success when nail within 0.05 of goal (or joint-based in raw env).
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerHammer, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerHammer, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerHammer, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    pos_objects = self._get_pos_objects()
    nail_pos = pos_objects[3:6] if len(pos_objects) >= 6 else pos_objects[:3]
    dist = np.linalg.norm(self._goal - nail_pos)
    obs = self._get_obs()
    r = float(dist < 0.05)
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:10]: hand(0-2), gripper(3), hammer_pos(4-6), nail_pos(7-9).
    # Goal [0:10]: desired_hand(0-2), desired_gripper(3), desired_hammer_pos(4-6), desired_nail_pos(7-9); then padded to GOAL_DIM_UNIFIED.
    # _get_pos_objects() returns hammer (3) + nail (3); both included so state is not truncated.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    pos_objects = self._get_pos_objects()  # hammer (3) + nail (3)
    hammer_pos = pos_objects[:3]
    nail_pos = pos_objects[3:6] if len(pos_objects) >= 6 else pos_objects[:3]
    state = np.concatenate((pos_hand, [gripper_distance_apart], hammer_pos, nail_pos))
    # Goal mirrors state semantics: desired_hand, desired_gripper, desired_hammer_pos, desired_nail_pos.
    hand_above = self._goal + np.array([0.0, 0.0, 0.03])
    goal = np.concatenate([hand_above, [0.4], self._goal, self._goal])  # 10 dims
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerPushWall(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['push-wall-v2']):
  """Wrapper for PushWall: push object to target against the wall.

  State = hand (3), gripper (1), object position (3) = 7 dims.
  Goal = same structure. Success when object within 0.07 of goal.
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerPushWall, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerPushWall, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerPushWall, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    obj_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - obj_pos)
    obs = self._get_obs()
    r = float(dist < 0.07)
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), object_pos(4-6). Goal [7:14]: hand_above(7-9), gripper(10), object_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    obj_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper_distance_apart], obj_pos))
    goal = np.concatenate([
        self._goal + np.array([0.0, 0.0, 0.03]),
        [0.4],
        self._goal,
    ])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerFaucetClose(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['faucet-close-v2']):
  """Wrapper for FaucetClose: close faucet handle to target position.

  State = hand (3), gripper (1), handle position (3) = 7 dims.
  Goal = same structure. Success when handle within 0.07 of goal.
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerFaucetClose, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerFaucetClose, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerFaucetClose, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    handle_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - handle_pos)
    obs = self._get_obs()
    r = float(dist < 0.07)
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), handle_pos(4-6). Goal [7:14]: hand_above(7-9), gripper(10), handle_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    handle_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper_distance_apart], handle_pos))
    goal = np.concatenate([
        self._goal + np.array([0.0, 0.0, 0.03]),
        [0.4],
        self._goal,
    ])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerStickPull(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['stick-pull-v2']):
  """Wrapper for StickPull: pull object (handle/insertion) to target using stick.

  State = hand (3), gripper (1), stick position (3), handle/insertion (3) = 10 dims.
  _get_pos_objects() returns stick (3) + handle (3); both included so state is not truncated.
  Goal = hand_above (3), gripper (1), handle target (3) = 7 dims. Success when handle within 0.05 of goal.
  Padding: state 10->STATE_DIM_UNIFIED, goal 7->GOAL_DIM_UNIFIED for continual RL.
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerStickPull, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerStickPull, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerStickPull, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    pos_objects = self._get_pos_objects()
    handle_pos = pos_objects[3:6] if len(pos_objects) >= 6 else pos_objects[:3]
    dist = np.linalg.norm(self._goal - handle_pos)
    obs = self._get_obs()
    r = float(dist < 0.12)  # MetaWorld uses 0.12 for stick-pull
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:10]: hand(0-2), gripper(3), stick_pos(4-6), handle_pos(7-9).
    # Goal [0:10]: desired_hand(0-2), desired_gripper(3), desired_stick_pos(4-6), desired_handle_pos(7-9); then padded to GOAL_DIM_UNIFIED.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    pos_objects = self._get_pos_objects()  # stick (3) + handle/insertion (3)
    stick_pos = pos_objects[:3]
    handle_pos = pos_objects[3:6] if len(pos_objects) >= 6 else pos_objects[:3]
    state = np.concatenate((pos_hand, [gripper_distance_apart], stick_pos, handle_pos))
    # Goal mirrors state semantics: desired_hand, desired_gripper, desired_stick_pos, desired_handle_pos.
    hand_above = self._goal + np.array([0.0, 0.0, 0.03])
    goal = np.concatenate([hand_above, [0.4], self._goal, self._goal])  # 10 dims
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerHandlePressSide(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['handle-press-side-v2']):
  """Wrapper for HandlePressSide: press handle to target position.

  State = hand (3), gripper (1), handle position (3) = 7 dims.
  Goal = same structure. Success when handle within 0.02 of goal (TARGET_RADIUS).
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerHandlePressSide, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerHandlePressSide, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerHandlePressSide, self).step(action)
    handle_pos = self._get_pos_objects()
    task_axis = sawyer_success.task_axis_sparse_reward(
        self, handle_pos, self._goal, axis=2, threshold=0.02)
    if task_axis is not None:
      reward, info = task_axis
      return self._get_obs(), reward, False, info
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    dist = np.linalg.norm(self._goal - handle_pos)
    obs = self._get_obs()
    r = float(dist < 0.02)
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), handle_pos(4-6). Goal [7:14]: hand_above(7-9), gripper(10), handle_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    handle_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper_distance_apart], handle_pos))
    goal = np.concatenate([
        self._goal + np.array([0.0, 0.0, 0.03]),
        [0.4],
        self._goal,
    ])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerPush(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['push-v2']):
  """Wrapper for Push: push object to target position.

  State = hand (3), gripper (1), object position (3) = 7 dims.
  Goal = same structure. Success when object within 0.05 of goal.
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerPush, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerPush, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerPush, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    obj_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - obj_pos)
    obs = self._get_obs()
    r = float(dist < 0.05)
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), object_pos(4-6). Goal [7:14]: hand_above(7-9), gripper(10), object_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    obj_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper_distance_apart], obj_pos))
    goal = np.concatenate([
        self._goal + np.array([0.0, 0.0, 0.03]),
        [0.4],
        self._goal,
    ])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerShelfPlace(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['shelf-place-v2']):
  """Wrapper for ShelfPlace: place object on shelf at target position.

  State = hand (3), gripper (1), object position (3) = 7 dims.
  Goal = same structure. Success when object within 0.07 of goal.
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerShelfPlace, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerShelfPlace, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerShelfPlace, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    obj_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - obj_pos)
    obs = self._get_obs()
    r = float(dist < 0.07)
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), object_pos(4-6). Goal [7:14]: hand_above(7-9), gripper(10), object_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    obj_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper_distance_apart], obj_pos))
    goal = np.concatenate([
        self._goal + np.array([0.0, 0.0, 0.03]),
        [0.4],
        self._goal,
    ])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerWindowClose(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['window-close-v2']):
  """Wrapper for WindowClose: close window (handle) to target position.

  State = hand (3), gripper (1), handle position (3) = 7 dims.
  _get_pos_objects() returns site handleCloseStart (3).
  Goal = same structure. Success when handle within 0.05 of goal (TARGET_RADIUS).
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerWindowClose, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerWindowClose, self).reset()
    # MetaWorld Window-Close V2 writes the window-slide qpos after it last
    # queried the handle site. Forward once so the reset observation exposes
    # the actual open-window handle position, not the stale pre-reset site.
    sawyer_success.synchronize_simulator_after_reset(self)
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerWindowClose, self).step(action)
    handle_pos = self._get_pos_objects()
    task_axis = sawyer_success.task_axis_sparse_reward(
        self, handle_pos, self._goal, axis=0, threshold=0.05)
    if task_axis is not None:
      reward, info = task_axis
      return self._get_obs(), reward, False, info
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    dist = np.linalg.norm(self._goal - handle_pos)
    obs = self._get_obs()
    r = float(dist < 0.05)
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), handle_pos(4-6). Goal [7:14]: hand_above(7-9), gripper(10), handle_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    handle_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper_distance_apart], handle_pos))
    goal = np.concatenate([
        self._goal + np.array([0.0, 0.0, 0.03]),
        [0.4],
        self._goal,
    ])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)


class SawyerPegUnplugSide(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['peg-unplug-side-v2']):
  """Wrapper for PegUnplugSide: unplug peg to target position.

  State = hand (3), gripper (1), peg position (3) = 7 dims.
  _get_pos_objects() returns site pegEnd (3).
  Goal = same structure. Success when peg within 0.07 of goal.
  """

  def __init__(self, fixed_start_end=None):
    self._goal = np.zeros(3)
    super(SawyerPegUnplugSide, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerPegUnplugSide, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.array(self._fixed_start_end, dtype=np.float32)
      self._target_pos = self._goal.copy()
    else:
      self._goal = self._target_pos.copy()
    return self._get_obs()

  def step(self, action):
    native_result = super(SawyerPegUnplugSide, self).step(action)
    native_sparse = _native_sparse_transition(self, native_result, action)
    if native_sparse is not None:
      return native_sparse
    peg_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - peg_pos)
    obs = self._get_obs()
    r = float(dist < 0.07)
    done = False
    info = {}
    return obs, r, done, info

  def _get_obs(self):
    # State and goal use the same index semantics (see 2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md).
    # State [0:7]: hand(0-2), gripper(3), peg_pos(4-6). Goal [7:14]: hand_above(7-9), gripper(10), peg_target(11-13).
    # Padding: state and goal padded to STATE_DIM_UNIFIED and GOAL_DIM_UNIFIED for continual RL.
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    peg_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper_distance_apart], peg_pos))
    goal = np.concatenate([
        self._goal + np.array([0.0, 0.0, 0.03]),
        [0.4],
        self._goal,
    ])
    state_padded = _pad_to_len(state, STATE_DIM_UNIFIED)
    goal_padded = _pad_to_len(goal, GOAL_DIM_UNIFIED)
    return np.concatenate([state_padded, goal_padded]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(FULL_OBS_DIM, -np.inf),
        high=np.full(FULL_OBS_DIM, np.inf),
        dtype=np.float32)
