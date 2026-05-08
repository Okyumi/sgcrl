# Why This Codebase Subclasses Metaworld Sawyer Envs Instead of Using Metaworld Directly

This document explains why the SGCRL codebase does **not** use Metaworld’s high-level API “directly,” and instead **subclasses** the low-level Sawyer environments from `metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS`. Each section describes a requirement, why Metaworld “as-is” doesn’t satisfy it, and **where in the code** that requirement is implemented.

---

## 1. Goal-conditioned RL vs Metaworld’s task API

### What the algorithm needs

The algorithm is **goal-conditioned contrastive RL**. It needs:

- A **goal** that is chosen in a controlled way (e.g. sampled every episode or fixed for evaluation).
- Observations of the form **`[state, goal]`** so the agent sees both current state and desired goal.
- Reward and success defined with respect to **that** goal (for training and evaluation).

### What Metaworld “directly” provides

Metaworld’s standard usage is built for **multi-task / meta-RL**:

- You get a **task** (e.g. “put block in this bin”), often set via `set_task()` or by sampling from a task distribution.
- The “goal” is whatever the task defines internally; you don’t get to **resample a new goal every reset** in a way the algorithm controls.
- Observations are typically **task-specific state** (and sometimes an internal goal), not a generic **`[state, goal]`** layout.

So using Metaworld “directly” would mean using its task-based API and observation format, which do **not** match the goal-conditioned interface required here.

### Where this is reflected in the code

The code **subclasses** the low-level V2 env and injects its own goal and observation logic.

**Location:** `env_utils.py` — each Sawyer wrapper inherits from a single Metaworld V2 env.

```66:69:env_utils.py
class SawyerBin(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']):
  """Wrapper for the SawyerBin environment."""
```

Same pattern for `SawyerBox` (line 135–136) and `SawyerPeg` (line 211–212): they subclass `ALL_V2_ENVIRONMENTS['box-close-v2']` and `ALL_V2_ENVIRONMENTS['peg-insert-side-v2']` respectively. The rest of the document shows how these subclasses implement the goal-conditioned interface.

---

## 2. Observation format: `[state, goal]`

### What the algorithm needs

The rest of the stack assumes:

- **`observation = [state, goal]`** with a fixed layout.
- **`obs_dim = observation_space.shape[0] // 2`** so that the first half is state and the second half is goal.

Downstream code (e.g. `ObservationFilterWrapper`, `obs_to_goal_2d`, learning) relies on this layout.

### What the wrappers do

They **override `_get_obs()`** to build exactly that: state (hand, gripper, object state) concatenated with a **goal** vector they define (desired object position, gripper, etc.).

### Code: SawyerBin — building state and goal, then concatenating

**File:** `env_utils.py`

**State part (hand + gripper + object):**

```110:119:env_utils.py
  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    obs = np.concatenate((pos_hand, [gripper_distance_apart],
                          self._get_pos_objects()))
```

**Goal part (ideal state for this goal) and final observation:**

```120:125:env_utils.py
    # the ideal goal state has the block in the blue bin and the gripper slightly 
    # higher than the block center
    goal = np.concatenate([self._goal + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal])

    return np.concatenate([obs, goal]).astype(np.float32)
```

So the observation is **always `[state, goal]`** with a consistent structure. Standard Metaworld envs do not expose this exact convention.

### Code: Observation space must match `2 * obs_dim`

**File:** `env_utils.py`

The wrapper also overrides `observation_space` so that `obs_dim = observation_space.shape[0] // 2` works:

```127:132:env_utils.py
  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 7, -np.inf),
        high=np.full(2 * 7, np.inf),
        dtype=np.float32)
```

So the **observation format** and **space** are under full control of the wrapper, not Metaworld’s default.

### Code: Downstream use of `obs_dim` and `[state, goal]`

**File:** `env_utils.py` — `load()` computes `obs_dim` from the wrapped env:

```62:65:env_utils.py
  gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
  obs_dim = gym_env.observation_space.shape[0] // 2
  return gym_env, obs_dim, max_episode_steps
```

**File:** `contrastive/utils.py` — `make_environment()` uses `obs_dim` and builds indices so the wrapper exposes `[state, goal]`:

```176:186:contrastive/utils.py
  gym_env, obs_dim, max_episode_steps = env_utils.load(env_name, fixed_start_end)
  goal_indices = obs_dim + obs_to_goal_1d(np.arange(obs_dim), start_index,
                                          end_index)
  indices = np.concatenate([
      np.arange(obs_dim),
      goal_indices
  ])
  env = gym_wrapper.GymWrapper(gym_env)
  env = step_limit.StepLimitWrapper(env, step_limit=max_episode_steps)
  env = ObservationFilterWrapper(env, indices)
  return env, obs_dim
```

All of this assumes the **raw** gym env (from `env_utils.load`) already returns `obs = [state, goal]` with length `2 * obs_dim`, which is why the wrappers override `_get_obs()` and `observation_space`.

---

## 3. Custom goal sampling every reset

### What the algorithm needs

On **every `reset()`**, the goal must be set in a controlled way:

- **Fixed goal** (e.g. for evaluation): use a predefined position (or start/end pair).
- **Random goal** (e.g. for training): sample e.g. uniformly between start and end positions, or with specific ranges.

The env’s internal “target” (e.g. `_target_pos`) must be set to **this** goal so that physics and visualization match.

### What Metaworld “directly” does

With the standard API, the task (and thus the effective goal) is usually fixed per env or set via `set_task()`; you don’t get per-reset goal sampling in the form needed here.

### Code: SawyerBin — goal sampling in `reset()`

**File:** `env_utils.py`

After calling the base `reset()`, the wrapper gets start/end positions from the sim and either uses a fixed goal or samples one:

```80:98:env_utils.py
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
```

So **each episode** can have a different goal (or the same fixed one), and the env’s target is updated to match.

### Code: SawyerBox — same pattern

**File:** `env_utils.py`

```150:166:env_utils.py
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
```

### Code: SawyerPeg — same pattern

**File:** `env_utils.py`

```223:238:env_utils.py
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
```

### Code: Where fixed goals come from (evaluation)

**File:** `lp_contrastive.py`

Fixed goal coordinates per env are defined here; when `fix_goals=True`, these are passed as `fixed_start_end` and used in the `reset()` branches above:

```31:36:lp_contrastive.py
# fixed goal coordinates for supported environments
fixed_goal_dict={'point_Spiral11x11': [np.array([5,5], dtype=float), np.array([10,10], dtype=float)],
                     #note: sawyer fixed goal positions vary slightly with each episode
                      'sawyer_bin': np.array([0.12, 0.7, 0.02]),
                      'sawyer_box': np.array([0.0, 0.75, 0.133]),
                      'sawyer_peg': np.array([-0.3, 0.6, 0.0])}
```

So “custom goal sampling” is implemented in each wrapper’s `reset()` and driven by `fixed_start_end` from this dict (or `None` for random sampling).

---

## 4. Reward and success w.r.t. the chosen goal

### What the algorithm needs

Reward (and success) must be defined with respect to **the goal chosen in `reset()`** (`self._goal` / `self._goal_pos`), not necessarily the task’s built-in goal. The same physical task (e.g. bin-picking) is used with **many different goals** per episode; the reward must align with the current goal.

### What the wrappers do

They **override `step()`**: call the base `step(action)`, then compute distance to **their** goal and return a 0/1 reward (and optionally `info`) based on that.

### Code: SawyerBin — reward from distance to `self._goal`

**File:** `env_utils.py`

```99:108:env_utils.py
  def step(self, action):
    super(SawyerBin, self).step(action)
    obj_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - obj_pos)
    obs = self._get_obs()
    r = float(dist < 0.05)  # Taken from metaworld
    done = False
    info = {}
        
    return obs, r, done, info
```

So success is “object within 0.05 of **our** goal,” not the base env’s internal target.

### Code: SawyerBox — position and quaternion

**File:** `env_utils.py`

```167:181:env_utils.py
  def step(self, action):
    super(SawyerBox, self).step(action)
    obj_pos = self._get_pos_objects()
    obj_quat = self._get_quat_objects()
    
    dist_pos = np.linalg.norm(self._goal_pos - obj_pos)
    dist_quat = np.linalg.norm(self._goal_quat - obj_quat)
    
    obs = self._get_obs()
    r = float(dist_pos < 0.08 and dist_quat < 0.08)  # Taken from metaworld
    done = False
    info = {}
    
    return obs, r, done, info
```

### Code: SawyerPeg — scaled distance to peg head

**File:** `env_utils.py`

```240:250:env_utils.py
  def step(self, action):
    super(SawyerPeg, self).step(action)
    obj_head = self._get_site_pos("pegHead")
    
    scale = np.array([1.0, 2.0, 2.0])
    dist_pos = float(np.linalg.norm((obj_head - self._goal_pos) * scale))
       
    r = float(dist_pos < 0.07)  # Taken from metaworld
    done = False
    info = {}
    return self._get_obs(), r, done, info
```

So in all three Sawyer wrappers, **reward is explicitly tied to the wrapper’s goal** (`_goal` / `_goal_pos` / `_goal_quat`), not to the base Metaworld task’s internal goal.

---

## 5. Single task, many goals

### What the algorithm does

They are **not** using MT10/MT50 or switching between different **tasks**. They use **one** Sawyer task (e.g. bin-picking) and vary the **goal** (e.g. where the block should go) every episode. That’s a different abstraction from “sample a task” or “set_task.”

### Where this shows up

- **One** base env per wrapper: e.g. `ALL_V2_ENVIRONMENTS['bin-picking-v2']` (and similarly for box and peg).
- **Many goals per run**: in `reset()`, the goal is either fixed from `fixed_goal_dict` or resampled (e.g. `t * pos1 + (1 - t) * pos2`).
- **`load()`** (env_utils.py) maps env names to these **single-task** wrapper classes and passes `fixed_start_end` so the same class can do fixed-goal or random-goal:

```34:59:env_utils.py
def load(env_name, fixed_start_end=None):
  """Loads the train and eval environments, as well as the obs_dim."""
  # pylint: disable=invalid-name
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
  elif env_name.startswith('point_'):
    CLASS = point_env.PointEnv
    ...
```

So “single task, many goals” is implemented by having one Metaworld task per wrapper and controlling goal via `reset()` and `fixed_start_end`.

---

## 6. Summary table

| Requirement | Metaworld “directly” | This codebase | Main code locations |
|------------|----------------------|----------------|----------------------|
| Goal chosen every reset | Task-based; no per-reset goal sampling | Custom in `reset()` (fixed or random) | `env_utils.py`: `reset()` in SawyerBin (80–98), SawyerBox (150–166), SawyerPeg (223–238) |
| Observation = `[state, goal]` | Task-specific state layout | Override `_get_obs()`, return `concatenate([state, goal])` | `env_utils.py`: `_get_obs()` and `observation_space` in each Sawyer class |
| Reward w.r.t. our goal | Task’s internal goal | Override `step()`, reward from `self._goal` / `_goal_pos` | `env_utils.py`: `step()` in SawyerBin (99–108), SawyerBox (167–181), SawyerPeg (240–250) |
| Single task, many goals | Multi-task API (MT10, set_task) | One V2 env per wrapper; many goals via `reset()` | `env_utils.py`: class inheritance (66–69, 135–136, 211–212), `load()` (34–59) |
| Fixed goals for evaluation | N/A | `fixed_start_end` from `fixed_goal_dict` | `lp_contrastive.py`: `fixed_goal_dict` (31–36); `env_utils.py`: `reset()` branches using `_fixed_start_end` |

---

## 7. Short answer

They **subclass** the low-level Sawyer envs (from `ALL_V2_ENVIRONMENTS`) instead of using Metaworld “directly” because:

1. They need a **goal-conditioned** interface: custom goal sampling each reset and reward/success w.r.t. that goal.
2. They need a **fixed observation format**: `obs = [state, goal]` with known `obs_dim` for the rest of the pipeline.
3. They want **one** Metaworld task (e.g. bin-picking) with **many goals** per run, not Metaworld’s multi-task “sample/set task” API.

All of the above is implemented in **`env_utils.py`** (SawyerBin, SawyerBox, SawyerPeg), with fixed-goal values supplied from **`lp_contrastive.py`** and `obs_dim` / filtering used in **`contrastive/utils.py`**.
