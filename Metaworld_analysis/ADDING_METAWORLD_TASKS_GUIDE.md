# Guide: Adding New Metaworld Tasks to SGCRL

This document provides step-by-step instructions for adding new Metaworld tasks (beyond sawyer_bin, sawyer_box, sawyer_peg) to the SGCRL codebase. The codebase subclasses Metaworld's Sawyer V2 environments from `metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS` and modifies them to provide a goal-conditioned interface.

---

## Overview

SGCRL subclasses the low-level Sawyer environments from Metaworld instead of using Metaworld's high-level API because the algorithm needs:

1. **Custom goal sampling** every reset (fixed or random)
2. **Observation format** `[state, goal]` with known structure
3. **Reward** defined with respect to the chosen goal, not Metaworld's internal task goal
4. **Single task, many goals** per run (not Metaworld's multi-task API)

Each wrapper must override: `__init__`, `reset()`, `step()`, `_get_obs()`, and `observation_space`.

---

## Part 1: Scripts to Modify

### 1. `env_utils.py` (primary)

- **`load()` function**: Add `elif` branches for each new environment name. Map env_name → CLASS, max_episode_steps, and kwargs (e.g. `fixed_start_end`).
- **Wrapper classes**: Add one new class per task (e.g. `SawyerHammer`, `SawyerPushWall`). Each class subclasses `ALL_V2_ENVIRONMENTS['<key>']` and implements the goal-conditioned interface.

### 2. `lp_contrastive.py`

- **`fixed_goal_dict`**: Add an entry for each new environment with a 3D fixed goal position (for evaluation when `fix_goals=True`).

### 3. `draft.sh` (or your launch script)

- Update `--env=` flag when you want to run the new environment.

---

## Part 2: `ALL_V2_ENVIRONMENTS` Keys for the 10 Tasks

The keys in `ALL_V2_ENVIRONMENTS` follow kebab-case: `task-name-v2`. Mapping for the 10 tasks:

| Task | Environment (paper/table) | ALL_V2_ENVIRONMENTS key | Goal Refers To |
|------|---------------------------|-------------------------|----------------|
| 0 | HammerV2 | `'hammer-v2'` | Nail position |
| 1 | PushWallV2 | `'push-wall-v2'` | Object position |
| 2 | FaucetCloseV2 | `'faucet-close-v2'` | Faucet handle position |
| 3 | PushBackV2 | `'push-back-v2'` | Object position |
| 4 | StickPullV2 | `'stick-pull-v2'` | Object position |
| 5 | HandlePressSideV2 | `'handle-press-side-v2'` | Handle position |
| 6 | PushV2 | `'push-v2'` | Object position |
| 7 | ShelfPlaceV2 | `'shelf-place-v2'` | Object position |
| 8 | WindowCloseV2 | `'window-close-v2'` | Window position |
| 9 | PegUnplugSideV2 | `'peg-unplug-side-v2'` | Peg position |

**Note**: If your Metaworld version uses different naming (e.g. V3), you may need to inspect `ALL_V2_ENVIRONMENTS.keys()` or use a compatible Metaworld version. The current codebase assumes V2.

---

## Part 3: Implementation Pattern (per task)

Use the existing `SawyerBin`, `SawyerBox`, or `SawyerPeg` classes in `env_utils.py` as templates.

### 1. Create the wrapper class

```python
class SawyerHammer(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['hammer-v2']):
  """Wrapper for Hammer: goal = nail target position."""
```

### 2. Implement `__init__`

- Set `self._goal`, `self._fixed_start_end`
- Set `self._partially_observable = False`, `self._freeze_rand_vec = False`, `self._set_task_called = True`
- Call `super().__init__()`, then `self.reset()`

### 3. Implement `reset()`

- Call `super().reset()`
- Get start/end positions from the sim (use the correct method for the target object)
- Either use `self._fixed_start_end` (fixed goal) or sample goal between start/end
- Set `self._target_pos = self._goal` (or equivalent for the base env)
- Return `self._get_obs()`

### 4. Implement `step()`

- Call `super().step(action)`
- Get current target object position (nail, object, handle, etc.)
- Compute distance to `self._goal`
- Use success radius: **0.05** for most tasks, **0.02** for HandlePressSideV2
- Return `obs, r, done, info`

### 5. Implement `_get_obs()`

- Build **state**: hand pos, gripper distance, target object state (pos, quat if needed)
- Build **goal**: ideal state matching goal (e.g. goal pos + gripper offset, goal quat)
- Return `np.concatenate([obs, goal]).astype(np.float32)` so observation is `[state, goal]`

### 6. Override `observation_space`

- Return `gym.spaces.Box(low=..., high=..., shape=(2 * obs_dim,), dtype=np.float32)`

---

## Part 4: Target Object Position Methods by Task

Different tasks track different objects. Use the correct method for each:

| Task | Goal Refers To | Method to get target position |
|------|----------------|------------------------------|
| HammerV2 | Nail | Check base env for nail body/site; may need `body_name2id` or `_get_pos_objects` |
| PushWallV2 | Object | `_get_pos_objects()` |
| FaucetCloseV2 | Faucet handle | `_get_pos_objects()` |
| PushBackV2 | Object | `_get_pos_objects()` |
| StickPullV2 | Object | `_get_pos_objects()` |
| HandlePressSideV2 | Handle | `_get_pos_objects()` |
| PushV2 | Object | `_get_pos_objects()` |
| ShelfPlaceV2 | Object | `_get_pos_objects()` |
| WindowCloseV2 | Window | `_get_pos_objects()` |
| PegUnplugSideV2 | Peg | `_get_pos_objects()` or `_get_site_pos('pegHead')` |

**Tip**: Inspect each base env class (e.g. `SawyerHammerEnvV2`) in the Metaworld source to find the correct method and body/site names.

---

## Part 5: Add to `load()` in `env_utils.py`

```python
elif env_name == 'sawyer_hammer':
  CLASS = SawyerHammer
  max_episode_steps = 150
  kwargs['fixed_start_end'] = fixed_start_end
# Repeat for: sawyer_push_wall, sawyer_faucet_close, sawyer_push_back,
#             sawyer_stick_pull, sawyer_handle_press_side, sawyer_push,
#             sawyer_shelf_place, sawyer_window_close, sawyer_peg_unplug_side
```

---

## Part 6: Add to `fixed_goal_dict` in `lp_contrastive.py`

```python
'sawyer_hammer': np.array([0.24, 0.74, 0.11]),      # example; tune per task
'sawyer_push_wall': np.array([0.05, 0.85, 0.015]),
'sawyer_faucet_close': np.array([-0.14, 0.82, 0.13]),
'sawyer_push_back': np.array([0.06, 0.62, 0.02]),
'sawyer_stick_pull': np.array([0.41, 0.54, 0.02]),
'sawyer_handle_press_side': np.array([-0.07, 0.68, 0.07]),
'sawyer_push': np.array([0.02, 0.89, 0.02]),
'sawyer_shelf_place': np.array([0.02, 0.89, 0.30]),
'sawyer_window_close': np.array([0., 0.80, 0.2]),
'sawyer_peg_unplug_side': np.array([0.01, 0.66, 0.13]),
```

*These values are examples; adjust based on goal space bounds from TASK_GOAL_ANALYSIS.md or empirical runs.*

---

## Success Condition Reference

| Task | Success radius |
|------|----------------|
| Most tasks | 0.05 |
| HandlePressSideV2 | 0.02 |

---

## Quick Reference: env_name → ALL_V2 key

| env_name | ALL_V2_ENVIRONMENTS key |
|----------|-------------------------|
| sawyer_hammer | hammer-v2 |
| sawyer_push_wall | push-wall-v2 |
| sawyer_faucet_close | faucet-close-v2 |
| sawyer_push_back | push-back-v2 |
| sawyer_stick_pull | stick-pull-v2 |
| sawyer_handle_press_side | handle-press-side-v2 |
| sawyer_push | push-v2 |
| sawyer_shelf_place | shelf-place-v2 |
| sawyer_window_close | window-close-v2 |
| sawyer_peg_unplug_side | peg-unplug-side-v2 |

---

## Verifying ALL_V2_ENVIRONMENTS Keys

If your Metaworld version differs, run:

```bash
python -c "
import metaworld
for k in sorted(metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS.keys()):
    print(k)
"
```

to list the actual keys for your installation.
