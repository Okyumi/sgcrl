# State and Goal Index Semantics

This document lists the **semantic meaning of each index** for **state** and **goal** in every task. **Option A (pad to max dimension) is implemented** in `env_utils.py`: all Sawyer task wrappers return a unified observation of length **22** (state 11 + goal 11), with smaller state/goal vectors zero-padded.

**Convention:** Every Sawyer environment exposes `observation = [state_padded, goal_padded]` where:
- `state_padded` has length **STATE_DIM_UNIFIED = 11**, `goal_padded` has length **GOAL_DIM_UNIFIED = 11**.
- Full observation length is **FULL_OBS_DIM = 22**. Indices **0:11** = state, **11:22** = goal.
- Tasks with raw state dim &lt; 11 (or goal dim &lt; 11) are padded with zeros at the end. Semantic indices are unchanged; padding occupies the highest indices.

**Alignment:** For every task in `env_utils.py`, the **goal has the exact same semantic meaning as the state**: goal index `(11 + i)` in the full observation is the *desired value* for the quantity at state index `i`. So `goal[i]` (in goal-local indices) = desired value for `state[i]`. Each component of the state (hand, gripper, object positions, etc.) has a corresponding desired value in the goal at the same index. Each `_get_obs()` in `env_utils.py` documents the raw state/goal layout and padding; see those for the exact index ranges.

---

## Part 1: Tasks in `env_utils.py` (current wrappers)

### 1.1 point_Spiral11x11 (and other point_*)

State and goal use the **same index semantics**: state 0–1 (agent x, y) ↔ goal 2–3 (target x, y).

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–1     | 2   | Agent position (x, y) in the maze. |
| Goal   | 2–3     | 2   | Target position (x, y) to reach. |

- **Full observation length:** 4  
- **Success:** Agent reaches goal (e.g. within threshold).  
- **Note:** `fixed_goal_dict` gives `[start_pos, end_pos]` (each 2D). Implemented in `point_env.py` (loaded via `env_utils.load()`).

---

### 1.2 sawyer_bin (bin-picking-v2)

State and goal use the **same index semantics**: hand(0–2)↔7–9, gripper(3)↔10, block(4–6)↔11–13.

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position (x, y, z). |
| State  | 3       | 1   | Gripper distance apart (normalized 0–1). |
| State  | 4–6     | 3   | Block position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (block location + small z offset). |
| Goal   | 10      | 1   | Desired gripper state (e.g. 0.4 = open). |
| Goal   | 11–13   | 3   | Desired block position (in bin). |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Block within 0.05 of goal position.  
- **Goal refers to:** Block’s target position (in bin).

---

### 1.3 sawyer_box (box-close-v2)

State and goal use the **same index semantics**: hand(0–2)↔11–13, gripper(3)↔14, lid_pos(4–6)↔15–17, lid_quat(7–10)↔18–21.

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Lid position (x, y, z). |
| State  | 7–10    | 4   | Lid quaternion. |
| Goal   | 11–13   | 3   | Desired hand position (above lid). |
| Goal   | 14      | 1   | Desired gripper state. |
| Goal   | 15–17   | 3   | Desired lid position (on box). |
| Goal   | 18–21   | 4   | Desired lid quaternion. |

- **State dim:** 11, **goal dim:** 11, **full obs:** 22  
- **Success:** Lid position within 0.08 of goal and quat within 0.08.  
- **Goal refers to:** Lid position and orientation on the box.  
- **Implementation:** `env_utils.SawyerBox._get_obs()`; see inline index comment there.

---

### 1.4 sawyer_peg (peg-insert-side-v2)

State and goal use the **same index semantics**: hand(0–2)↔7–9, gripper(3)↔10, peg_head(4–6)↔11–13.

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Peg head position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above hole). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Desired peg position (in hole). |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Peg head within 0.07 of goal (with scale [1, 2, 2]).  
- **Goal refers to:** Peg head’s target position (in hole).

---

### 1.5 sawyer_push_back (push-back-v2)

State and goal use the **same index semantics**: each goal index 7–13 corresponds to the same semantic role as state index (i−7) for i = 7…13.

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Object position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Desired object position (pushed-back location). |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Object within 0.07 of goal.  
- **Goal refers to:** Object’s target position.  
- **Implementation:** `env_utils.SawyerPushBack._get_obs()` builds `[obs, goal]` with this layout; see inline comments there.

**Summary (Part 1):** All environments loaded via `env_utils.load()` use aligned state/goal index semantics. Each Sawyer wrapper’s `_get_obs()` in `env_utils.py` documents the exact state and goal index ranges in an inline comment; the goal vector mirrors the state layout (hand, gripper, object/handle/nail/peg/lid position—and quaternion for box only).

---

## Part 2: Ten-task CKA-RL Meta-World sequence

The following ten tasks are the CKA-RL sequence (task names only; see *Metaworld_analysis/TASK_GOAL_ANALYSIS.md* for the task list). For each task, the table below is derived from the **corresponding Sawyer environment source code** (V3 envs in this repo: `metaworld_envscripts/envs/sawyer_*_v3.py`) as the reference for building a goal-conditioned wrapper with `observation = [state, goal]`. Experiments use V2; the V3 code defines object positions, targets, and success criteria used here.

---

### 2.1 hammer-v2

*Source: `Metaworld-2.0.0/.../sawyer_hammer_v2.py` — `_get_pos_objects()` returns `np.hstack((hammer (3), nail_link (3)))`. `_target_pos` from site `"goal"` (nail target). Success: `NailSlideJoint` > 0.09. **State must include both hammer and nail so it is not truncated.** Goal mirrors state index-by-index.*

| Role   | Indices (full obs) | Dim | Semantic meaning |
|--------|--------------------|-----|------------------|
| State  | 0–2                | 3   | End-effector position. |
| State  | 3                  | 1   | Gripper distance apart. |
| State  | 4–6                | 3   | Hammer position (x, y, z). |
| State  | 7–9                | 3   | Nail position (x, y, z) — success is nail at goal. |
| Goal   | 11–13              | 3   | Desired hand position (above target). |
| Goal   | 14                 | 1   | Desired gripper state. |
| Goal   | 15–17              | 3   | Desired hammer position (at target when task done). |
| Goal   | 18–20              | 3   | Desired nail position (nail target / goal site). |

- **State dim:** 10 (hand, gripper, hammer, nail), **goal dim:** 10 (same semantics as state), **full obs (after padding):** 22  
- **Success:** Nail driven in (e.g. joint > 0.09 in base env); wrapper uses nail within 0.05 of goal.  
- **Goal refers to:** Desired state at success: hand above target, gripper open, hammer and nail at target.  
- **Implementation:** `env_utils.SawyerHammer._get_obs()` builds goal as `[hand_above, gripper, _goal, _goal]` so goal mirrors state; no truncation.

---

### 2.2 push-wall-v2

*Source: `sawyer_push_wall_v3.py` — `_get_pos_objects()` returns object geom position (3); `_get_quat_objects()` returns object quat (4). obs[4:7] = object. Success: obj_to_target ≤ 0.07.*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Object position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Object target position (against wall). |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Object within 0.07 of goal.  
- **Goal refers to:** Object’s target position against the wall.

---

### 2.3 faucet-close-v2

*Source: `sawyer_faucet_close_v3.py` — `_get_pos_objects()` returns handle position (site `handleStartClose` + offset); `_target_pos` = handle closed position. Success: target_to_obj ≤ 0.07.*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Faucet handle position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Handle closed (target) position. |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Handle within 0.07 of target.  
- **Goal refers to:** Faucet handle’s closed position.

---

### 2.4 push-back-v2

*Source: same as env_utils wrapper; also `sawyer_push_back_v3.py` — `_get_pos_objects()` returns object geom position (3). Success: target_to_obj ≤ 0.07 (TARGET_RADIUS 0.05 in reward).*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Object position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Object target position (pushed-back location). |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Object within 0.07 of goal.  
- **Goal refers to:** Object’s target position.

---

### 2.5 stick-pull-v2

*Source: `sawyer_stick_pull_v3.py` — `_get_pos_objects()` returns stick (3) + insertion/handle (3). Success: handle to target and stick inserted (norm ≤ 0.12). Goal mirrors state index-by-index.*

| Role   | Indices (full obs) | Dim | Semantic meaning |
|--------|--------------------|-----|------------------|
| State  | 0–2                | 3   | End-effector position. |
| State  | 3                  | 1   | Gripper distance apart. |
| State  | 4–6                | 3   | Stick position (x, y, z). |
| State  | 7–9                | 3   | Handle/insertion position (x, y, z) — object being pulled. |
| State  | 10                 | 1   | Corrected/native signed insertion margin; nonnegative iff inserted. Legacy mode pads zero. |
| Goal   | 11–13              | 3   | Desired hand position (above target). |
| Goal   | 14                 | 1   | Desired gripper state. |
| Goal   | 15–17              | 3   | Desired stick position (at target when task done). |
| Goal   | 18–20              | 3   | Desired handle position (handle target). |
| Goal   | 21                 | 1   | Corrected/native desired signed insertion margin. Legacy mode pads zero. |

- **State/goal semantic dim:** 11 in corrected/native fixed-goal runs; 10 plus one zero-padding coordinate in legacy mode. **Full obs:** 22
- **Success:** Handle within 0.12 of target and stick inserted.  
- **Goal refers to:** An audited state actually visited while all official success gates were true.

---

### 2.6 handle-press-side-v2

*Source: `sawyer_handle_press_side_v3.py` — `_get_pos_objects()` returns site `handleStart` (3). `_target_pos` = site `goalPress`. TARGET_RADIUS = 0.02. Success: target_to_obj ≤ 0.02.*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Handle position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Handle pressed (target) position. |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Handle within 0.02 of goal.  
- **Goal refers to:** Handle’s pressed position.

---

### 2.7 push-v2

*Source: `sawyer_push_v3.py` — `_get_pos_objects()` returns body `obj` (3); obs[4:7] = object. TARGET_RADIUS = 0.05. Success: target_to_obj ≤ 0.05.*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Object position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Object target position. |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Object within 0.05 of goal.  
- **Goal refers to:** Object’s target position.

---

### 2.8 shelf-place-v2

*Source: `sawyer_shelf_place_v3.py` — `_get_pos_objects()` returns body `obj` (3); obs[4:7] = object. Success: obj_to_target ≤ 0.07.*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Object position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Object target position on shelf. |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Object within 0.07 of goal.  
- **Goal refers to:** Object’s target position on the shelf.

---

### 2.9 window-close-v2

*Source: `sawyer_window_close_v3.py` — `_get_pos_objects()` returns site `handleCloseStart` (3). `_target_pos` = window closed position. TARGET_RADIUS = 0.05. Success: target_to_obj ≤ 0.05.*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Window handle position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Window closed (target) position. |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Window/handle within 0.05 of goal.  
- **Goal refers to:** Window’s closed position.

---

### 2.10 peg-unplug-side-v2

*Source: `sawyer_peg_unplug_side_v3.py` — `_get_pos_objects()` returns site `pegEnd` (3). obs[4:7] used in compute_reward. `_target_pos` = peg unplugged position. Success: obj_to_target ≤ 0.07.*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Peg end position (x, y, z). |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Peg unplugged (target) position. |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Peg within 0.07 of goal.  
- **Goal refers to:** Peg’s unplugged position.

---

## Part 3: Unified space (Option A implemented)

### Raw dimension summary (before padding)

| Task / env           | Raw state dim | Raw goal dim | After padding (full obs) |
|----------------------|---------------|--------------|---------------------------|
| point_Spiral11x11    | 2             | 2            | 4 (not padded)            |
| sawyer_bin           | 7             | 7            | 22                        |
| sawyer_box           | 11            | 11           | 22                        |
| sawyer_peg           | 7             | 7            | 22                        |
| sawyer_push_back     | 7             | 7            | 22                        |
| sawyer_hammer        | 10            | 7            | 22                        |
| sawyer_push_wall     | 7             | 7            | 22                        |
| sawyer_faucet_close  | 7             | 7            | 22                        |
| sawyer_stick_pull    | 11 corrected / 10 legacy | 11 corrected / 10 legacy | 22 |
| sawyer_handle_press_side | 7         | 7            | 22                        |
| sawyer_push          | 7             | 7            | 22                        |
| sawyer_shelf_place   | 7             | 7            | 22                        |
| sawyer_window_close | 7             | 7            | 22                        |
| sawyer_peg_unplug_side | 7           | 7            | 22                        |

### Option A: Implemented in `env_utils.py`

- **STATE_DIM_UNIFIED = 11**, **GOAL_DIM_UNIFIED = 11**, **FULL_OBS_DIM = 22** (defined in `env_utils.py`).
- **Padding:** `_pad_to_len(arr, length)` appends zeros so that state and goal reach 11 each. Semantic indices 0–6 are unchanged. For corrected/native fixed-goal Stick Pull, index 10 is the signed insertion margin rather than padding; legacy Stick Pull keeps zero padding there.
- **Observation layout:** `observation[0:11]` = state (padded), `observation[11:22]` = goal (padded). So for 7-dim tasks: state semantics in 0–6, zeros in 7–10; goal semantics in 11–17, zeros in 18–21.
- **`load(env_name)`** returns `obs_dim = STATE_DIM_UNIFIED` (11) for all `sawyer_*` envs so that a single policy/critic can be used across the 10 continual tasks.
- **point_*** envs are unchanged (no padding); they are not part of the continual Sawyer 10-task set.

### Option B: Common subset

- Common state: 0–2 hand, 3 gripper, 4–6 main object/target position.
- Tasks with extra state (e.g. stick-pull stick+handle, sawyer_box quat) either drop extras or place them in fixed positions with a mask.
- **Goal:** 0–2 hand above target, 3 gripper, 4–6 target position (same for all 7-dim goal tasks).

### Unified observation index reference (after padding)

For **all Sawyer tasks** the observation has length 22: `obs[0:11]` = state, `obs[11:22]` = goal.

| Unified index | State (obs[0:11])   | Goal (obs[11:22])    | Notes |
|---------------|---------------------|----------------------|-------|
| 0–2           | Hand (x, y, z)      | Desired hand (above target) | Goal mirrors state: same index = same quantity. |
| 3             | Gripper             | Desired gripper (e.g. open)  | Valid for all. |
| 4–6           | Main/first object   | Desired main/first object   | Hammer: hammer. Stick_pull: stick. 7-dim: single object. |
| 7–10          | Second object or padding | Desired second object or padding | Hammer: nail (7–9), pad (10). Stick_pull: handle (7–9), pad (10). Box: lid_quat (7–10). 7-dim tasks: padding (zeros). |

**Two-object tasks (raw state and goal 10 dims):** **Hammer:** state 4–6 = hammer pos, 7–9 = nail pos; goal 15–17 = desired hammer pos, 18–20 = desired nail pos (goal mirrors state). **Stick_pull:** state 4–6 = stick pos, 7–9 = handle pos; goal 15–17 = desired stick pos, 18–20 = desired handle pos (goal mirrors state). In both, state index 10 and goal index 21 are padding.

This table is the reference for keeping state and goal semantics consistent when using the unified observation in continual RL.

---

## Part 4: Verification vs Metaworld-2.0.0 (no position truncation)

For each continual task (excluding bin, peg, box), the following table confirms that **all position dimensions** from `_get_pos_objects()` in Metaworld-2.0.0 are used in our wrapper. No position information is truncated.

| Task (env_utils name) | Metaworld-2.0.0 source file | `_get_pos_objects()` returns | Our wrapper state uses | Truncation? |
|-----------------------|-----------------------------|-------------------------------|-------------------------|-------------|
| sawyer_push_back      | sawyer_push_back_v2.py      | 3 (objGeom xpos)              | obj_pos (3)             | No          |
| sawyer_hammer         | sawyer_hammer_v2.py         | 6 (hammer 3 + nail_link 3)    | hammer_pos (3) + nail_pos (3) | No   |
| sawyer_push_wall      | sawyer_push_wall_v2.py      | 3 (objGeom xpos)              | obj_pos (3)             | No          |
| sawyer_faucet_close   | sawyer_faucet_close_v2.py   | 3 (handleStartClose + offset) | handle_pos (3)          | No          |
| sawyer_stick_pull     | sawyer_stick_pull_v2.py     | 6 (stick body + insertion site) | stick_pos (3) + handle_pos (3) | No  |
| sawyer_handle_press_side | sawyer_handle_press_side_v2.py | 3 (handleStart site)       | handle_pos (3)          | No          |
| sawyer_push           | sawyer_push_v2.py           | 3 (obj body COM)              | obj_pos (3)             | No          |
| sawyer_shelf_place    | sawyer_shelf_place_v2.py    | 3 (obj body COM)              | obj_pos (3)             | No          |
| sawyer_window_close   | sawyer_window_close_v2.py   | 3 (handleCloseStart site)     | handle_pos (3)          | No          |
| sawyer_peg_unplug_side| sawyer_peg_unplug_side_v2.py| 3 (pegEnd site)               | peg_pos (3)             | No          |

**Note:** Our wrappers use only **position** from the base env (from `_get_pos_objects()`). The base env also provides `_get_quat_objects()` (orientation) for some tasks; we do not include quaternions in state/goal, so orientation is not represented. Success criteria in the base env are position-based (e.g. object-to-target distance); our goal is the target position, so goal representation is complete for the task.
