# State and Goal Index Semantics

This document lists the **semantic meaning of each index** for **state** and **goal** in every task. Use it to implement a **unified state/goal space** across tasks (Option A: pad to max dimension; Option B: common subset of indices).

**Convention:** Every environment exposes `observation = [state, goal]`. The first `state_dim` values are the current state; the next `goal_dim` values are the desired goal (same semantic structure as state where applicable).

**Alignment:** For every task in `env_utils.py`, state and goal use the **same index semantics**: goal index `(state_dim + i)` corresponds to the same semantic role as state index `i`. Each `_get_obs()` in `env_utils.py` documents this with an inline comment; see those for the exact index ranges.

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

*Source: `sawyer_hammer_v3.py` — `_get_pos_objects()` returns hammer (3) + nail (3); `_target_pos` from site `"goal"` (nail target). Success: nail joint > 0.09.*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Nail position (x, y, z) — success is nail at goal. |
| Goal   | 7–9     | 3   | Desired hand position (above target). |
| Goal   | 10      | 1   | Desired gripper state. |
| Goal   | 11–13   | 3   | Nail target position (goal site). |

- **State dim:** 7, **goal dim:** 7, **full obs:** 14  
- **Success:** Nail driven in (e.g. joint position > 0.09 in V3).  
- **Goal refers to:** Nail’s target position.

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

*Source: `sawyer_stick_pull_v3.py` — `_get_pos_objects()` returns stick (3) + insertion/handle (3). obs[4:7] = stick, obs[11:14] = handle. Success: handle to target and stick inserted (norm ≤ 0.12).*

| Role   | Indices | Dim | Semantic meaning |
|--------|---------|-----|------------------|
| State  | 0–2     | 3   | End-effector position. |
| State  | 3       | 1   | Gripper distance apart. |
| State  | 4–6     | 3   | Stick position (x, y, z). |
| State  | 7–9     | 3   | Handle/insertion position (x, y, z) — object being pulled. |
| Goal   | 10–12   | 3   | Desired hand position (above target). |
| Goal   | 13      | 1   | Desired gripper state. |
| Goal   | 14–16   | 3   | Handle target position. |

- **State dim:** 10, **goal dim:** 7, **full obs:** 17  
- **Success:** Handle within 0.12 of target and stick inserted.  
- **Goal refers to:** Handle/insertion target position (object pulled to goal).

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

## Part 3: Summary for unified space (Option A / Option B)

### Dimension summary

| Task / env           | State dim | Goal dim | Full obs |
|----------------------|-----------|----------|----------|
| point_Spiral11x11    | 2         | 2        | 4        |
| sawyer_bin           | 7         | 7        | 14       |
| sawyer_box           | 11        | 11       | 22       |
| sawyer_peg           | 7         | 7        | 14       |
| sawyer_push_back     | 7         | 7        | 14       |
| hammer-v2            | 7         | 7        | 14       |
| push-wall-v2         | 7         | 7        | 14       |
| faucet-close-v2      | 7         | 7        | 14       |
| push-back-v2         | 7         | 7        | 14       |
| stick-pull-v2        | 10        | 7        | 17       |
| handle-press-side-v2 | 7         | 7        | 14       |
| push-v2              | 7         | 7        | 14       |
| shelf-place-v2       | 7         | 7        | 14       |
| window-close-v2      | 7         | 7        | 14       |
| peg-unplug-side-v2   | 7         | 7        | 14       |

### Option A: Pad to max dimension

- **state_dim = 11**, **goal_dim = 11** for all Sawyer tasks (sawyer_box is already 11).
- For 7-dim tasks: pad state and goal with zeros at indices 7–10 so indices 0–6 keep the same meaning.
- For stick-pull-v2 (state 10): pad state with one zero at index 10 to reach 11.
- point_* stays 2+2 or is padded if a single global size is desired.

### Option B: Common subset

- Common state: 0–2 hand, 3 gripper, 4–6 main object/target position.
- Tasks with extra state (e.g. stick-pull stick+handle, sawyer_box quat) either drop extras or place them in fixed positions with a mask.
- **Goal:** 0–2 hand above target, 3 gripper, 4–6 target position (same for all 7-dim goal tasks).

### Index alignment across 7-dim Sawyer tasks

For tasks with **state dim 7, goal dim 7**:

| Index | State meaning      | Goal meaning         |
|-------|--------------------|----------------------|
| 0–2   | Hand (x, y, z)     | Hand above target    |
| 3     | Gripper            | Gripper (e.g. open)  |
| 4–6   | Main object (x,y,z)| Target (x, y, z)     |

This table is the reference for keeping state and goal semantics consistent when implementing wrappers or applying Option A/B.
