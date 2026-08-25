# Sawyer goal-wrapper paired positive-control audit

Date: 2026-08-25  
Scope: Task 5 (`sawyer_handle_press_side`) and Task 8
(`sawyer_window_close`)

> **Superseded audit contract:** this note documents the V2 full-3D proxy.
> MetaWorld Task 5 uses only handle z and Task 8 uses only window x for
> success, so V2 outputs cannot authorize promotion. Use the V5 procedure in
> `2026-08-25_native_success_wrapper_repair_and_10task_revalidation.md`.
> V3 outputs are also non-promotable because its compatibility fallback passed
> the custom goal-conditioned observation to MetaWorld `evaluate_state`.
> V4 outputs are non-promotable because the custom V2 subclasses were
> internally classified as V1 by MetaWorld's runtime class-name check.

## Motivation

The first wrapper audit obtained zero scripted-policy success for both custom
and re-evaluated native predicates. Agreement between two predicates that are
always false does not validate the task. The subsequent 1M comparison also
showed that mechanism-only goals reduced success AUC, so nuisance goal
coordinates are not supported as the primary failure.

The unresolved validity question is narrower and more serious: each local
wrapper first runs MetaWorld's randomized reset and then replaces the native
per-reset `_target_pos` with a fixed absolute coordinate. The fixed coordinate
may not be the physical endpoint associated with that reset. This audit
isolates that possibility before any promotion training.

## Paired experiment

For each seed, task, and official ML1 task object, the same frozen MetaWorld
random vector is applied to three environments:

1. **Native official:** untouched MetaWorld environment and official scripted
   policy. Native `info["success"]` is authoritative.
2. **Wrapper/native target:** the custom goal-conditioned wrapper receives
   `fixed_start_end=None` and preserves the `_target_pos` created by the native
   reset.
3. **Wrapper/fixed target:** the current historical wrapper overwrites
   `_target_pos` with the fixed global coordinate.

The native action sequence is then replayed from the identical reset through
both wrappers. Let $m_t$ be the mechanism position, $g_{\mathrm{native}}$ the
target produced by the official reset, and $g_{\mathrm{fixed}}$ the historical
fixed target. The audit independently records

$$
d_{\mathrm{native},t}
=\lVert m_t-g_{\mathrm{native}}\rVert_2,
\qquad
d_{\mathrm{fixed},t}
=\lVert m_t-g_{\mathrm{fixed}}\rVert_2.
$$

It also checks the replayed physical trajectory:

$$
e_{\mathrm{traj}}
=\max_t\lVert m_t^{\mathrm{wrapper}}
-m_t^{\mathrm{native}}\rVert_\infty.
$$

This makes the causal signatures distinct:

- If untouched native success is below $0.80$, the behavior-policy adapter,
  MetaWorld version, or audit setup is invalid; no wrapper conclusion is
  allowed.
- If native succeeds but the native-target wrapper or exact action replay
  fails, the custom wrapper has a problem beyond the fixed target.
- If native and native-target controls succeed, fixed-target success is at
  most $0.20$, the fixed replay still reaches the native endpoint, and
  $e_{\mathrm{traj}}\le10^{-5}$, the fixed absolute target is semantically
  invalid.
- If the fixed-target wrapper succeeds at least $0.80$, the current target
  contract passes this audit.

Target pairing must satisfy $10^{-6}$ and the native/fixed target separation
must exceed the task success radius ($0.02$ for Task 5 and $0.05$ for Task 8)
before the audit calls the fixed target misaligned.
The frozen MetaWorld random vectors must also match within $10^{-6}$. The JSON
records the site nearest each native/fixed target, which reveals whether the
historical coordinate is separated from MetaWorld's own physical goal marker.

## Implementation

- `scripts/audit_sawyer_goal_positive_controls.py`
  - constructs official ML1 tasks;
  - applies the same task object to native and custom environments;
  - runs all three primary conditions;
  - replays the exact native action sequence through both wrappers;
  - records native/custom/independent-distance success, target pairing,
    best/final distances, action clipping, and trajectory equality;
  - emits JSON and W&B metrics with an explicit decision code.
- `experiment_configs_goal_wrapper_positive_controls.py`
  - enumerates seeds 5, 6, and 7 with 50 episodes per task.
- `DRAFT_goal_wrapper_positive_controls.sh`
  - Torch-HPC array launcher; requests one GPU so `MUJOCO_GL=egl` does not
    fall back to compiling OSMesa on CPU nodes;
  - explicitly activates `contrastive_rl` before importing NumPy, MetaWorld,
    or MuJoCo.
- `scripts/evaluate_goal_wrapper_positive_controls.py`
  - requires all three seed JSON files;
  - aggregates the decisions and writes JSON plus a Markdown results table;
  - blocks promotion unless the current fixed-target wrapper validates on
    both tasks.
- `tests/test_goal_wrapper_positive_controls.py`
  - dependency-light decision-tree, metadata, config, and launcher checks.
- `contrastive/goal_semantics.py`
  - records the untouched MetaWorld task names used by the positive controls.

No actor, critic, replay, learner, or training configuration is changed.

## Launch and aggregation

Run the three array jobs (one GPU each, EGL headless, no training):

```bash
sbatch DRAFT_goal_wrapper_positive_controls.sh
```

After all three finish:

```bash
python scripts/evaluate_goal_wrapper_positive_controls.py \
  logs/goal_validity/positive_controls_seed{5,6,7}.json
```

The W&B group is:

```text
GOAL-WRAPPER-POSITIVE-CONTROLS-V2
```

Important metrics use:

```text
positive_control/task{5,8}/native_official/native_info_success_mean
positive_control/task{5,8}/wrapper_native_target_policy/positive_reward_success_mean
positive_control/task{5,8}/wrapper_fixed_target_policy/positive_reward_success_mean
positive_control/task{5,8}/wrapper_fixed_target_replay/native_target_success_mean
positive_control/task{5,8}/wrapper_fixed_target_replay/fixed_target_success_mean
positive_control/task{5,8}/wrapper_fixed_target_replay/trajectory_linf_error_vs_native_max
positive_control/task{5,8}/pairing/fixed_to_native_target_distance_mean
positive_control/task{5,8}/pairing/rand_vec_pair_linf_error_max
positive_control/task{5,8}/fixed_global_target_misaligned
positive_control/task{5,8}/fixed_target_valid
```

## Wrapper repair if the conjecture is confirmed

The recommended repair is **not** to replace `_target_pos` after a randomized
native reset.

The minimal faithful mode is:

```python
super().reset()
self._goal = self._target_pos.copy()
```

This preserves MetaWorld's per-reset endpoint and makes the exposed desired
mechanism goal, sparse reward, and native success predicate refer to the same
physical state.

If a fixed endpoint is required across episodes, freeze the **entire official
MetaWorld task/random vector**, call `set_task()` with that task on every
reset, and then copy the native `_target_pos`. Fixing the whole task keeps the
mechanism geometry and endpoint consistent; replacing only the global target
does not.

The corrected continual wrapper should support explicit modes:

- `native_reset`: sample the official task distribution and expose the native
  per-reset target; recommended for benchmark-faithful training.
- `fixed_native_task`: reuse one official task object/random vector per
  continual task and expose its native target; recommended for controlled
  fixed-goal comparisons.
- `legacy_absolute`: retain the historical coordinate only for reproducing
  old results, with a validity warning.

For Tasks 5 and 8, the desired goal used for success should be the native
mechanism target. The wrapper should propagate `info["success"]` and set its
sparse reward from that same predicate. If a uniform continual observation
width is required, pad the success-relevant goal and provide an explicit goal
mask; padded or invented hand/gripper coordinates must not silently become
additional desired constraints.

After repair, rerun the positive controls first, then a short vanilla-policy
or sparse-SAC positive control, and only then repeat DCC comparisons. Historical
Task-5/Task-8 results should be labelled as using the legacy wrapper until the
audit determines whether they remain comparable.

## Validation and limitations

Validation includes Python compilation, shell syntax, three-config expansion,
and dependency-light tests for every decision branch. Full physics validation
must run on Torch inside the project's `contrastive_rl` environment, on a
GPU node so `mujoco_py` can use EGL. CPU-only jobs fail compiling OSMesa
because Torch CPU images do not ship `GL/osmesa.h`.

The scripted policy establishes interface and task solvability, not DCC
learnability. The audit intentionally uses official ML1 task objects; if the
installed MetaWorld release changes task serialization or policy APIs, the
native control fails closed rather than silently blaming the wrapper.
