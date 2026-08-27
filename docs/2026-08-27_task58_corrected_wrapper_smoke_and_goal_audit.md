# Task-5/Task-8 corrected-wrapper smoke and goal audit (2026-08-27)

## Motivation

The corrected original wrappers should be validated in the real MuJoCo
simulator before launching the expensive ten-seed, nine-baseline pipeline.
The earlier unit checks prove the scalar predicates in isolation, but they do
not prove that reset observations, internal targets, simulator sites, emitted
goals, and transition rewards agree at runtime.

A related question is whether Task 5 is invalid because its policy receives a
full desired state even though official success depends only on handle height.
The SGCRL paper and released code answer that question: the policy receives a
desired goal observation/state, while evaluation may use a smaller subset of
that state. It is not conditioned on a Boolean success flag.

## Goal-representation audit

The original SGCRL `SawyerBin` wrapper emits a seven-dimensional state and a
seven-dimensional goal: hand position, gripper opening, and block position.
Its sparse evaluation checks only the block's three-dimensional distance to
the target. `SawyerBox` similarly emits hand, gripper, lid position, and lid
quaternion, while success checks only lid position and quaternion. `SawyerPeg`
emits hand, gripper, and peg-head position, while success checks only a
weighted peg-head distance. Therefore coordinates absent from the success
predicate are deliberately present in every released Sawyer goal.

The ten-task continual wrappers use the same pattern. Every goal is a full
state-shaped vector and every success predicate ignores at least the desired
hand and gripper coordinates. Task 5 is not exceptional in this respect. In
particular, its fixed goal includes the pressed handle position at z=0.07, so
the actor is explicitly shown that the desired mechanism state is pressed.

This comparison rules out the narrow claim that DCC fails merely because the
goal contains more coordinates than the official success predicate. It does
not prove that every manually synthesized hand/gripper coordinate is optimal,
so mechanism-only goal conditioning remains a valid ablation rather than a
wrapper repair.

## Smoke-test objective

For task `e`, mechanism position `m`, fixed target `g`, official axis `a`, and
threshold `epsilon`, the independent success oracle is

`success = 1[abs(m[a] - g[a]) <= epsilon]`.

The runtime gate requires all of the following:

- the wrapper reports the 150-step training horizon;
- reset is not already successful;
- observation mechanism coordinates match `_get_pos_objects()`;
- emitted goal, `_goal`, and `_target_pos` match the fixed goal;
- a zero-action first transition has no reset-induced mechanism jump;
- every wrapper reward and `info["success"]` matches the independent oracle;
- MetaWorld's scripted policy solves at least 80% of resets within 150 steps.

The rollout may continue to 200 steps only to report successes that occur
after the actual training horizon. Such late success does not pass the gate.

## Code and exact configuration

- `scripts/smoke_test_task58_corrected_wrapper.py` runs the simulator audit.
- `tests/test_task58_corrected_wrapper_smoke.py` checks fail-closed gate logic
  without requiring MetaWorld.
- `DRAFT_task58_wrapper_smoke.sh` runs one evaluation-only GPU job.
- Tasks: `sawyer_handle_press_side`, `sawyer_window_close`.
- Success mode: `corrected`.
- Fixed goals: the historical values in `contrastive/goal_semantics.py`.
- Seeds: 5, 6, 7.
- Episodes per seed and task: 5.
- Training horizon: 150; diagnostic maximum: 200.
- Required scripted-policy success rate by step 150: 0.80.
- Observation tolerance: `1e-5`; goal tolerance: `1e-6`.
- Maximum zero-action mechanism displacement: 0.02.

## Launch

After pulling the commit on the Torch cluster:

```bash
sbatch DRAFT_task58_wrapper_smoke.sh
```

The machine-readable result is written to
`logs/wrapper_smoke/task58_corrected_wrapper.json`. The process exits nonzero
if either task fails any gate. No training or W&B logging occurs.

## Logged metrics

The JSON records reset-success count, maximum observation/goal/internal-target
errors, maximum zero-action displacement, reward and info mismatch counts,
success rate by step 150, success rate by step 200, late-success rate, and the
first success step for every episode.

Protocol V2 also separates benchmark success from complete-goal reachability.
At every officially successful state inside the 150-step horizon it records
the error between the seven semantic state coordinates and the seven
synthetic goal coordinates: hand xyz, normalized gripper opening, and
mechanism xyz. It reports bitwise equality, thresholds at `1e-6`, `1e-3`, and
`1e-2`, and the minimum/mean/maximum L-infinity and L2 errors. The JSON embeds
the closest actually visited successful state. That vector is a candidate
replacement goal which is reachable and successful by construction.

Exact equality is descriptive, not a sensible continuous-control acceptance
gate: even the SGCRL problem statement notes that continuous systems do not
hit a real-valued goal exactly. The useful decision is whether the synthetic
goal lies near the observed successful-state manifold; if it does not, the
captured successful state should replace the invented hand/gripper goal before
rerunning DCC.

## Validation

The dependency-light tests exercise a passing result, the historical Task-8
stale-reset signature, independent reward failure, independent solvability
failure, and the evaluation-only launcher contract. The real simulator result
is intentionally pending the cluster smoke job because MetaWorld/mujoco-py is
not installed in the local project interpreter.

## Known limitations and baseline-launch boundary

This gate certifies the corrected Task-5/Task-8 wrappers only. It does not yet
certify the full ten-task suite. Static comparison against MetaWorld found two
remaining discrepancies outside Tasks 5/8: the local Hammer success check uses
a nail-position distance instead of the native nail-joint condition, and the
local Stick Pull check omits MetaWorld's stick-insertion condition. These may
matter for reward-using baselines even though DCC itself ignores rewards.
Consequently the full nine-baseline sweep should wait for an all-task success
predicate audit (and any resulting repairs) in addition to this smoke passing.
