# Reachable Task-5/Task-8 goals and cheap Stick-Pull repair

Date: 2026-08-27
Status: Task-5/Task-8 V3 and Stick-Pull V2 simulator audits passed; robust
Stick-Pull goal integration implemented, with V3 post-integration smoke
pending.

## What the Task-5/Task-8 V2 audit established

The evaluation-only wrapper audit passed every dynamics, reset, horizon, and
success-predicate gate. It also falsified the remaining synthetic full-goal
assumption.

For both tasks, the exposed goal used gripper value `0.4`, while every
successful scripted-policy state was near `0.29668`. This creates an
unavoidable full-goal L-infinity error of about `0.1033`. There were zero
visits within `1e-2`, `1e-3`, or `1e-6` of either complete synthetic goal.

The audit captured one state that was actually visited while the corrected
task predicate was true for each task. Corrected and native-info runs now use
those captured seven-coordinate states as their conditioned goals. The
simulator mechanism target remains unchanged:

- Task 5 still targets `[-0.07, 0.68, 0.07]` and succeeds on handle z within
  `0.02`.
- Task 8 still targets `[0.0, 0.80, 0.20]` and succeeds on window x within
  `0.05`.

Legacy-distance mode retains the old synthetic hand/gripper/mechanism goal so
historical checkpoints and baseline reproduction remain available.

The six clean single-task DCC jobs now write to a new W&B group and new log and
checkpoint directories. This prevents an earlier corrected-reward-only run
from being resumed as if it used the reachable goal.

## Rerun sequence for DCC

First validate that the selected reachable goals are exposed correctly and
remain successful under the cluster simulator:

```bash
sbatch DRAFT_task58_wrapper_smoke.sh
```

The output protocol must be `task58_corrected_wrapper_smoke_v3`, all gates must
pass, and the conditioned goal printed for each task must match the captured
reachable-success goal.

Then run the twelve single-task DCC cells:

```bash
sbatch DRAFT_task58_dcc_corrected.sh
```

This runs Tasks 5 and 8 with seeds 5, 6, and 7 for one million steps each,
once with the dynamics auxiliary and once without it. The resulting W&B group
is `TASK58-DCC-CORRECTED-DYN-ABLATION-1M`. The existing lightweight evaluation
metrics separate failure to approach, engage, or move the mechanism.

## Cheap Stick-Pull success repair

The historical wrapper declared success whenever the handle was within `0.12`
of the target. That accepts the direct-push behavior visible in the evaluation
video.

Corrected mode now mirrors MetaWorld's task predicate. Success requires both:

1. handle-to-target distance at most `0.12`; and
2. the far stick endpoint inserted through the handle, meaning it is beyond
   the handle in x and within `0.040` in y and `0.060` in z.

This adds only one site-position read and scalar comparisons. It does not add a
simulator step, alter the observation, or make DCC consume reward. Legacy mode
continues to use handle distance only, and native-info mode continues to use
MetaWorld's own `info["success"]`.

## Why the full ten-seed sweep is still gated

The reward predicate is now cheap and correct, but Stick-Pull's current DCC
goal remains inconsistent. The state contains hand, gripper, stick center, and
handle, while the old goal places the stick center and handle at the same
target and contains no insertion coordinate. Correcting reward alone cannot
change reward-free DCC training or prevent the critic from ignoring stick use.

The new evaluation-only audit runs the official scripted policy, compares the
corrected wrapper reward and info to the independent insertion predicate, and
captures the most robust successful state already present in those
trajectories. Robustness maximizes the minimum slack across the handle-distance
and insertion gates. The audit also computes a signed insertion margin that
fits the existing unused `state[10]` slot and is nonnegative exactly when all
insertion gates pass:

```bash
sbatch DRAFT_stick_pull_wrapper_smoke.sh
```

After that output passes, use its `captured_successful_state` to version the
corrected Stick-Pull state/goal contract. The expensive ten-seed, nine-plus-one
pipeline should start only after this goal patch and a short all-task runtime
smoke pass. The native-info pipeline already uses MetaWorld's official Hammer
success and therefore avoids the other known hand-written predicate mismatch.

## 2026-08-31 results and launch decision

Task-5/Task-8 V3 passed every gate. Both tasks succeeded in all 15 episodes,
and each corrected full-state goal was revisited exactly. This promotes the
single-task DCC validation.

Stick-Pull V2 also passed every gate: 12 of 15 scripted episodes succeeded,
and reward and info had zero disagreements with the independent official
predicate. The robust selected state has success slack `0.007919`, compared
with only `0.002561` for the first-success state selected by V1. Corrected and
native-info fixed-goal runs now expose that robust 11-coordinate state and use
`state[10]` for the signed insertion margin. Legacy mode retains its historical
ten-coordinate state/goal plus zero padding.

The Task-5/Task-8 DCC validation is expanded to a matched dynamics ablation:
Tasks 5 and 8, dynamics auxiliary weights `1.0` and `0.0`, and seeds 5, 6, and
7, for twelve one-million-step runs. The weight-zero cell is the registered
"DCC without dynamics" ablation; it preserves the decomposed critic while
disabling the dynamics auxiliary loss.

The guarded full experiment is now an explicit 11-variant by 10-seed matrix:
the nine `{reset, persistent, CKA}` actor/critic combinations plus DCC with
the dynamics auxiliary on and off, all on matched seeds 5 through 14. This is
110 runs, not a nine-plus-one grid. It remains locked until the twelve Task-5/8
DCC runs and the short ten-task wrapper smoke pass.

## Local validation

The dependency-light tests cover:

- successful coordinates and reachable gripper values for the captured
  Task-5/Task-8 goals;
- legacy isolation;
- corrected Stick-Pull success, direct-push rejection, insertion boundaries,
  and handle-target distance;
- fail-closed Task-5/Task-8 and Stick-Pull smoke classifications;
- Python compilation and launcher syntax.

Real MuJoCo validation remains a cluster step because the local project
interpreter does not include the pinned simulator stack.

## 2026-08-31 launcher fix (job 16659678)

Array job `16659678` failed immediately on array task 0 with
`ModuleNotFoundError: No module named 'numpy'` because
`DRAFT_task58_dcc_corrected.sh` ran the dependency-light preflight tests
before activating `contrastive_rl` and sourcing `set_up/torch_hpc_env.sh`.
The launcher now mirrors `DRAFT_goal_semantics.sh`: conda activation and
`MUJOCO_GL=egl` happen before `tests/test_task58_dcc_corrected.py`.

Resubmit with:

```bash
sbatch DRAFT_task58_dcc_corrected.sh
```
