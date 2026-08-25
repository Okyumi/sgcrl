# Corrected original wrapper and clean Task-5/Task-8 DCC baselines

Date: 2026-08-25
Status: implemented; six individual-task 1M-step runs are ready.

## Wrapper decision

The original custom Sawyer wrapper now defaults to `corrected` success:

$$
y_5=\mathbb{1}[|z_{handle}-z_{goal}|\leq0.02],\qquad
y_8=\mathbb{1}[|x_{window}-x_{goal}|\leq0.05].
$$

All other Sawyer tasks retain their existing local success predicates. The
historical full-3D rule remains available as the explicit
`legacy_distance` mode so old experiments and checkpoints can still be
reproduced. Corrected checkpoints receive `_success_corrected` in their path
and cannot silently resume from or overwrite legacy checkpoints.
Single-task DCC checkpoint paths also include the environment name even when
the in-trajectory repetition factor is one; this prevents Task 5 and Task 8
from both writing `task_0.pkl` into the same seed directory.

The lightweight `task_axis` spelling remains an alias for existing scripts,
but `corrected` is the normal repository behavior. No MetaWorld native
observation or reward adapter is used in this path.

## Individual-task DCC baseline

The new matrix contains six cells: Task 5 and Task 8, seeds 5/6/7. Each trains
an ordinary reset-actor/decomposed-critic DCC policy for one million steps,
using the historical full-state goal, 1024-wide residual actor/critic, dynamics
auxiliary weight 1.0, and no task ID.

All simulator-based counterfactual, oracle, action-landscape, shortcut, and
representation sweeps are disabled. This is a clean DCC baseline, not another
hypothesis intervention. Launch with:

```bash
sbatch DRAFT_task58_dcc_corrected.sh
```

W&B group:

```text
TASK58-DCC-CORRECTED-WRAPPER-1M
```

## Metrics from ordinary evaluation trajectories

At every 50k steps, the existing deterministic evaluation episodes now report
the following under `evaluator/task58/` without taking any extra simulator
steps:

- `task_axis_success`: corrected success rate;
- `legacy_success`: old full-3D success rate on the same trajectories;
- `axis_rescued_success`: false negatives caused by the historical metric;
- `approach_success`: fraction of episodes reaching hand--mechanism distance
  at most 0.09;
- `interaction_step_fraction`: fraction of evaluation steps in that region;
- `minimum_hand_mechanism_distance`;
- `mechanism_moved`: fraction of episodes moving the official success axis by
  at least 0.005;
- `max_mechanism_axis_displacement`;
- `max_task_axis_progress`: reduction in official axis distance;
- `legacy_min_distance` and `task_axis_min_distance`;
- `success_reward_mismatch_steps`, which must remain zero or the run aborts.

Together these form a small failure-stage funnel:

1. low `approach_success`: reaching/exploration failure;
2. high approach but low `mechanism_moved`: contact/control failure;
3. high movement but low axis success: insufficient or wrong-direction
   mechanism displacement;
4. high axis success but low legacy success: evaluation-only false negatives.

The ordinary learner logs still provide critic loss, positive/negative logits,
categorical accuracy, actor loss, and entropy. Expensive representation sweeps
are disabled to keep the run close to base DCC throughput.

## Interpretation boundary

DCC remains reward-free, so the corrected wrapper flag itself does not enter
the actor or critic loss. These runs provide trustworthy evaluation, current
checkpoints, and stage metrics for deciding what actually limits each task.
They are not described as a reward-based learning improvement.
