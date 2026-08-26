# Task-8 reset-observation repair

Date: 2026-08-26
Status: implemented and covered by dependency-light tests.

## Motivation and diagnosis

The corrected-wrapper Task-8 runs reported an impossible pattern. At reset,
the window handle appeared approximately 0.01 m from its target, already
inside the 0.05 m success threshold. After the first environment step it
appeared approximately 0.20 m farther away. Across seeds 5, 6, and 7, the
logged quantities obeyed

$$
d_{\mathrm{reset}} = d_{\min} + \Delta_{\mathrm{progress}} = 0.010\;\mathrm{m}.
$$

The pattern did not depend on the learned action and therefore pointed to an
inconsistent reset observation rather than policy behavior.

The versioned MetaWorld `SawyerWindowCloseEnvV2.reset_model()` implementation
confirms the mechanism. It queries the handle site, writes
`window_slide = 0.2`, and immediately returns an observation. It does not call
`sim.forward()` after writing the joint position. With mujoco-py, the derived
site positions remain stale until a forward pass or physics step. The local
wrapper then called `_get_obs()` again without forwarding, so replay and the
Task-5/Task-8 stage observer received a false near-goal reset frame.

## Code change

`contrastive/sawyer_success.py` now exposes
`synchronize_simulator_after_reset(environment)`. It requires and invokes
`environment.sim.forward()` exactly once. Missing simulator support raises an
explicit error rather than silently reintroducing the defect.

`SawyerWindowClose.reset()` calls this helper immediately after the MetaWorld
parent reset and before constructing the local 22-dimensional observation.
No step is taken, no action is applied, and no reward or learning objective is
changed. The repair is deliberately limited to Window-Close because its
versioned reset routine contains the verified post-observation joint write;
the Task-5 reset data did not show this discontinuity.

The existing Task-5/Task-8 trajectory observer now also emits
`initial_task_axis_distance`, so each W&B evaluation records the reset distance
directly instead of requiring it to be reconstructed from two other metrics.

## Mathematical objective

There is no new optimization objective. DCC remains reward-free. The repair
enforces the observation invariant

$$
o_0 = h(x_{\mathrm{reset}}),
$$

where both MuJoCo generalized coordinates and derived site/body positions in
`x_reset` describe the same physical configuration.

## Validation

Run the dependency-light checks with:

```bash
python tests/test_sawyer_native_success_wrapper.py
python tests/test_task58_dcc_corrected.py
```

The tests verify that synchronization forwards the simulator exactly once,
fails explicitly when no forward method exists, and occurs in the Task-8
reset path after the parent reset but before the observation is returned.

On the cluster, the simulator-level acceptance check is:

1. reset `sawyer_window_close`;
2. verify the handle's initial x-axis distance is approximately 0.20 m, not
   0.01 m;
3. apply a zero action and verify no action-independent approximately 0.20 m
   discontinuity occurs;
4. confirm Task-8 `max_task_axis_progress` is no longer mechanically shifted
   downward by approximately 0.19 m.

## Logged metrics

One direct reset metric is added to the existing metrics under
`evaluator/task58/`; together they provide the regression signal:

- `task_axis_min_distance`;
- `initial_task_axis_distance`;
- `max_task_axis_progress`;
- `max_mechanism_axis_displacement`;
- `task_axis_success`;
- `success_reward_mismatch_steps`.

## Known limitations

- The local development environment does not include the project MetaWorld
  and mujoco-py runtime, so the real-simulator acceptance check must run in the
  existing cluster environment before interpreting new Task-8 learning data.
- Old Task-8 replay buffers and checkpoints were trained with the false reset
  frame. Evaluation can use the repaired reset, but a clean causal comparison
  requires a fresh Task-8 run.
- This repair does not establish why Task 5 underperforms and does not by
  itself establish that Task 8 will learn successfully.
