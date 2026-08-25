# Sawyer goal-wrapper validity sequence and runtime correction

Date: 2026-08-25  
Scope: single-task Task 5 (`sawyer_handle_press_side`) and Task 8
(`sawyer_window_close`) DCC validity experiments

## Motivation

Stage B did not establish that MetaWorld itself is broken. Sparse SAC and
earlier DCC runs solved both tasks at nontrivial rates, while the Stage-B
candidate oracle was only a small one-action/chunk intervention followed by
the current unsuccessful policy. However, the custom continual wrapper had
not been checked end to end against MetaWorld's native success predicate and
scripted behavior policy.

The wrapper audit found a real semantic concern. For both tasks, the custom
state is

$$
s = [h_x,h_y,h_z,q,m_x,m_y,m_z,0,0,0,0],
$$

where $h$ is the hand position, $q$ is gripper opening, and $m$ is the handle
position. The historical desired goal is

$$
g_{\mathrm{full}} =
[m_x^*,m_y^*,m_z^*+0.03,0.4,m_x^*,m_y^*,m_z^*,0,0,0,0].
$$

Yet benchmark success depends only on

$$
y = \mathbb{1}\left[\lVert m-m^*\rVert_2 < \epsilon\right],
$$

with $\epsilon=0.02$ for Task 5 and $\epsilon=0.05$ for Task 8. The desired
hand pose and gripper opening are therefore invented auxiliary constraints:
they need not equal the hand and gripper values in a state that MetaWorld
labels successful. This does not prove that the wrapper caused the failure,
but it makes the wrapper a falsifiable primary suspect.

## Goal contracts

The historical behavior remains the default:

$$
g = g_{\mathrm{full}}, \qquad
\texttt{goal\_conditioning\_mode=full\_state}.
$$

The validity ablation uses exactly the coordinates in the success predicate:

$$
g = g_{\mathrm{success}} = m^*, \qquad
\texttt{goal\_conditioning\_mode=success\_mechanism}.
$$

The ablation changes both desired goals and HER achieved goals because the
standard relabeler now extracts `state[4:7]`. It does not mask the state: the
actor and critic still observe the full robot/mechanism state, but the goal
side contains only the three desired handle coordinates.

`success_mechanism` is deliberately restricted to Task 5/Task 8 and to runs
without the historical task-ID wrapper. Checkpoint paths include the goal
contract so a mechanism-goal run cannot resume or overwrite a full-state run.

## Staged validity sequence

### V0: wrapper and behavior-policy audit (no GPU training)

`scripts/validate_sawyer_goal_wrapper.py` runs MetaWorld's official scripted
policy through the custom wrapper. At every step it bypasses the local
`_get_obs` override, invokes the native parent class's `evaluate_state`, and
compares native `info['success']` with the custom wrapper's positive reward.

For each task and seed it records:

- exposed mechanism-goal versus internal `_target_pos` maximum error;
- native-success availability and custom/native predicate agreement;
- custom and native scripted-policy episode success rates;
- at successful states, mechanism error plus the hand/gripper mismatch to the
  historical full-state goal.

Strict gate:

$$
\begin{aligned}
\lVert g_m-\_target\_pos\rVert_\infty &\le 10^{-6},\\
\operatorname{agreement}(y_{wrapper},y_{native}) &\ge 0.999,\\
\operatorname{success}_{expert}^{wrapper} &\ge 0.80,\\
\operatorname{success}_{expert}^{native} &\ge 0.80.
\end{aligned}
$$

Launch:

```bash
sbatch DRAFT_goal_wrapper_audit.sh
```

All three array jobs (seeds 5, 6, and 7) must pass before V1.

### V1: matched 1M goal-contract falsification

V1 contains eight runs: Task 5/Task 8, seeds 5/6, and full-state/mechanism-only
goals. All other algorithm choices are matched: reset actor, DCC with dynamics,
12 in-trajectory relabels, and no task ID.

Launch:

```bash
sbatch DRAFT_goal_semantics.sh
```

Apply the W&B gate after all eight runs finish:

```bash
python scripts/evaluate_goal_semantics.py
```

Promotion requires on both tasks:

- mean paired success-AUC gain at least $0.05$;
- no paired seed regression below $-0.05$;
- mechanism-goal late/peak success retention at least $0.75$;
- serial diagnostic runtime fraction at most $0.01$.

If this fails, the invented hand/gripper goal is not supported as the primary
cause of the Task-5/Task-8 failure. If it passes, the correct interpretation is
that goal semantics materially contributed; it still does not prove that
critic calibration is fixed.

### V2: full-horizon promotion

The promotion enumerator refuses to run without an explicit passed-gate
environment variable. It runs the mechanism-only contract for seeds 5/6/7 on
both tasks for 8M steps:

```bash
GOAL_VALIDITY_PROMOTED=true sbatch DRAFT_goal_semantics_promotion.sh
```

## Runtime diagnosis

The roughly 10x slowdown and low GPU utilization are expected from the staged
diagnostic hot path, not from the base DCC update alone.

One old Stage-B oracle event ran

$$
4\ \text{conditions}\times
8\ \text{anchors}\times
16\ \text{candidates}\times
100\ \text{rollout steps}
=51{,}200
$$

additional MuJoCo steps. Four events add approximately $204{,}800$ serial CPU
simulator steps to a nominal 100k-step training job, excluding anchor search,
snapshot/restore, policy inference, evaluation, and prefill. MuJoCo candidates
are evaluated in Python one after another; JAX/GPU learning waits for the
entire event.

Even without diagnostics, the sequential runner performs a full 150-step
CPU environment episode and CPU actor inference before one burst of 64 JAX
updates. A single small DCC model therefore cannot continuously saturate an
L40S.

This change makes three practical corrections:

1. V1/V2 set counterfactual ranking, the four-condition oracle, action
   landscape, shortcut diagnostics, and RL representation sweeps to zero/off.
2. `profile_runtime` logs cumulative actor, learner, evaluation,
   representation-metric, and each simulator-diagnostic wall time, plus
   diagnostic fraction, learner fraction, and environment steps/second.
3. The V1/V2 launchers run two independent ordinary learners per L40S with a
   0.45 JAX memory fraction each, filling CPU-collection gaps without the
   previous simulator-probe oversubscription.

For future diagnostic jobs, `counterfactual_oracle_condition_set=promotion_only`
runs only scripted-contact/repeat-5 (about one quarter of the four-condition
rollouts), and `counterfactual_oracle_max_events=N` caps events. Defaults
preserve previous behavior.

## Exact code/config changes

- `contrastive/goal_semantics.py`: explicit goal contracts, validated task
  metadata, goal parsing, and contract metrics.
- `contrastive/counterfactual_outcomes.py`: correct distance handling for both
  11-D historical goals and 3-D mechanism goals.
- `run_continual_contrastive.py`: goal-contract flag/slicing, checkpoint
  separation, W&B manifest fields, runtime profiling, and bounded lean-oracle
  controls.
- `DRAFT.sh`: forwards and prints all new controls.
- `scripts/validate_sawyer_goal_wrapper.py`: V0 expert/native-predicate audit.
- `experiment_configs_goal_semantics.py`, `DRAFT_goal_semantics.sh`: eight V1
  cells.
- `scripts/evaluate_goal_semantics.py`: finished-run and paired-seed V1 gates
  using `evaluator/success_rate` and `evaluator/env_steps`.
- `experiment_configs_goal_semantics_promotion.py` and
  `DRAFT_goal_semantics_promotion.sh`: guarded six-run V2 promotion.
- `DRAFT_goal_wrapper_audit.sh`: CPU-only V0 array.
- `tests/test_goal_semantics.py`: dependency-light contract, metric, config,
  and promotion-guard tests.

## Logged metrics

V0 W&B group: `GOAL-WRAPPER-VALIDITY`

- `validity/task{5,8}/target_linf_error_max`
- `validity/task{5,8}/success_predicate_agreement`
- `validity/task{5,8}/custom_expert_success_rate`
- `validity/task{5,8}/native_expert_success_rate`
- `validity/task{5,8}/successful_hand_goal_distance_mean`
- `validity/task{5,8}/successful_gripper_goal_error_mean`

V1 W&B groups:

- `GOAL-VALIDITY-V1-full-state-task{5,8}`
- `GOAL-VALIDITY-V1-success-mechanism-task{5,8}`

Primary performance keys are `evaluator/success_rate` and
`evaluator/env_steps`. Runtime keys are under `runtime/`, including
`runtime/diagnostic_fraction`, `runtime/learner_fraction`, and
`runtime/env_steps_per_second`.

## Validation performed

- Python compilation for every modified/new Python module.
- Shell syntax checks for the shared and three new launchers.
- Dependency-light goal-contract tests: 5 passed.
- V1 enumeration: exactly 8 matched configs.
- V2 enumeration under the explicit guard: exactly 6 configs.
- The full V0 MuJoCo/MetaWorld audit cannot run in this lightweight workspace;
  the CPU launcher runs it in the project's `contrastive_rl` Torch environment
  and fails closed if native semantics or expert solvability do not validate.

## Known limitations

- The V0 parent-class `evaluate_state` check is intentionally strict and may
  expose a MetaWorld-version API mismatch. That is a validity failure to
  inspect, not something the launcher silently ignores.
- A passing expert test validates task/interface solvability, not learnability
  by DCC.
- A 3-D goal changes the goal encoder input width. This is intentional and is
  the cleanest test of whether nuisance desired-goal coordinates caused the
  issue, but it combines semantic correction with a smaller goal input.
- Two processes per GPU improve duty cycle but do not make serial MuJoCo run
  on the GPU. The W&B runtime fractions, not instantaneous `nvidia-smi`, are
  the reliable attribution metrics for these experiments.

