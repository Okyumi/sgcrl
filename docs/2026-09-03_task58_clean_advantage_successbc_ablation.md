# Corrected Task-5/Task-8 Advantage and terminal-SuccessBC ablation

Date: 2026-09-03
Status: implemented and locally validated; not yet launched

## Purpose

The corrected-wrapper runs show that plain DCC is unreliable on Tasks 5 and
8, while raw-H25 Advantage-DCC plus SuccessBC reached stable high success on
Task 5. That combined result changed two mechanisms at once. This pilot
separates them without repeating the existing plain-DCC control.

## Experiment matrix

Every variant uses the corrected original Sawyer wrapper, the reachable
full-state goal, a 1M-step single-task budget, dynamics auxiliary weight 1,
and seeds 5 and 6 on both `sawyer_handle_press_side` and
`sawyer_window_close`.

1. `advantage_1step`: original Advantage-DCC. The task-local head predicts
   the one-step normalized goal-embedding displacement
   `gamma * psi_hat(s_next) - psi_hat(s)`; the actor maximizes normalized DCC
   score plus the goal-directed action-effect score.
2. `advantage_h25`: raw H=25 outcome head without SuccessBC. For raw
   mechanism-state distance `d` in metres, the progress target is
   `d(s_t,g) - min_{h=1..25} d(s_{t+h},g)` and the second label reports whether
   any of those distances is at most 0.05. This is a privileged diagnostic,
   not a method satisfying the terminal-success-only protocol.
3. `terminal_success_bc`: ordinary decomposed DCC without an action-effect
   head. A separate task-local buffer accepts transitions only when the final
   sparse reward of that replay episode is positive. The actor receives the
   existing BC loss with weight 0.1 on 64 retained examples per update.

The three W&B groups are deliberately explicit:

- `TASK58-CORRECTED-ADVANTAGE-1STEP-1M`
- `TASK58-CORRECTED-ADVANTAGE-H25-1M`
- `TASK58-CORRECTED-TERMINAL-SUCCESS-BC-1M`

## Terminal-success-only implementation

The historical SuccessBC mode is preserved as `raw_horizon`. A new
`success_bc_label_mode=terminal_episode` path reads only the last transition
reward of the sampled episode. If it is positive, the episode's original
task-goal `(state, action)` examples are eligible for the success ring buffer.
It performs no H-step lookahead, mechanism-coordinate selection, distance
calculation, or task-specific thresholding.

The ordinary replay buffer remains unchanged and contains both successful and
unsuccessful episodes. The added ring buffer contains only examples from
terminally successful episodes. It changes only the actor objective; DCC
positive/negative sampling and critic training are unchanged.

## Runtime decision

The launcher uses one run per L40S and requests 12 CPU cores. This avoids the
previous two-process GPU/CPU contention. The one-step and terminal-BC cells do
not perform the expensive H=25 scan. Only the four explicitly named H25
diagnostic cells retain that preprocessing cost. The Slurm array is capped at
six concurrent jobs.

## Launch

```bash
cd /scratch/yd2247/sgcrl
git pull --ff-only origin section3_done
sbatch DRAFT_task58_algorithm_ablation.sh
```

The array has 12 cells (`0-11%6`): three variants times two tasks times two
seeds. The launcher is self-contained and intentionally does not call the
repository's generic `DRAFT.sh`.

## Metrics and interpretation

Primary metrics are corrected `evaluator/task58/task_axis_success`, late-five
evaluation success, and success AUC. `success_reward_mismatch_steps` must stay
zero. SuccessBC additionally logs `retention/buffer_size`, `retention/bc_loss`,
`retention/bc_active`, and `retention/source_success_fraction`.

- One-step Advantage success would support representation-space local action
  credit without task-specific outcome knowledge.
- H25-only success would show that denser privileged outcome labels, not BC,
  supplied the improvement.
- Terminal-SuccessBC success would show that retaining complete successful
  trajectories is sufficient and is the likely source of the earlier
  combined result.

## Validation

- Python compilation for changed and added Python files.
- Shell syntax for `DRAFT_task58_algorithm_ablation.sh`.
- Dependency-light 12-cell matrix and wiring test in
  `tests/test_task58_algorithm_ablation.py`.
- Existing corrected-wrapper configuration and axis-reward tests.

## Limitations

- Two seeds are a pilot, not a final estimate.
- A terminal success label provides trajectory-level credit, not exact
  per-action causal credit. Behavior cloning retains the whole successful
  trajectory, including potentially unnecessary actions.
- The H25 method uses privileged mechanism geometry and falls outside the
  terminal-success-only protocol; it is included only to separate the earlier
  combined result.
