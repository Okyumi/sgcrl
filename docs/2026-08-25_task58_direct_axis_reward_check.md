# Direct Task-5/Task-8 axis-reward DCC check

Date: 2026-08-25
Status: implemented; four 1M-step DCC cells are ready to run.

## Purpose

This is the minimal experiment needed to answer whether the historical sparse
reward is responsible for DCC's weak Task-5 and Task-8 performance. It keeps
the existing Sawyer wrapper, full-state goal, DCC learner, replay, HER, actor,
and simulator path unchanged. It does not use MetaWorld reward or success
information.

The only change is the local sparse-reward comparison:

$$
r_5 = \mathbb{1}[|z_{handle}-z_{goal}|\le 0.02],
\qquad
r_8 = \mathbb{1}[|x_{window}-x_{goal}|\le 0.05].
$$

The historical wrapper instead used a full three-dimensional Euclidean
distance. The new mode is named `task_axis`; `legacy_distance` remains the
default until the performance check is complete.

## Runtime

`task_axis` reads the mechanism position already used by the wrapper and makes
one scalar comparison. It adds no environment rollout, native-observation
conversion, reward reconstruction, or simulator step, so its training cost is
effectively the same as the original wrapper.

## Minimal performance check

The matrix contains only four runs:

- Task 5 and Task 8;
- seeds 5 and 6;
- reset actor and decomposed DCC critic;
- one million steps per run;
- the historical `full_state` goal contract;
- all counterfactual, oracle, action-landscape, shortcut, and representation
  diagnostics disabled.

Two learners share each L40S. Launch with:

```bash
sbatch DRAFT_task58_axis_reward.sh
```

W&B group:

```text
TASK58-DCC-AXIS-REWARD-1M
```

Use `evaluator/success_rate` and `evaluator/env_steps` to compare the curves
with the existing Task-5/Task-8 DCC runs. If both tasks improve clearly, make
the axis predicates the normal Task-5/Task-8 wrapper behavior and rerun only
the paper comparison that depends on those tasks. If they do not improve,
the reward bug is real but is not the main cause of DCC's performance failure.

## Code changes

- `contrastive/sawyer_success.py`: dependency-light `task_axis` reward helper.
- `env_utils.py`: Task 5 uses axis 2 at threshold 0.02; Task 8 uses axis 0 at
  threshold 0.05; other tasks reject this mode.
- `run_continual_contrastive.py`: exposes `--sawyer_success_mode=task_axis` and
  keeps checkpoint identities separate from historical runs.
- `experiment_configs_task58_axis_reward.py`: four minimal DCC cells.
- `DRAFT_task58_axis_reward.sh`: two-array Torch launcher, two cells per GPU.
- `tests/test_task58_axis_reward.py`: reward, wiring, config, and launcher
  checks.

## Limitations

One million steps is a fast signal check, not a final paper horizon. It is
enough to determine whether corrected rewards produce a materially healthier
learning curve before spending time on longer runs. The experiment isolates
reward semantics; it does not change DCC itself.
