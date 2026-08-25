# Direct Task-5/Task-8 axis-success wrapper correction

Date: 2026-08-25
Status: wrapper evaluator implemented; the proposed 1M training matrix was
withdrawn in favor of checkpoint-only reevaluation.

## Purpose

The historical Task-5/Task-8 wrapper had a real success-label error, but DCC is
reward-free: its actor and contrastive critic do not consume the environment's
reward. Changing only the emitted sparse reward cannot change DCC gradients.
It can only change evaluation and reveal false negatives in old success
curves.

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

## Correct minimal check

Do not retrain DCC to test this correction. Load the historical checkpoints
and score the old and corrected predicates on the same deterministic rollout.
That procedure is documented in
`2026-08-25_task58_checkpoint_reevaluation.md`.

The withdrawn files `experiment_configs_task58_axis_reward.py` and
`DRAFT_task58_axis_reward.sh` were removed so the redundant four-run training
matrix cannot be submitted accidentally.

## Code changes

- `contrastive/sawyer_success.py`: dependency-light `task_axis` reward helper.
- `env_utils.py`: Task 5 uses axis 2 at threshold 0.02; Task 8 uses axis 0 at
  threshold 0.05; other tasks reject this mode.
- `run_continual_contrastive.py`: exposes `--sawyer_success_mode=task_axis` and
  keeps checkpoint identities separate from historical runs.
- `tests/test_task58_axis_reward.py`: reward and wrapper-wiring checks.

## Limitations

This corrects the reported success definition only. It is not presented as a
learning intervention for reward-free DCC.
