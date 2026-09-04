# Normalized DCC shared-scale Task-5 sweep

Date: 2026-09-04

## Motivation

The first Task-5 alpha sweep used

`z(s,a) = alpha * z_shared(s,a) + z_task(s,a)`.

It showed a clear on/off effect: removing the shared branch hurt. It did not
show a reliable ordering among non-zero alpha values. The main reason is that
both branches continue learning and their output magnitudes are unconstrained.
When alpha changes, the networks can change their raw norms in the opposite
direction. In the completed sweep, the raw shared norm fell from 59.31 at
alpha=0.25 to 17.92 at alpha=1.5, while the effective shared score fraction
stayed near 50%.

Therefore the original alpha changes both the intended branch balance and the
overall score scale, and the learned networks can partially absorb it.

## Normalized objective

For every state-action sample, define

`u_shared = z_shared / max(||z_shared||_2, 1e-8)`

`u_task = z_task / max(||z_task||_2, 1e-8)`

and for every goal define

`u_goal = z_goal / max(||z_goal||_2, 1e-8)`.

The normalized score is

`f_alpha(s,a,g) = <(alpha*u_shared + u_task)/(alpha+1), u_goal>`.

The effective branch coefficients are now

`w_shared = alpha/(alpha+1)` and `w_task = 1/(alpha+1)`.

Thus alpha=0.25 means 20% shared and 80% task, alpha=1 means 50%/50%, and
alpha=1.5 means 60% shared and 40% task. Because each raw branch is converted
to unit length before mixing, changing a branch's raw norm cannot undo these
weights. Dividing by alpha+1 keeps the total coefficient mass equal to one, so
the sweep changes branch balance without deliberately changing the overall
logit scale.

This mode is restricted to additive, inner-product DCC with `repr_norm=False`.
The legacy default remains `shared_repr_normalization=none`, so all previous
commands preserve the original equation and checkpoint parameter shapes.

## Why reuse the existing Tasks-0-to-4 checkpoints

The new mode adds no parameters and changes no parameter shapes. The completed
prefix checkpoints can therefore be loaded directly.

Reusing them is the stronger first causal test for the narrow question: every
alpha starts from exactly the same learned shared knowledge, and only the Task-5
mixing rule changes. Retraining Tasks 0-4 would add prefix-training variation
and cost while making the first comparison less controlled.

There is one limitation: Tasks 0-4 learned under the original unnormalized
score. If the normalized Task-5 result is promising and the method is proposed
as a complete algorithm, a later full continual run from Task 0 is required to
measure end-to-end performance under normalization.

## Code and configuration

- `contrastive/decomposed_networks.py` implements `unit_mix` consistently in
  the full score matrix, matched actor score, and diagnostic score paths.
- `contrastive/continual_learning_decomposed.py` computes diagnostics from the
  effective normalized branch terms.
- `run_continual_contrastive.py` adds
  `--shared_repr_normalization=none|unit_mix` and records it in W&B.
- `experiment_configs_dcc_normalized_shared_scale_task5.py` creates the same
  six-alpha, three-seed Task-5 sweep using the existing prefix directory.
- `DRAFT_dcc_normalized_shared_scale_task5.sh` is the direct Torch launcher.

## Launch

The original prefix jobs do not need to be rerun. After confirming that

`/scratch/yd2247/sgcrl/logs/dcc_shared_scale/task5_prefix5/checkpoints`

still contains the completed seed-5/6/7 Task-4 checkpoints, submit:

```bash
sbatch DRAFT_dcc_normalized_shared_scale_task5.sh
```

The launcher runs six array jobs. Each job runs the three matched seeds for one
alpha, for 18 Task-5 branches and 18 million new environment steps total.

W&B group:

`DCC-NORMALIZED-SHARED-SCALE-TASK5-BRANCH-1M`

## Metrics

Primary outcomes remain Task-5 best success, success AUC, final-three success,
and steps to fixed success thresholds. The branch diagnostics include:

- `decomp/shared_coefficient` and `decomp/task_coefficient`;
- `decomp/shared_norm` and `decomp/task_norm` for raw learned norms;
- `decomp/scaled_shared_norm` and `decomp/effective_task_norm` after unit
  normalization and mixing;
- `decomp/scaled_shared_to_task_norm` for the effective branch ratio;
- `decomp/shared_goal_score_abs`, `decomp/task_goal_score_abs`, and
  `decomp/shared_score_fraction` for actual score contribution;
- `decomp/shared_task_cosine`, critic accuracies, losses, entropy, and normal
  evaluation success.

Under `unit_mix`, the effective norm ratio should be alpha, up to numerical
precision: 0, 0.25, 0.5, 0.75, 1, or 1.5. This is the key manipulation check.

## Interpretation

Evidence that alpha is now a real strength control requires both:

1. the effective branch ratio and shared score contribution move with alpha;
2. performance changes consistently with those contributions across matched
   seeds.

If the manipulation check passes but performance is still unordered, then the
shared branch helps as reusable knowledge but Task 5 does not prefer a stable
intermediate sharing strength. If performance becomes ordered or peaks
reliably at an intermediate alpha, a trainable bounded alpha or alpha(s,g)
becomes justified.

## Validation and limitations

`tests/test_dcc_normalized_shared_scale.py` checks the 18 matched configs,
checkpoint reuse, bounded weights, legacy default, runner wiring, scoring-path
coverage, diagnostics, and launcher flag.

The pilot has only three seeds and one downstream task. It can justify or reject
the mechanism on Task 5, but a proposed general method still needs more seeds
and at least one additional task or a full continual run.
