# DCC shared-scale Task-5 results

Date: 2026-09-04

## Question

Does changing

`f_k = alpha * f_shared + f_task`

control the strength of the shared component in the same way that changing the
strength of a policy prior would?

## Experiment checked

- W&B project: `nyuad_mmvc/continual-contrastive-rl`
- Group: `DCC-SHARED-SCALE-TASK5-BRANCH-1M`
- Task: Task 5, `sawyer_handle_press_side`
- Alpha: `0, 0.25, 0.5, 0.75, 1, 1.5`
- Seeds: `5, 6, 7`
- All 18 branch runs finished.
- Each alpha for a seed starts from the same Task-0-to-4 prefix checkpoint, so
  comparisons within a seed are matched.

Success AUC is the area under the evaluation-success curve divided by the
maximum evaluation step. "Final-3" averages the last three evaluations and is
less sensitive than the single last evaluation.

## Results

| alpha | Best success | Success AUC | Final-3 success | Shared score share | Scaled shared/task norm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.300 +/- 0.100 | 0.075 +/- 0.036 | 0.078 +/- 0.051 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| 0.25 | 0.533 +/- 0.306 | 0.190 +/- 0.121 | 0.222 +/- 0.102 | 0.498 +/- 0.002 | 6.093 +/- 2.523 |
| 0.5 | 0.433 +/- 0.153 | 0.161 +/- 0.109 | 0.167 +/- 0.145 | 0.507 +/- 0.007 | 6.334 +/- 0.920 |
| 0.75 | 0.567 +/- 0.252 | 0.210 +/- 0.144 | 0.200 +/- 0.120 | 0.515 +/- 0.009 | 6.225 +/- 0.831 |
| 1 | **0.700 +/- 0.173** | **0.225 +/- 0.057** | 0.233 +/- 0.067 | 0.505 +/- 0.008 | 6.764 +/- 0.705 |
| 1.5 | 0.533 +/- 0.208 | 0.194 +/- 0.079 | 0.233 +/- 0.067 | 0.498 +/- 0.003 | 7.050 +/- 0.885 |

Values are mean +/- standard deviation across three seeds.

## What the results say

### 1. Removing the shared branch hurts

Comparing alpha=1 with alpha=0 on the same three seed checkpoints:

- Success AUC rises by 0.150 on average. All three seeds improve.
- Best success rises by 0.400 on average. All three seeds improve.
- Final-3 success rises by 0.156 on average. All three seeds improve.
- Success at 500k steps rises by 0.267 on average. All three seeds improve.

This supports the broad idea that the shared component transfers useful
knowledge to Task 5.

### 2. Non-zero alpha is not a reliable strength control

Across alpha=0.25 to 1.5, success does not rise or fall consistently:

- Alpha versus success AUC: Spearman r=0.087, p=0.757.
- Alpha versus best success: Spearman r=0.173, p=0.538.
- Alpha versus final-3 success: Spearman r=0.105, p=0.710.
- A matched-seed test across the five non-zero alpha values gives p=0.569 for
  AUC, p=0.602 for best success, and p=0.781 for final-3 success.

The best alpha also changes by seed: 1.5 for seed 5, 0.75 for seed 6, and 1.0
for seed 7 when judged by AUC.

The positive correlation seen when alpha=0 is included is therefore mainly an
on/off effect: zero removes shared knowledge. It is not evidence for a smooth
policy-prior-like control among non-zero values.

### 3. The networks compensate for alpha

For every non-zero alpha, the shared branch supplies about half of the final
absolute goal score: 0.498 to 0.515. The scaled shared/task representation-norm
ratio also stays in a narrow range, roughly 6 to 7.

At the same time, the raw shared norm falls from 59.31 at alpha=0.25 to 17.92 at
alpha=1.5. Its rank correlation with alpha is -0.982 (p<1e-10). In simple terms,
when alpha gets larger, the shared network makes its unscaled output smaller.
When alpha gets smaller, it makes the output larger. This cancels much of the
fixed multiplier.

The shared/task cosine falls as alpha increases (Spearman r=-0.742, p=0.0015),
so alpha does change the internal geometry. That change does not translate into
a reliable success change.

Categorical and binary accuracy remain almost unchanged and very high for all
alpha values (about 0.98 and 0.998). These metrics show that the critic solves
its classification objective; they do not show that the actor receives a
better action landscape.

## Conclusion

The broad statement "the shared component acts as useful prior knowledge" is
supported. The stronger statement "the current fixed alpha controls the
strength of that prior" is not supported.

In this implementation, alpha behaves mostly like an on/off switch: alpha=0 is
worse, while alpha=0.25 to 1.5 produces similar effective shared contribution
and noisy, non-ordered performance.

For the current paper setup, keep alpha=1. It has the best aggregate mean and
there is not enough evidence to tune a different fixed value.

A trainable scalar alpha is unlikely to help because the branch norms can absorb
it, making alpha hard to interpret or identify. Before testing alpha(s,g), first
make the mixture identifiable: normalize or calibrate the shared and task scores,
then mix them with a bounded gate, and prevent immediate compensation (for
example, freeze or stop the gradient through the shared branch while learning
the gate). Repeat the fixed-alpha sweep under that design. Only if the effective
shared contribution and performance then move with alpha is a trainable
state-goal gate justified.

## Reproducibility

The exact downloaded histories and summaries are stored in
`docs/wandb_analysis/dcc_shared_scale_task5/`. They can be regenerated with
`scripts/analyze_dcc_shared_scale_wandb.py` after setting `WANDB_API_KEY`.
