# Distral-faithful normalized DCC alpha test on Task 5

Date: 2026-09-04

## Question

Does the coefficient on DCC's shared contrastive score act like the strength
of the shared policy prior in Distral?

The completed unnormalized sweep could not answer this cleanly because the
learned branch magnitudes compensated for alpha. The later `unit_mix` design
removes that compensation but divides the score by `alpha + 1`, which changes
the exact policy mapping. This experiment keeps normalization while removing
that division.

## Three distinct score definitions

The completed original sweep used

`f = alpha * <z_shared, z_goal> + <z_task, z_goal>`.

The fixed-budget `unit_mix` control uses

`f = (alpha * <u_shared, u_goal> + <u_task, u_goal>) / (alpha + 1)`.

The new primary Distral test uses

`f = alpha * <u_shared, u_goal> + <u_task, u_goal>`,

where each `u` is its branch embedding divided by its own L2 norm. It is a
third setup: it is neither the unnormalized completed sweep nor the bounded
fixed-budget mixture.

## Why the division by alpha plus one was proposed

Once both branches have unit norm, increasing alpha increases both the
shared-to-task ratio and the overall score scale. Dividing by `alpha + 1`
makes the two scalar coefficients sum to one, which approximately holds the
overall score budget fixed. That is useful for asking whether a fixed budget
should be allocated to shared or task-specific features.

It is not the closest test of the Distral proof. With a fixed actor entropy
temperature, multiplying the whole score by `1 / (alpha + 1)` also changes the
effective softmax temperature. The shared-policy exponent becomes
`alpha / (alpha + 1)`, not alpha, and the task-score coefficient changes too.

## Policy-level derivation

For nonzero branch outputs, define

`u_shared = phi_shared(s,a) / ||phi_shared(s,a)||_2`,

`u_task = phi_task(s,a) / ||phi_task(s,a)||_2`,

and

`u_goal = psi(g) / ||psi(g)||_2`.

Let

`f_shared_hat = <u_shared, u_goal>`

and

`f_task_hat = <u_task, u_goal>`.

The new critic score is

`f_alpha = alpha * f_shared_hat + f_task_hat`.

For a fixed critic, the maximum-entropy actor objective is

`J = E_pi[f_alpha - tau * log pi]`.

Define the goal-conditioned reference policy induced by the normalized shared
score:

`pi_0(a|s,g) = exp(f_shared_hat / tau) / Z_0(s,g)`.

Therefore

`f_shared_hat = tau * log pi_0 + tau * log Z_0`.

Substitution gives

`J = E_pi[f_task_hat + alpha*tau*log pi_0 - tau*log pi] + const`.

Using

`alpha*log pi_0 - log pi`

`= -alpha*(log pi - log pi_0) - (1-alpha)*log pi`,

the actor objective becomes

`J = E_pi[f_task_hat]`

`    - alpha*tau*KL(pi || pi_0)`

`    + (1-alpha)*tau*H(pi) + const`.

The unrestricted soft optimum is equivalently

`pi*(a|s,g) proportional to`

`pi_0(a|s,g)^alpha * exp(f_task_hat(s,a,g) / tau)`.

This is the same shared-prior-times-task-correction policy factorization used
in the supplied DCC-Distral proof. Normalization changes how the two scores are
constructed, but it does not alter the algebra because the final score still
has the form `alpha * shared + task`.

The safe claim remains local: for a fixed critic and an unrestricted
maximum-entropy policy update, the actor objective has the Distral form. DCC
still learns its shared object through contrastive and dynamics losses rather
than Distral's multitask distillation objective, and the implemented Gaussian
actor only approximates the unrestricted soft optimum.

For `0 <= alpha <= 1`, the mapping in the supplied proof is

`c_KL = alpha * tau`,

`c_Ent = (1 - alpha) * tau`,

and `beta = 1 / tau`.

The alpha=1.5 cell is retained because it was requested in the original sweep,
but it should be reported as a stronger-than-standard-Distral extrapolation.

## Experiment

Reuse the completed seed-5/6/7 checkpoints after Tasks 0-4 and train only Task
5, `sawyer_handle_press_side`, for one million steps. This preserves the same
starting shared representation for every alpha and costs only 18 million new
environment steps.

The task uses plain DCC with its original dynamics auxiliary. It has no
behavior cloning, action-effect or advantage head, Bellman Q critic,
counterfactual supervision, or extra negative repeats. Only the normalized
score equation changes.

The reused prefix was trained with original alpha=1. This makes the pilot a
controlled test of changing prior strength at the Task-5 boundary, not proof
that one alpha is optimal for an entire curriculum. If the pilot produces a
stable ordered effect, the rigorous follow-up is to apply the normalized rule
throughout a fresh continual run.

## Manipulation checks and outcomes

Under `unit_distral`, the effective branch norm ratio must equal alpha up to
numerical precision. The important logged quantities are:

- `decomp/shared_scale`, `decomp/shared_coefficient`, and
  `decomp/task_coefficient`;
- raw and effective shared/task norms;
- `decomp/scaled_shared_to_task_norm`;
- shared/task goal scores, shared score fraction, and branch cosine;
- success curve, success AUC, best success, final-three success, critic
  accuracies, losses, and actor entropy.

Support for the hypothesis requires alpha to change the effective shared
contribution and for learning behavior to change consistently with that
contribution across matched seeds. If the norm-ratio check passes but
performance remains unordered, alpha is a real prior-strength control but Task
5 does not show a stable behavioral preference for a particular strength.

## Launch

The Tasks-0-to-4 prefix does not need to be rerun. Confirm its checkpoint
directory still exists, then run:

```bash
sbatch DRAFT_dcc_distral_unit_shared_scale_task5.sh
```

W&B group:

`DCC-DISTRAL-UNIT-SHARED-SCALE-TASK5-BRANCH-1M`

After all 18 runs finish, download the same learning curves and mechanism
metrics used for the original sweep with:

```bash
python scripts/analyze_dcc_shared_scale_wandb.py \
  --group DCC-DISTRAL-UNIT-SHARED-SCALE-TASK5-BRANCH-1M \
  --output-dir docs/wandb_analysis/dcc_distral_unit_shared_scale_task5
```
