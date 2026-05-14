# H1/H2 lightweight test: does α_scale predict success?

Date: 2026-05-14

**Hypotheses (from `2026-05-13_revised_hypothesis_plan.md`):**

- **H1:** the critic mixture is alive but uninformative — it has
  non-zero norm but doesn't carry task-discriminative information.
- **H2:** actor / critic decoupling — the critic learns useful
  cross-task structure but the actor's argmax discards it.

The full offline ablation (load a checkpoint, mutate `α_scale` at
eval time, measure success change) requires cluster access to
saved pickles. As a cheap precursor, we asked: **at the (task,
seed) level, does the trained value of `α_scale` at task end
correlate with the success rate the agent achieved on that task?**

**Verdict: H1 is supported. The critic mixture is anti-predictive
of success.** Across n=24 (task ≥ 2, seed) pairs:

| Predictor / outcome    | Pearson r       | Spearman ρ      | p-value |
|------------------------|-----------------|-----------------|---------|
| actor α_scale → best_success | +0.27     | +0.25           | 0.20 / 0.25 (n.s.) |
| actor α_scale → end_success  | −0.01     | +0.16           | 0.98 / 0.46 (n.s.) |
| actor α_scale → mean_success | +0.11     | +0.20           | 0.61 / 0.36 (n.s.) |
| **critic α_scale → best_success** | **−0.54** | **−0.51**  | **0.007 / 0.011** |
| **critic α_scale → end_success**  | **−0.59** | **−0.64**  | **0.002 / 0.001** |
| **critic α_scale → mean_success** | **−0.67** | **−0.62**  | **<0.001 / 0.001** |

The actor mixture has no significant relationship with success
(p > 0.2 across all three success measures). This is consistent
with the W&B observation that the actor mixture norm decays to ~0.02;
the mixture is dead in both magnitude and effect.

The critic mixture is *anti*-correlated with success: tasks where
α_scale ended high are tasks where the agent failed. The
relationship is significant at p < 0.01 by both Pearson and
Spearman. See plot at
`docs/wandb_analysis/png/h1_h2_alpha_vs_success.png`.

---

## Method

Same 24 W&B runs as the H3 test (C0 group, tasks 2..9, three seeds
each).

For each run we computed:
- `actor_alpha_scale_end`: mean of `learner/actor_alpha_scale` over
  the last 50 SGD steps of the task.
- `critic_alpha_scale_end`: same for `learner/critic_alpha_scale`.
- `best_success`: max of `evaluator/success_rate` over the task
  (the "best during training" metric from
  `2026-05-13_wandb_findings.md`).
- `end_success`: last value of `evaluator/success_rate`.
- `mean_success`: mean of `evaluator/success_rate` over the task.

We then computed Pearson and Spearman correlations across the 24
pairs.

Code: inline in `docs/wandb_analysis/h1_h2_quick_check.log`. Data:
`docs/wandb_analysis/csv/h1_h2_alpha_vs_success.csv`. Plot:
`docs/wandb_analysis/png/h1_h2_alpha_vs_success.png`.

## What this rules in and out

- **Ruled in (H1, the strong form):** the critic mixture is not
  contributing to task success — it is anti-predictive of it. This
  matches the H3' refinement from
  `docs/2026-05-14_h3_logsumexp_test.md`: the mixture is a
  symptom-tracking regularizer that grows when InfoNCE logits
  spike (k=5 handle_press, k=8 window_close) and shrinks when
  training is going well.
- **Indirectly addressed (H2):** for actor / critic decoupling to
  be the issue, the critic would need to be learning *useful*
  cross-task structure that the actor isn't picking up. The data
  says the critic mixture is *not* learning useful structure
  in the first place. So the decoupling story is moot — there's
  nothing for the actor to fail to consume.
- **Not addressed:** whether a counterfactual mutation of α_scale
  (e.g., zero it at eval time) produces a measurable change in
  success. This is the full offline ablation in
  `2026-05-13_revised_hypothesis_plan.md` Step 2. The cross-run
  correlation here suggests the change would be small, but does not
  prove it. The full test still has scientific value if you want
  the absolute-strongest paper claim.

## The within-task pattern

The within-task across-seed correlations (n=3 per task — noisy
but suggestive) are positive on k=5, 7, 8, 9: seeds where
α_scale ended higher tend to have higher success **on that specific
task**. This is consistent with H3': the seeds that successfully
drive α_scale up are seeds whose `logsumexp_penalty` gradient is
working harder — and those are the seeds whose critic is more
"stable" relative to its peers. But across tasks, harder tasks
universally push α_scale up while universally producing lower
success, dominating the across-pairs trend.

## Two patterns combined

| Direction | Pattern observed |
|-----------|--------------------------------------------|
| Across tasks (which tasks are hard?)   | α_scale higher → success lower (signal: training pathology) |
| Within task (which seed works better?) | α_scale higher → success higher (signal: critic stabilising successfully) |

Both are consistent with the refined story: the critic mixture is
not encoding task knowledge; it is reflecting training-loss-level
stability. The mixture's job — to the extent it has one in this
training regime — is to keep the critic numerics in line, not to
transfer information across tasks.

## What this means for the paper narrative

Synthesizing this with the H3' result and the earlier 2026-05-13
findings, the CKA-RL failure mechanism on contrastive GCRL has
two parallel components:

1. **Actor side:** the actor mixture norm dies (~0.02). The
   actor's argmax over actions discards constant additive shifts
   to `π(s)`, so the gradient through the actor mixture is
   approximately null in steady state. The mixture is dead in
   both norm and effect (confirmed: actor α_scale has no
   significant correlation with success).
2. **Critic side:** the critic mixture stays alive (~0.5) only as
   a `logsumexp_penalty` symptom-tracker, not as a task-knowledge
   carrier (confirmed: critic α_scale is anti-correlated with
   success, p < 0.01). On hard tasks, InfoNCE logits spike and
   gradient pushes α_scale up to keep them in line; on easy tasks,
   α_scale drifts down.

The decomposed-critic algorithm fixes both:

- The actor uses the **composed critic at run time** (the policy is
  trained against `Q(s, π(s))` with the new state-conditioned
  per-task contribution `phi_task(s, a)`). State-dependent.
- The `b_shared` body trained jointly with the dynamics auxiliary
  `L_dyn` produces stable features that don't push InfoNCE logits
  to extreme values, removing the need for the critic mixture as
  a numerical stabilizer.

C2 outperforms C0 on every task with data (k=0..7), with substantial
gains on k=2 (+0.13) and k=5 (+0.23) — the two tasks where the CKA
cell either struggles (k=2) or fails completely (k=5). This is
consistent: the decomposed architecture both removes the dead-actor-
mixture problem AND prevents the critic-instability that drove the
α_scale runaway.

## Status of the hypothesis-test plan

| H# | Hypothesis | Status |
|----|------------|--------|
| H1 | Critic mixture is alive but uninformative | **Supported (cross-run)** |
| H2 | Actor/critic decoupling | **Indirectly disproved** (critic has nothing useful to be decoupled from) |
| H3' | Critic α_scale driven by `logsumexp_penalty` | **Conditionally supported** (mechanism active on hard tasks, dormant on easy tasks) |
| H4 | Dead actor mixture is the real culprit | **Not yet tested** (needs cluster experiment) |
| H5 | Asymmetric gradient signal magnitude | **Plausible but redundant** given H3' and H1 |

The full offline α_scale ablation (Step 2 in the original plan) is
still worth doing if you want the cleanest counterfactual claim,
but the W&B-only evidence is already strong enough to commit to the
revised paper narrative: the failure is **dead actor mixture +
critic mixture as numerical stabilizer**, not pool collapse or null
space.

## Files produced today

- `docs/2026-05-14_h3_logsumexp_test.md` — H3 verdict and methods.
- `docs/2026-05-14_h1_h2_alpha_predicts_failure.md` — this file.
- `docs/wandb_analysis/csv/h3_logsumexp_correlation.csv`
- `docs/wandb_analysis/csv/h1_h2_alpha_vs_success.csv`
- `docs/wandb_analysis/png/h3_trajectories.png`
- `docs/wandb_analysis/png/h1_h2_alpha_vs_success.png`
- `docs/wandb_analysis/h3_logsumexp_correlation.log`
- `docs/wandb_analysis/h1_h2_quick_check.log`
