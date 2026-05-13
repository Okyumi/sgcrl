# Revised hypothesis-test plan, 2026-05-13

The 3-seed W&B data (see `docs/2026-05-13_wandb_findings.md`) refuted
two prior hypotheses about why CKA-RL fails on the contrastive
goal-conditioned setup:

1. **"Vectors collapse to one direction" (plan §3.1):** refuted —
   actor `mean_offdiag` ~ 0.17, critic ~ 0, both well below 0.9.
2. **"Mixture lives in a gradient null space" (`2026-05-12_cka_failure_results.md`):**
   refuted — actor mixture decays to ~0.02 but critic mixture
   decays only to ~0.5 and critic `α_scale` actively GROWS on
   later tasks. The critic's mixture channel is not in a null space.

The actor / critic asymmetry is the new observation that needs
explaining. This document is a triage of candidate hypotheses,
each with a falsifiable measurement we can take without rerunning
training (W&B data + new offline analysis scripts) or with one
short cluster experiment.

---

## What we know with high confidence

From the W&B trajectories (`docs/wandb_analysis/png/c0_trajectories.png`
and CSVs):

- **Actor:** mixture norm ~0.4 → ~0.02 across all 8 tasks, three
  seeds; `α_scale` decays or fluctuates near 1.
- **Critic:** mixture norm decays from ~12 (task entry, just after
  `v_k` reset) to a non-zero steady state in [0.4, 1.3]; `α_scale`
  grows on later tasks (k=5..8 reach 1.4-2.0).
- **C0 cell eval performance:** `eval/mean_success` across all
  tasks seen reaches ~0.93 on task 0 and falls to ~0.05 by task 9.
  Severe forgetting. Per-task "best during training" reaches 1.0 on
  most tasks but the agent does not retain skills.

So: the algorithm IS failing (forgetting is real), the critic
mixture IS alive (not in a null space), but the actor mixture IS
dead. The mechanism connecting "critic mixture alive" to
"performance collapses" is what we need to identify.

---

## Candidate hypotheses

For each hypothesis I describe the mechanism, what data would
falsify it, and an estimated cost to test (in {analysis-only,
1-h cluster, 1-day cluster, 1-week cluster}).

### H1. The critic mixture is alive but uninformative

**Mechanism:** the critic mixture has non-zero norm but produces a
constant additive bias in feature space that doesn't carry
task-discriminative information. `α_scale` grows because the
gradient says "this bias-term magnitude is useful for the InfoNCE
loss in some shallow way" (e.g., setting the absolute level of
logits to keep `logsumexp` stable). The mixture is not actually
helping cross-task transfer.

**Falsification test:** at eval time, set `α_scale = 0` for the
critic (zero the mixture) and re-evaluate. If success rate is
unchanged, the mixture is uninformative.

**Cost:** one new analysis script that loads a checkpoint, mutates
`α_scale` to zero, runs `evaluate_on_task` for all 10 tasks.
**Analysis-only.** No re-training needed.

### H2. Actor / critic decoupling

**Mechanism:** the critic learns useful cross-task structure in its
mixture, but the actor doesn't get the benefit. The actor loss is
`−Q(s, π(s))`, a single scalar per `s`. The critic's mixture
shapes `Q` in ways that don't change the argmax over `a`, so the
actor's gradient is invariant to the critic mixture. The actor's
own mixture is in argmax's null space and goes to zero. Net result:
the critic carries cross-task info, the policy ignores it.

**Falsification test:** compare two ablations on a trained C0
checkpoint:
1. Eval with the trained actor and the trained critic mixture (the
   baseline).
2. Eval with the trained actor and the critic mixture zeroed.
3. Eval with a per-task actor (the C2 decomposed cell, which we
   already have) and the trained critic mixture from C0.

If (1) and (2) match, H1 holds. If (3) >> (1), the policy
architecture is the bottleneck, not the critic.

**Cost:** **analysis-only.** Builds on the same eval script as H1.

### H3. The mixture is doing something useful for the critic only

**Mechanism:** the critic mixture is a regularizer that keeps
InfoNCE numerics stable on later tasks. Without it, the
`logsumexp_penalty` term would explode and training would diverge.
This explains why `α_scale` grows: the gradient on the
`logsumexp_penalty` term pushes `α_scale` up to keep the score
matrix's logsumexp small.

**Falsification test:** check `logsumexp` history across tasks in
the C0 W&B data. If `logsumexp` grows as `α_scale` shrinks, and
shrinks as `α_scale` grows, this hypothesis is consistent. If they
are uncorrelated, H3 is unlikely.

**Cost:** **analysis-only.** Pull
`learner/logsumexp` from W&B alongside `learner/critic_alpha_scale`
for tasks k=2..9 and compute the correlation.

### H4. The actor mixture being dead is the real culprit

**Mechanism:** flip the framing. The critic mixture is fine; the
problem is that the actor policy can't use cross-task knowledge
because its mixture is dead. The proposal-1 fix (decomposed
critic) might be missing a symmetric fix on the actor side.

**Falsification test:** if true, fixing the actor (e.g.,
state-conditioned actor mixture) should improve performance
*without* changing the critic. Test by running CKA actor +
decomposed critic, or by adding a state-conditioned per-task
actor head while keeping the regular CKA critic.

**Cost:** **1-day cluster.** Requires a new ablation cell with a
state-conditioned actor mixture. Worth doing only if H1-H3 turn
out to support this story.

### H5. The asymmetry is just because the critic loss has more
gradient than the actor loss

**Mechanism:** the critic sees `B²` pairs per InfoNCE batch
(`batch_size = 256` → 65k pairs). The actor sees `B` actions per
batch. The critic's gradient signal is much louder. The critic
mixture stays alive because there's enough signal to keep it
trained; the actor mixture dies because there isn't.

**Falsification test:** Two ways:
1. **Re-run C0 with larger actor batch.** Cost: 1-week cluster.
2. **Look at gradient norms in W&B.** Pull
   `learner/grad_norm_actor` and `learner/grad_norm_critic` (if
   they exist) and check the ratio.

If the gradient ratio is enormous (say 100x), H5 is supported. If
similar magnitude, ruled out.

**Cost:** **analysis-only first** (gradient norms), then
**1-week cluster** if the analysis supports it.

---

## Priority for next 24-48 hours

In order of (low cost, high information):

### Step 1 (today, ~1 hour of W&B + analysis): test H3 and H5(a)

Run the gradient-correlation / logsumexp analysis. This needs no
new training:

```python
# Pull learner/logsumexp and learner/critic_alpha_scale per task; compute correlation
# Pull learner/grad_norm_* if present
# Write docs/2026-05-13_h3_h5a_check.md with verdicts
```

Expected outcome: either H3 holds (logsumexp is tracking α_scale)
or it doesn't. Either way we narrow the search.

### Step 2 (next 2 days): test H1 and H2

Write a new offline-eval script that:

1. Loads a C0 checkpoint at task 9.
2. Mutates `α_scale` (set to 0, 1, 2, ...) and re-evaluates on
   all 10 tasks.
3. Mutates `α_logits` (uniform, one-hot on each pool entry, zero
   vector) and re-evaluates.
4. Optionally swaps the actor between a C0 checkpoint and a C2
   checkpoint.

This is the most direct test of "is the mixture actually being
used?" and "is the actor / critic decoupling the issue?".

Cost: one ~200-line script that reuses `evaluate_on_task` from
`run_continual_contrastive.py`. Estimated 1-2 days to write,
test, and produce a small table of results. **No re-training.**

### Step 3 (deferred, conditional on Step 2): H4 fix

If Step 2 establishes that the actor side is the bottleneck (H2
or H4), write a state-conditioned actor mixture. This is a paper
contribution in itself if it works.

If Step 2 establishes the critic mixture is uninformative (H1),
the paper narrative shifts to "the mixture isn't doing anything
useful for either component" and the proposal-1 decomposed-critic
remains the right fix.

---

## What this means for the paper

The decomposed-critic algorithm (proposal-1) is **competitive with
the CKA baseline on best-during-training success**, beating it on
k=2 and k=5 (the harder tasks). See `docs/2026-05-13_wandb_findings.md`
§2 for the comparison table. So the paper still has a working
algorithm and a meaningful improvement to report.

What's missing is the **mechanism story** for *why* the decomposed
critic helps. The original framing ("CKA fails because vectors
collapse / mixture lives in null space") is gone. A new framing
must come from Steps 1-2 above.

The cleanest paper structure given current evidence:

1. Show that CKA-RL fails on contrastive GCRL (success rate
   plummets after task 2).
2. Show that the C2 decomposed-critic recovers most of the loss.
3. Identify the mechanism through the ablation in Step 2 above and
   present it as the third pillar of the analysis section.

Until Step 2 lands, the paper has only #1 and #2. That's still a
publishable contribution but the analysis section needs the third
pillar to be strong.

---

## Files referenced

- `docs/2026-05-13_wandb_findings.md` — quantitative tables.
- `docs/wandb_analysis/` — CSVs and trajectory plot.
- `docs/2026-05-08_plan_proposal1_dyn_aux.md` — original plan with
  REVISED 2026-05-12 and FURTHER REVISED 2026-05-13 callouts.
- `docs/2026-05-12_cka_failure_results.md` — superseded by today's
  findings; kept for record.
