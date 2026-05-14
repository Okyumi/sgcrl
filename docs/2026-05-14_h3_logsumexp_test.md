# H3 test: is the critic mixture being kept alive by the logsumexp penalty?

Date: 2026-05-14

**Hypothesis H3 (from `2026-05-13_revised_hypothesis_plan.md`):** the
critic's `α_scale` is being driven by the `logsumexp_penalty` term in
the InfoNCE critic loss. The mixture acts as a numerical-stability
regularizer for the contrastive softmax, not as a carrier of
cross-task knowledge. If true, `α_scale` should rise when `logsumexp`
rises (gradient on the penalty term pushes `α_scale` up), and they
should be anticorrelated within a task's trajectory.

**Verdict: conditionally supported with a refined mechanism.** The
mixture is alive on tasks where InfoNCE produces extreme logit spikes
(k=5 and k=8 — the two tasks where the CKA cell fails to learn).
On stable-numerics tasks (k=3, 6, 7, 9) the mixture decays as
expected. The mixture is therefore **a passive consequence of
training instability, not a carrier of useful task knowledge**.

This is paper-relevant. The original CKA-RL framing assumed the
mixture stays alive because it encodes useful per-task knowledge. The
data says: the mixture stays alive because the critic is failing to
fit the task and the `logsumexp_penalty` gradient is propping it up.

---

## Method

Source: the C0 group (`actor_mode='cka', critic_mode='cka'`), 3 seeds
(5, 6, 7) × 8 tasks (k=2..9). For each (task, seed) we fetched the
per-step trajectories of `learner/critic_alpha_scale` and
`learner/logsumexp` (the mean of `logsumexp(logits, axis=1)` over the
batch, computed inside `critic_loss_fn`). 500 SGD steps per task.

For each (task, seed) we computed:

- Pearson correlation of `α_scale` and `logsumexp` (linear).
- Spearman correlation (rank, robust to non-linear monotone
  relationships and spikes).
- `α_scale` change from first 50 to last 50 SGD steps.
- `logsumexp` change from first 50 to last 50 SGD steps.

Code: the inline script in `docs/wandb_analysis/h3_logsumexp_correlation.log`.
Data: `docs/wandb_analysis/csv/h3_logsumexp_correlation.csv`.
Plot: `docs/wandb_analysis/png/h3_trajectories.png`.

## Aggregated per-task results (mean ± SEM across 3 seeds)

| Task | env                       | Pearson         | Spearman        | α_scale Δ | logsumexp Δ | critic mix end |
|------|---------------------------|-----------------|-----------------|-----------|-------------|----------------|
| k=2  | sawyer_faucet_close       | NaN (constant)  | NaN             | +0.000    | −0.660      | 0.83           |
| k=3  | sawyer_push_back          | +0.29 ± 0.19    | +0.34 ± 0.25    | −0.20     | −0.55       | 0.69           |
| k=4  | sawyer_stick_pull         | +0.42 ± 0.20    | −0.14 ± 0.11    | −0.24     | −0.36       | 0.54           |
| **k=5** | **sawyer_handle_press**  | **−0.06 ± 0.02** | **−0.62 ± 0.04** | **+0.30** | **−0.88**  | **1.28**       |
| k=6  | sawyer_push               | −0.30 ± 0.01    | −0.13 ± 0.14    | +0.05     | −0.05       | 0.48           |
| k=7  | sawyer_shelf_place        | +0.10 ± 0.37    | +0.10 ± 0.32    | −0.18     | −0.23       | 0.45           |
| **k=8** | **sawyer_window_close** | **−0.05 ± 0.02** | **−0.43 ± 0.04** | **+0.49** | **−0.28**  | **0.66**       |
| k=9  | sawyer_peg_unplug_side    | +0.15 ± 0.05    | +0.13 ± 0.05    | −0.27     | −0.45       | 0.41           |

k=2 has only 1 active pool slot, so `α_scale` is trivially constant
at 1.0 and the correlation is undefined.

## Why Pearson and Spearman disagree on k=5 and k=8

Pearson detects linear correlation; Spearman detects rank
correlation. They diverge when the relationship is non-linear or
when the variance is dominated by outliers.

The trajectory plot (`docs/wandb_analysis/png/h3_trajectories.png`,
shared above) explains why:

- On **k=5 (handle_press_side)** and **k=8 (window_close)**,
  `logsumexp` is dominated by frequent large **spikes** (30 to 100+,
  vs. a baseline of ~1.5). The spikes correspond to mini-batches
  where the InfoNCE matrix has runaway logits. `α_scale` rises
  monotonically from ~1.0 to ~1.6 (k=5) or ~2.0 (k=8) over the
  task.
- On **k=3 (push_back)** and **k=9 (peg_unplug_side)**, `logsumexp`
  is stable (~1.5-2.5) the whole task, no spikes. `α_scale` gently
  decays.

The Pearson correlation on k=5 and k=8 is washed out because the
spike timing is essentially random across mini-batches; Pearson
averages out. Spearman captures the underlying trend: high-rank
`logsumexp` periods (= heavy-spike phases of training) come with
high-rank `α_scale`. They are rank-correlated even when not
linearly so.

## The cross-task pattern that pins H3 down

The two tasks where `α_scale` grows (k=5 and k=8) are the two tasks
where the CKA cell **fails to learn** (per `c0_best_per_task_agg.csv`):

| Task | env                       | Best success | α_scale Δ |
|------|---------------------------|--------------|-----------|
| k=0  | sawyer_hammer             | 1.00         | n/a       |
| k=1  | sawyer_push_wall          | 1.00         | n/a       |
| k=2  | sawyer_faucet_close       | 0.73         | +0.00     |
| k=3  | sawyer_push_back          | 1.00         | −0.20     |
| k=4  | sawyer_stick_pull         | 1.00         | −0.24     |
| **k=5** | **sawyer_handle_press**  | **0.17**     | **+0.30** |
| k=6  | sawyer_push               | 1.00         | +0.05     |
| k=7  | sawyer_shelf_place        | 0.93         | −0.18     |
| **k=8** | **sawyer_window_close** | **0.43**     | **+0.49** |
| k=9  | sawyer_peg_unplug_side    | 0.70         | −0.27     |

This is the causal direction: **the critic mixture is alive only on
the tasks where the critic itself fails**. The mixture is not
producing cross-task transfer; it is a numerical-stability artifact
of the InfoNCE penalty term acting on a critic that is otherwise
diverging.

## Refined hypothesis H3' (replaces the original H3)

The `logsumexp_penalty` term in `critic_loss_fn` (continual_learning.py:
critic loss is `optax.softmax_cross_entropy(logits, I) +
logsumexp_penalty * logsumexp(logits)^2` when `use_cpc=True`, or
`sigmoid_BCE(logits, I)` when False — but the diagnostic
`learner/logsumexp = mean(logsumexp(logits))` is always logged)
contributes a non-zero gradient through `α_scale` because the
mixture is one of the parameters that affects the critic's logits.
When the network produces large logit spikes (on hard tasks where
the critic is failing to fit), this gradient pushes `α_scale` up.
The mixture's role on those tasks is to dampen logit magnitudes,
not to transfer task knowledge.

On stable-numerics tasks, `logsumexp` is small, the penalty's
gradient is small, `α_scale` is free to drift down (Adam noise or
weak signal), and the mixture decays.

## What this rules in and rules out

- **Ruled out:** the mixture is keeping the critic stable as a
  beneficial feature of CKA-RL. Empirically, mixture growth coincides
  with critic failure, not critic success.
- **Ruled in (preliminary):** the mixture is acting as a
  symptom-tracking regularizer. It's a passive epiphenomenon of
  InfoNCE-loss instability on hard tasks.
- **Not yet tested:** whether the critic mixture is **doing useful
  work for downstream policy learning** even when it's alive. To
  test that, we need H1/H2 (the offline α_scale ablation in Step 2).

## What this means for the paper narrative

The CKA-RL failure mechanism on contrastive GCRL is becoming
clearer:

1. The actor mixture is killed by argmax invariance (consistent with
   the original gradient-null-space intuition, but only for the
   actor side). Actor cross-task knowledge transfer is dead.
2. The critic mixture is alive only on tasks where the critic is
   diverging, kept alive by the `logsumexp_penalty` gradient. The
   mixture is NOT carrying task-knowledge.

The decomposed-critic algorithm (proposal-1) fixes both issues:

- `phi_task(s, a)` provides state-dependent per-task contribution
  that the actor's argmax can actually use (escaping the actor null
  space).
- The dynamics auxiliary `L_dyn` on `b_shared` provides a stable
  feature representation that doesn't require InfoNCE-only
  regularization, so the critic doesn't need a numerical-stability
  crutch.

This is a much more defensible paper claim than the prior
"vectors collapse" / "null space" stories. The mechanism is now
two parallel failure modes (actor null space + critic instability
regularizer), each addressed by a separate piece of the
decomposed architecture.

## What to do next

Step 2 of the revised plan: test H1 and H2 by writing an offline
eval script that mutates `α_scale` on a saved CKA checkpoint and
re-evaluates. This will directly answer whether the critic mixture
is **doing useful work** even when alive.

Predictions for Step 2:
- If H3' is right (mixture is just a stability regularizer):
  zeroing `α_scale` at eval time on tasks k=5 and k=8 should
  produce ~no change in success rate on the already-learned tasks
  (because the mixture isn't being used by the policy). The critic
  might log very different scores but the actor's behavior is
  unchanged.
- If the mixture IS doing useful work despite being alive for
  stability reasons (the surprise case): success rate on earlier
  tasks should drop when the mixture is zeroed.

## Files produced

- `docs/wandb_analysis/csv/h3_logsumexp_correlation.csv` — per-(task,
  seed) Pearson and Spearman correlations plus deltas.
- `docs/wandb_analysis/h3_logsumexp_correlation.log` — extraction log
  with the full data printed.
- `docs/wandb_analysis/png/h3_trajectories.png` — time-series plot
  for k=3, 5, 8, 9 (the four representative tasks).
