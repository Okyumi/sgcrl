# CKA-failure diagnostic — observed results and revised hypothesis

Date: 2026-05-12

## Observed results (C0 cell, `actor_mode='cka', critic_mode='cka'`, single seed)

| Task | actor `mean_offdiag` | actor `max_offdiag` | critic `mean_offdiag` | critic `max_offdiag` | actor `mixture_norm` (start → end) | critic `mixture_norm` (start → end) |
|------|----------------------|---------------------|------------------------|----------------------|------------------------------------|--------------------------------------|
| 2    | 0.1807               | 0.1807              | −0.1866                | −0.1866              | 0.30 → ≈ 0                          | 12 → ≈ 0                              |
| 3    | 0.0749               | 0.3427              | −0.1030                | −0.0552              | 0.17 → < 0.05                       | 12 → ≈ 0                              |
| 5    | 0.1683               | 0.3769              | −0.0408                | +0.0139              | 0.22 → < 0.05                       | 8 → ≈ 0                               |
| 7    | 0.1884               | 0.3830              | −0.0130                | +0.0249              | 0.25 → ≈ 0                          | 8 → ≈ 0                               |
| 9    | 0.1662               | 0.3830              | −0.0012                | +0.0456              | 9.27 → ≈ 0                          | 7 → ≈ 0                               |

(Numbers reported by the user, 2026-05-12. `[pool]` line confirms 4 head params + 24 body params per pool entry, all tasks.)

## What the data says (vs the original audit hypothesis from plan §3.1 / §9)

The audit hypothesis predicted **two** signatures of failure:

1. `pool_cos_mean_offdiag > 0.9` (pool vectors nearly parallel — collapsed to a degenerate 1-D subspace).
2. `mixture_norm < 0.1` sustained (mixture term negligible relative to `v_k`).

**Half of it is confirmed, half refuted.**

### Confirmed: `mixture_norm → 0`

On both actor and critic, across all tasks `k ∈ {2, 3, 5, 7, 9}`, the mixture norm starts at a non-trivial value and decays to near zero by end of task. The critic case is dramatic: `||mixture||` starts at 12× `||v_k||` (because `v_k` is reset to zero at task boundary and the pool from previous tasks dominates initially) and is driven down to essentially 0 by 8 M steps. The actor follows the same pattern at smaller magnitude. Conclusion: **CKA degrades to per-task residual fine-tuning on `v_k`**, exactly as plan §3.2 predicted.

### Refuted: vectors-are-parallel

The actor pool has mean off-diagonal cosine ≈ 0.15–0.20 (max ≈ 0.38) across tasks 5/7/9. The critic pool has mean off-diagonal cosine ≈ 0 — slightly negative early (tasks 2/3 at −0.19/−0.10), drifting to roughly zero as more tasks accumulate. These are far below the 0.9 collapse threshold.

The per-task knowledge vectors **span useful directions**. They are not degenerate. The cosine-similarity story in plan §3.1 was wrong.

## The revised question

Given that `{v_j}` are reasonably diverse, **why does `mixture_norm → 0`?** And why does CKA-RL work for SAC (the canonical setup) but not here?

## Revised hypothesis: contrastive losses have a state-independent additive null space; the mixture lives in it

### The composition rule (recap)

In CKA-RL, the actor / critic at task `k` is parameterised as
\[
\theta'_k = \theta_{\mathrm{base}} + \sum_{j=1}^{k-1} \alpha_j(\alpha_\text{scale}, \alpha_\text{logits}) \, v_j + v_k
\]
where `θ_base` is the post-base-task body, `v_j` are frozen
per-task delta pytrees from the pool, `α_j` are softmax-normalised
weights with a learnable scalar temperature `α_scale`, and `v_k` is
the gradient-trainable per-task delta for the current task. **All
three groups (`v_k`, `α_logits`, `α_scale`) receive gradients in the
inner loop.** `v_k` is reset to zero at every task boundary.

### Why SAC's loss flows gradient through every channel

SAC's actor loss is roughly `E_s[Q(s, π(s)) − α · H(π(·|s))]`. The
target `Q` is state-dependent and shaped by a TD bootstrap with its
own per-step gradient. Any perturbation to the policy parameters —
including a constant additive direction like one of the `v_j`s —
changes `π(s)` for every `s`, which changes `Q(s, π(s))` in
state-dependent ways. The gradient through the mixture is non-null
and proportional to "how much the mixture's direction reduces the
expected Q-loss". Pool entries from previous tasks therefore retain
real gradient signal in the steady state.

### Why contrastive GCRL is different

In contrastive GCRL the actor loss reduces to `−Q(s, π(s))` where
`Q(s, a) = φ(s, a)ᵀ ψ(g)` and the critic loss is InfoNCE on the same
inner product (softmax over a batch with the diagonal as the positive
label).

Both of these losses are approximately **shift-invariant** in the
mixture's contribution:

1. **Critic InfoNCE.** The softmax over a (B, B) score matrix is
   strictly invariant under adding the same constant to every row of
   `φ`. The mixture term `Σ α_j v_j` is a **state-independent** linear
   combination of frozen vectors — when added to the network's weights,
   its effect on `φ(s, a)` is, to first order in the perturbation, a
   feature-space shift that is approximately constant across the
   batch. The softmax discards this shift exactly. So the critic loss
   has near-zero gradient through `{α, v_j}`.
2. **Actor.** The objective is `−Q(s, π(s)) = −φ(s, π(s))ᵀ ψ(g)`.
   `argmax_a Q(s, a)` is invariant under adding a constant to `π`'s
   output. The mixture's effect on `π(s)` is, again, approximately a
   constant shift (state-independent perturbation to a non-linear
   network produces a near-constant output shift to first order). The
   actor argmax discards this — gradient through the mixture is small.

In both cases, `v_k` escapes the null space because it sees the loss
through **all** of its parameters, including weight matrices of
intermediate layers, which produce **state-dependent** changes in the
network output. The mixture, being a fixed linear combination of
fixed vectors with no input-dependent gating, cannot produce
state-dependent output changes — only state-independent ones.

So the gradient flow is split:
- `v_k` channel: full state-dependent gradient. Captures all the
  task-specific learning.
- mixture channel: gradient lives entirely in the contrastive null
  space. Optimisation drifts `α_scale → 0` (Adam noise dominates
  approximately-zero gradients) and the mixture collapses.

### Why this predicts the observed asymmetry

- **Critic vectors more orthogonal than actor vectors** (observed:
  critic `mean_offdiag ≈ 0`, actor `mean_offdiag ≈ 0.17`). The critic
  loss has **stricter** shift-invariance (the softmax is mathematically
  exact, not first-order). So gradient signal through the critic
  mixture is closer to zero. With no signal to curate them, the
  critic `v_j`s are essentially random — and random high-dimensional
  vectors are nearly orthogonal in expectation.
- **Critic mixture-norm collapses faster than actor's** (observed:
  critic from 12 to ~0 within tasks; actor from ~0.25 to <0.05 with
  occasional spikes). Same reason: less gradient signal through the
  critic mixture means it has less stabilising structure and drifts
  to zero faster.
- **Actor `v_j`s have some weak alignment** (observed: actor
  `mean_offdiag = 0.18` stable across tasks, max climbing 0.18 →
  0.38). The actor's loss has first-order shift-invariance but not
  exact — there's a small residual gradient that gives `v_j`s a faint
  but real signal. Over many tasks, the actor `v_j`s acquire a small
  shared component (the residual gradient direction).

## What this means for the paper

### Negative-result figure (CKA-RL fails on contrastive GCRL)

The figure should now show **two curves over training**:

1. Top panel: per-step `cka/critic_mixture_norm` and `cka/actor_mixture_norm` over the 8M-step task, one curve per task. Both start at modest-to-large values and decay to zero. Critic starts at ~12, decays in 2-3M steps. Actor starts at ~0.25 (modal) or 9.27 (task 9 spike), decays similarly.
2. Bottom panel: pool cosine `mean_offdiag` over tasks. Both stay well below 0.9; actor ≈ 0.17 stable, critic ≈ 0 with mild drift.

Caption can read: "The per-task knowledge vectors span useful
directions (cosine similarity is not collapsing), yet the mixture
term `Σ α_j v_j` is driven to zero by optimisation. Gradient flow
through the mixture is approximately null because contrastive losses
are shift-invariant in state-independent additive perturbations."

### Mechanism (paper section 3)

The framing is **stronger** than the original "vectors collapse"
claim. State-independent additive perturbations live in the gradient
null space of contrastive losses. This is a property of the **loss**,
not of the **pool**. The pool could be perfect and the mixture would
still die.

### Motivation for the decomposed critic (paper section 6)

The decomposed critic replaces `Σ α_j v_j` with `phi_task(s, a)`. The
key word is `(s, a)`: `phi_task` is a network, evaluated on the
current `(s, a)`, so its contribution is **state-dependent**. It
therefore escapes the null space and gets real gradient signal. The
auxiliary dynamics loss `L_dyn` on `b_shared` further ensures the
shared body cannot absorb task-specific information.

This is a tighter motivation than the original. The decomposed
critic isn't "a more diverse pool", it's "a pool whose elements
project onto the gradient-active subspace by construction".

## Falsifiable predictions

If this explanation is right:

1. `alpha_scale` should drift toward zero, not to a non-zero
   steady state. **Plot `alpha_scale` over training for C0** to
   verify. (We log it already.)
2. The collapse should be faster on the critic than the actor.
   **Already observed.**
3. Replacing the mixture with a state-conditioned variant
   `Σ α_j(s) v_j` (i.e., gating `α` on `(s, a)` through a tiny
   per-input network) should restore non-zero `mixture_norm`. This
   is a **cheap ablation** that would close the argument
   experimentally.
4. The same CKA-RL formulation on a non-contrastive GCRL algorithm
   (e.g., HER-DDPG with a regression critic) should **not** show
   mixture-norm collapse, because the regression loss is not
   shift-invariant. Could be cited from the SAC results in the
   original CKA-RL paper; doesn't need a new run.
5. The same CKA-RL formulation on a non-goal-conditioned contrastive
   setup (e.g., DrQ + CURL on continual Meta-World) — prediction is
   ambiguous and not a natural ablation here.

## What we should run next

- **One more seed** (seed 6 or 7) on the same C0 config, to confirm
  the actor `mean_offdiag ≈ 0.17` is reproducible and not seed
  noise.
- **Log and plot `alpha_scale`** for the existing C0 run. The
  hypothesis predicts it decays toward zero; if it stays near 1.0
  but the mixture still collapses, the story is wrong and we need
  to look at `alpha_logits` distribution instead.
- **Add prediction 3 as an optional ablation** (state-conditioned
  mixture). One commit-worth of work to wire a per-input MLP that
  emits `α_logits(s, a)` instead of free parameters. Strong paper
  evidence if it lands; deferrable if time-constrained.

## Sources internally referenced

- Plan: `docs/2026-05-08_plan_proposal1_dyn_aux.md` (§3.1, §3.2, §9 — to be revised below).
- Mixture norm helper: `contrastive/knowledge_pool.py:mixture_to_vk_ratio` (D5 commit).
- Pool cosine: `contrastive/knowledge_pool.py:cosine_summary_from_vectors` (D1–D4).
- Composition rule in code: `contrastive/continual_learning.py` lines 5–66
  (header comment) and the `theta` reconstruction inside `update_step`.
