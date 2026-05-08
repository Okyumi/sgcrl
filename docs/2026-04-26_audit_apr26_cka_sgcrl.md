# SGCRL CKA Audit — Apr 26, 2026

## Symptoms

The user reports:

1. `actor_mode=cka` shows **no improvement** over `reset` and
   `persistent`, with hints of **performance degradation**.
2. `actor_mode=cka` runs **noticeably slower** per step than the
   non-CKA modes.

This audit cross-checks the SGCRL contrastive port
(`contrastive/continual_learning.py`, `contrastive/knowledge_pool.py`,
`run_continual_contrastive.py`) against the original CKA-RL reference
(`cka-rl-meta-world/models/{cka_rl,fuse_module}.py`), which both live
in this repo on `section3_done`.

The SGCRL code is in much better shape than the BuilderBench port: it
has learnable `beta_k` and learnable `alpha_scale` with their own
optimisers (continual_learning.py:56–61, 744–807) and a separate
outer-loop gradient step. So it does not have BuilderBench's central
"alpha is non-existent" bug. It has, however, four other bugs that
together explain both symptoms.

---

## Bug 1 — `beta_k` gradient is computed against a stale `pool_contribution` (correctness, primary)

The inner JIT loop in `_make_update_step.update_step` receives
`pool_contribution` as a *pre-computed Python pytree* (line 419, 421:
`transitions, pool_contribution = data`) and uses it to form
`combined_policy = base + pool_contribution + v_k`. This pool
contribution is computed by `_compute_pool_contribution` (lines
704–719) using the **current** `beta_k`.

The actor gradient w.r.t. `combined_policy` is then propagated to
`v_k` only (lines 458–477). `beta_k` and `alpha_scale` are explicitly
**passed through** in the inner loop:

```python
beta_k_new = state.beta_k        # line 484
beta_opt_state = state.beta_k_optimizer_state
alpha_scale_new = state.alpha_scale
alpha_scale_opt_state = state.alpha_scale_optimizer_state
```

Their actual update happens once **per `step()` call**, in the
non-JITed `_update_beta_and_alpha_scale` (lines 744–807). That outer
step does compute a real gradient through the pool blend, but it does
so from a **fresh forward pass** that re-builds `pool_contribution`,
re-composes the policy, and runs `q_network.apply(...)` on a single
fresh batch — completely independent of the actor / critic gradient
that the inner JIT loop just took.

In other words, the architecture is:

  - Inner JIT loop: takes `num_sgd_steps_per_step` SGD steps that
    update `(v_k, q_params, target_q_params, alpha_params)` against
    a fixed `pool_contribution`.
  - After all those steps, **one** non-JIT gradient step on
    `(beta_k, alpha_scale)` against the **last** mini-batch of the
    learner step.

This causes three problems:

  (a) **Imbalanced learning rates.** The inner loop runs
      `num_sgd_steps_per_step` (default 64) updates per learner step
      on `(v_k, q, ...)`, while `(beta_k, alpha_scale)` get exactly
      one. The attention weights move 64× more slowly than `v_k`,
      and `v_k` is fitted to the *current* `beta_k` value rather
      than co-evolving with it.

  (b) **Stale gradient.** Even that one outer step is computed
      against `self._state` *as it stood after the inner loop
      finished* (line 789–790), so the snapshot of
      `(v_k, q_params)` it sees is itself one step behind the
      `(beta_k, alpha_scale)` it is about to update. The
      gradient is biased, not catastrophically wrong but
      systematically off.

  (c) **No JIT.** `_update_beta_and_alpha_scale` is plain Python with
      `jax.grad` called inside it; every learner step re-traces and
      re-runs a small graph (network apply + softmax + reduce). This
      adds up at high learner-step counts.

The original CKA-RL puts `alpha` and `alpha_scale` *inside the
forward pass of every gradient step*: `FuseLinear.forward` computes
`alphas = softmax(alpha * alpha_scale)` and uses it to build the
weight on the spot (`fuse_module.py:97–110`), so `(alpha,
alpha_scale)` co-evolve with the rest of the policy at the same
update cadence. Our outer-loop split is a workaround for the variable
pool length, not a faithful port.

This is the central correctness bug: `alpha` is *learnable* in name,
but in practice gets ~64× less optimiser pressure than `v_k`, with a
stale gradient, computed by a non-JITed loop.

## Bug 2 — Pool contribution recomputed on host every learner step (slowdown, primary)

`step()` calls `_compute_pool_contribution()` (line 839 → 704–719)
**before** every JIT call. That helper is a Python `for` loop that
walks every entry in the pool and accumulates a weighted sum on the
host:

```python
contribution = _pytree_zeros_like(pool_vecs[0])
for j, v_j in enumerate(pool_vecs):
    contribution = jax.tree_map(
        lambda c, v: c + float(alpha[j]) * v,
        contribution, v_j)
```

`float(alpha[j])` forces a **host sync** on every iteration; with a
pool of 5 vectors that is 5 host syncs per learner step. Then the
result is a Python pytree that gets fed back into the JIT trace as a
new input every step, which is fine for correctness but does mean the
contribution does not benefit from JIT either.

This is the primary cause of the per-step slowdown. The combined
effect of (a) Bug 1's non-JIT outer loop and (b) Bug 2's per-step host
sync over a 5-vector pool is enough to be visibly slower than `reset`
or `persistent`, which simply do `combined = q_params` and skip the
whole machinery.

## Bug 3 — Critic CKA uses uniform pool blending (correctness, secondary)

`_compute_critic_pool_contribution` (lines 723–740) hard-codes
`pool_c = mean(w_j)` regardless of any learnable parameter:

```python
n = len(critic_vecs)
result = _pytree_zeros_like(critic_vecs[0])
for w_j in critic_vecs:
    result = jax.tree_map(lambda r, w: r + w / n, result, w_j)
return result
```

The accompanying comment admits this:

```text
For the critic CKA, we use uniform blending (no learnable β for
critic) to keep the implementation simple: pool_c = mean(w_j) if
pool non-empty.
```

For `critic_mode='cka'`, this means the critic at task `k` is
initialised at `q_base + mean(w_j)` and the inner loop fits `w_k` on
top of that. With every prior task contributing equally, including
ones that are unrelated to the current task, the critic spends much
of the early task budget unlearning unhelpful past contributions. In
the 9-cell ablation this manifests as `critic_mode=cka` (under any
actor mode) being noticeably worse than `critic_mode=persistent` at
the start of each new task.

The original CKA-RL has a learnable `alpha` for **every** `FuseLinear`
layer. There is no "critic exception" in the reference algorithm.

## Bug 4 — Pool seeded with a zero vector at task 0 (correctness, secondary)

`run_continual_contrastive.py:807–808`:

```python
# Per pseudocode, initialise the pool with a zero vector (not v_0).
pool.append(_pytree_zeros_like(out_theta_base))
```

This appends an all-zero knowledge vector at the end of task 0. At
task 1 the pool has one entry (the zero), `softmax(beta) = 1.0`, and
`pool_contribution = 0`. Harmless. At task 2 the pool has two
entries: the zero placeholder and the trained `v_1`. Even with a
learnable `beta`, the softmax has to push the placeholder slot's
weight towards zero, which is wasted optimiser effort — and worse,
the placeholder is a fixed point for the merge rule (it is most
similar to itself, but cosine similarity with another zero-direction
vector is undefined; with `v_1` it is `0/0`, which our merge code
guards as `0.0`). So the zero placeholder will never be merged out,
even when the pool overflows. It permanently occupies a slot that
could hold a real prior task.

The same bug exists for the critic pool (line 860).

In the **inner** loop, `pool_contribution` is reset to
`_pytree_zeros_like(...)` if the pool is empty (line 707–708). Once
we drop the zero seed (Fix C), there is one branch in
`compute_pool_contribution` that already does the right thing for an
empty pool. The seed is purely vestigial.

## Bug 5 (minor) — JAX 0.6 deprecation: `jax.tree_map`

`contrastive/knowledge_pool.py` and `contrastive/continual_learning.py`
use `jax.tree_map` in 14 places. JAX 0.4.25+ deprecated this and JAX
0.6.0 removed it. The same fix applied to BuilderBench last week
applies here: `jax.tree_map` → `jax.tree.map`. This is a portability
issue rather than a correctness bug; it has not yet bitten because
the runtime jax on Torch HPC is presumably < 0.6.

## Bug 6 (minor) — Head-tagging by string match

Both the inner `_mask_leaf` (line 466) and the orchestrator's
`_split_head_body` (line 813) decide whether a parameter leaf is
"head" by string-matching `'Normal'` against the path. This works
today because Haiku's actor uses a `NormalTanhDistribution` head, but
any future module rename silently breaks the head/body classification
without raising an error. Lifting the head identification into a
declared list (or a tag stored next to the actor module) makes it
robust.

## Bug 7 (cosmetic) — `_update_beta_and_alpha_scale` uses a deterministic key

```python
key = jax.random.PRNGKey(0)  # deterministic for gradient
```

Using the same RNG every learner step means the `(beta_k,
alpha_scale)` gradient is computed against an always-identical
sample of the actor's stochastic action. Probably negligible at
learner-step granularity, but worth noting.

---

## What this means for the symptoms

**No improvement / degradation** maps cleanly onto Bug 1 + Bug 3 +
Bug 4: `beta_k` is too sluggish to find a useful attention pattern,
the critic pool is uniform-averaged with no mechanism to fix it,
and the placeholder zero vector permanently dilutes the pool. Under
these three the agent is paying every cost of CKA's machinery while
getting almost none of its benefit.

**Slowness** maps onto Bug 2 + Bug 1c: per-step host syncs on
`alpha[j]` plus a non-JIT outer-loop gradient step on `(beta_k,
alpha_scale)` add measurable Python overhead that `reset` and
`persistent` do not pay.

---

## Recommended fixes (priority order, A → F)

### Fix A — fold β_k and α_scale into the inner JIT loop (correctness, primary).

This is the central fix and the one with the biggest expected
correctness impact. The cleanest path is to fix the pool length at
`K_max + 1` (the same trick used in the BuilderBench refactor): pool
becomes a stacked pytree with leading axis `capacity` plus a boolean
mask, and `pool_contribution` becomes a single `compute_contribution(
pool, beta_k, alpha_scale)` that runs inside JIT and is
differentiable in `beta_k` and `alpha_scale`. Inactive slots are
masked to `-∞` before the softmax.

Once that is done, `(beta_k, alpha_scale)` join `v_k` as
optimiser-tracked parameters in the inner loop. `_update_beta_and_alpha_scale`
goes away entirely.

### Fix B — make the critic CKA learnable too.

Symmetric to Fix A: critic also gets `beta_k_critic` and
`alpha_scale_critic`, computed from the same masked-softmax helper.
The "uniform blending for simplicity" comment in
`_compute_critic_pool_contribution` is removed.

### Fix C — drop the zero-vector pool seed.

Remove the `pool.append(_pytree_zeros_like(out_theta_base))` lines in
`run_continual_contrastive.py` for both actor (line 808) and critic
(line 860). With a properly learnable α, the placeholder is
unnecessary, and the merge rule (Bug 4) no longer has to fight an
undefined cosine similarity.

### Fix D — vectorise the pool merge.

The current `_merge_most_similar_pair` (knowledge_pool.py:106–134)
runs O(n²) cosine similarities in a pure Python loop with `float(...)`
host syncs. Replace with a single matmul over a `[k_max+1, D]`
flattened pool tensor, mask the diagonal, take the argmax. Runs only
at task boundaries, so the savings are small in absolute terms, but
removing the host syncs makes the code simpler and JIT-clean.

### Fix E — tag actor heads explicitly.

Stop string-matching `'Normal'` in `_mask_leaf` and
`_split_head_body`. The actor's NormalTanhDistribution head should
expose a list of "head leaf paths" that the orchestrator and the
inner loop both consume. The matching SGCRL module
`contrastive/networks.py` is the natural place to expose this.

### Fix F — JAX 0.6 portability.

Replace every `jax.tree_map` with `jax.tree.map` in
`contrastive/knowledge_pool.py` and
`contrastive/continual_learning.py`. Mechanical replacement; no
semantic change.

---

## Acceptance criteria after Fix A + B

1. `actor_mode=cka` matches or exceeds `actor_mode=persistent` on
   the 9-cell ablation by the end of task 1, averaged over seeds.
   At task 0 the modes are identical by construction.
2. Per-learner-step wall-clock of `cka` is within ~10% of
   `persistent`. The remaining overhead is the cost of the larger
   forward pass with the masked-softmax blend, which is small.
3. Logged `alpha_weights` and `alpha_scale` show non-uniform values
   that drift over training. A flat curve at uniform-1/n would mean
   the inner-loop gradient on `beta_k` is not flowing.

---

## Pending coordination with BuilderBench

The BuilderBench port already has a clean Fix A + B refactor (commit
`efe63ba`). Once we apply the same shape of refactor to SGCRL the two
codebases will share the same algorithmic skeleton, which makes the
appendix-table comparison and any cross-codebase runs apples-to-apples.

The negative-bank variant remains deferred until SGCRL CKA passes the
acceptance criteria above.
