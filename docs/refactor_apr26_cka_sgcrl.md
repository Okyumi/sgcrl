# SGCRL CKA Refactor — Apr 26, 2026

Companion to `docs/audit_apr26_cka_sgcrl.md`. The audit names seven bugs (1–7)
and six fixes (A–F). This doc records what changed on disk to apply Fixes A–F
to the SGCRL contrastive learner. Code lives on `section3_done`; the docs in
this file are pushed only to `section3_done`.

## Files modified

- `contrastive/knowledge_pool.py` — full rewrite. Replaces the variable-length
  `KnowledgePool` (a Python `list[pytree]`) with a fixed-shape, JIT-friendly
  `CKAPool` and `CKAState`. Old `KnowledgePool` retained as a transitional
  shim used by the orchestrator's checkpointing path.
- `contrastive/continual_learning.py` — restructured around the new CKA API.
- `contrastive/networks.py` — centralised the actor-head path tag in
  `ACTOR_HEAD_PATH_TAGS` / `is_actor_head_path`.
- `run_continual_contrastive.py` — passes `actor_mode` through to the learner;
  drops zero-vector pool seeding at task 0; uses the centralised head tag.

## What each fix did

### Fix A — single-optimiser path for `(v_k, alpha_logits, alpha_scale)` (Bug 1)

The old learner had two separate update cadences: a JIT inner loop that
ran `num_sgd_steps_per_step` (~64) updates on `v_k` per call, and a
host-side `_update_beta_and_alpha_scale` that ran exactly one update per
learner step against an independently re-derived forward pass. Result:
~64x cadence imbalance and `beta_k` gradients computed from stale
forward passes.

The new path differentiates the actor loss directly through
`compose_from_trainable(cka_state, {'v_k', 'alpha_logits', 'alpha_scale'})`
inside the inner JIT loop. A single optax optimiser
(`vk_optimizer.init(bundle)`) carries opt state for all three trainables
together. Same scheme on the critic side via `q_optimizer`.

Concretely (`continual_learning.py`):

- New fields on `ContinualTrainingState`:
  `actor_cka_state`, `actor_cka_opt_state`, `critic_cka_state`,
  `critic_cka_opt_state` (all `Optional[...] = None`).
- `__init__` now branches on `self._actor_cka_path` /
  `self._critic_cka_path`. When active, builds the new CKA state via
  `init_cka_state(...)` then `cka_reinit_for_new_task(...)`, and primes
  the optimiser with the trainable bundle.
- `update_step` (the JIT body) builds `combined_policy` from the
  trainable bundle in CKA mode (line ~469) and differentiates the actor
  loss against the bundle (line ~566). The critic side is symmetric.
- `_compute_pool_contribution`, `_compute_critic_pool_contribution`, and
  `_update_beta_and_alpha_scale` are deleted. `step()` no longer calls
  them and no longer threads `pool_c` through `_update_step`.

### Fix B — vectorised pool contribution (Bug 2)

The old `_compute_pool_contribution` ran a Python `for j, v_j` loop with
a `float(alpha[j])` host sync per pool slot per learner step. The new
`compute_contribution` (in `knowledge_pool.py`) stacks the pool tensor
once (`stack[j]` along leading axis) and uses a single broadcasted
multiply-accumulate. The pool is fixed-capacity `(k_max + 1, *leaf)`;
inactive slots are masked out via boolean `pool.mask` and a
`-jnp.inf` fill in the softmax pre-image. No host syncs, no Python
loops, fully JIT-fusable.

### Fix C — drop zero-vector pool seed (Bug 4)

`run_continual_contrastive.py` previously seeded both the actor pool
and the critic pool with `_pytree_zeros_like(theta_base)` at the end of
task 0. That placeholder permanently diluted contributions: with `n`
real entries the softmax distribution was over `n + 1` entries, the
extra slot being zero-valued (so contributing nothing to composition
but taking ~`1/(n+1)` of the alpha mass).

The new CKA pool starts genuinely empty (`mask = [False, ...]`) and
the masked softmax computes correct contributions when no slots are
active (`compute_contribution` returns a zeros pytree, which is then
added to `theta_base` to recover identity). The runner now writes:

```python
if task_id == 0:
    out_theta_base = jax.tree.map(lambda b, v: b + v, learner.theta_base, v_k)
    # No pool.append here: leave the pool empty.
```

### Fix D — vectorised merge

`_merge_most_similar_pair_host` in `knowledge_pool.py` does the cosine
similarity computation on the active slots in one matrix product:
`(flat @ flat.T) / (norms[:, None] * norms[None, :])`. Diagonal +
lower-triangle is masked with `-inf` and `argmax` selects the pair.
Replaces the old O(n²) Python loop with `float(sim)` host syncs.

### Fix E — centralised head tagging (Bug 6)

The CKA path masks gradients on the body (encoder) of `v_k` so only
the actor head receives updates. The old code spelled out
`'Normal' in path_str` in three places. We added
`ACTOR_HEAD_PATH_TAGS` and `is_actor_head_path` in
`contrastive/networks.py` and refactored:

- `continual_learning.py`: 2 call sites in the gradient masker (CKA
  and non-CKA branches both use it).
- `run_continual_contrastive.py`: 1 call site in
  `_split_head_body`.

The submodule name `'Normal'` is unchanged so existing checkpoints
still match. To add a new head module type, register a unique
substring of its Haiku name in `ACTOR_HEAD_PATH_TAGS`.

### Fix F — portable tree map (Bug 5)

The original code used the top-level alias `jax.tree_map`, which was
removed in JAX 0.6. The first pass of this refactor replaced it with
`jax.tree.map`, but that namespace only appeared in JAX 0.4.25; the
actual training environment runs **JAX 0.4.10**, where neither alias
exists in the form we used. The portable answer is
`jax.tree_util.tree_map`, which has been the stable public API since
JAX 0.2 and is still present in JAX 0.6+.

Replaced all 32 call sites (across `knowledge_pool.py`,
`continual_learning.py`, and `run_continual_contrastive.py`) with
`jax.tree_util.tree_map`. The crash that surfaced this:

```
File ".../contrastive/knowledge_pool.py", line 289,
  in _pytree_zeros_like
    return jax.tree.map(jnp.zeros_like, tree)
AttributeError: module 'jax' has no attribute 'tree'
```

## Verification

Smoke test on the new `knowledge_pool.py` (Haiku-shape pytree;
`capacity=6` corresponding to `k_max=5+1`):

```
Initial pool mask: [False False False False False False]
After 2 host-side appends: [True True False False False False] sum: 2
After reinit_for_new_task, alpha_logits: [0.0162 0.0203 0 0 0 0 ]
Optimiser init over {'v_k', 'alpha_logits', 'alpha_scale'}: OK
compose_from_trainable matches input pytree structure: True
JIT compile of compose_from_trainable: OK
```

Gradients flow through the trainable bundle:

```
|grad alpha_logits|: 0.81
|grad alpha_scale|: 0.0023
|grad v_k.mlp.w|: 0.57
```

Inactive pool slots receive zero gradient through `alpha_logits` (the
masked-softmax kills their contribution), so the trainable bundle's
shape is fixed and JIT compile is stable across tasks (capacity is
fixed at construction).

## Acceptance

The audit named three acceptance criteria for Fixes A + B alone:

1. `actor_mode=cka` ≥ `actor_mode=persistent` on the 9-cell ablation
   by end of task 1.
2. Per-step wall-clock for `cka` within ~10% of `persistent` (was
   noticeably slower per the user's report).
3. Logged `alpha_weights` and `alpha_scale` show non-uniform drift.

These are validated by running the 9-cell ablation; the smoke test
above only verifies the API correctness, not the end-to-end training
behaviour.

## Pending

- Run the 9-cell ablation on `section3_done` to validate criteria 1–3.
- Mirror the code-only changes onto `clean` (cherry-pick the commit;
  this doc stays only on `section3_done`).
- BuilderBench `rl/impls/continual_crl.py` Fix E (head naming) is also
  pending commit on `main`.
