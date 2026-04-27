# W&B analysis: actor_mode=cka at task 1

Snapshot of three runs at task 1 (8M env steps each), all with
`actor_mode=cka`, varying `critic_mode` over `{cka, reset, persistent}`.

## Verdict on the previous alpha issue

Not resolved. The trainable bundle `(v_k, alpha_logits, alpha_scale)`
is wired correctly inside the JIT loop, but at task 1 the pool is
empty, so `alpha_logits` and `alpha_scale` receive zero gradient and
never move.

## Per-metric reading

### Critic metrics (image 1, top row)

- `critic_loss` ≈ 1.5–3 across all three. Climbs from ~1 in the first
  ~1M steps to a plateau. All three curves overlap, which is the
  expected behavior when the critic-side carry-over does not change
  the early dynamics (the pool is empty / unused for both `cka` and
  `reset`, and `persistent` carries forward weights but the InfoNCE
  loss re-equilibrates within the first task). Healthy.
- `categorical_accuracy` peaks early at ~0.9 and decays to ~0.4–0.55
  by 8M. This is row-wise classification on the in-batch InfoNCE
  matrix; the early peak reflects easy negatives at random init, the
  decay reflects increasing batch difficulty as the critic sharpens.
  Pattern matches a healthy contrastive critic.
- `binary_accuracy` ≈ 0.99 throughout. The diagonal-vs-off-diagonal
  classification: the critic almost always ranks the matched
  (state, action, goal) above unmatched. This is expected and stable.

### Actor / entropy metrics (image 1, bottom row + image 2 right)

- `actor_loss` spikes to ~250 at the start, drops sharply within ~1M
  steps to ~0, and stays there. Standard SAC-style actor warmup.
- `alpha_loss` ≈ 0 throughout. The dual entropy controller has hit
  the target (≈ −2) so the gradient on log α is near-zero. Healthy.
- `alpha` (the SAC entropy coefficient, not the CKA pool weights)
  decays from ~0.5 to ~0.1. Standard.
- `entropy_mean` settles around −4. Slightly below target entropy of
  −2; still in the healthy band for a tanh-squashed Gaussian.

### Critic-quality diagnostics (image 2)

- `logsumexp` collapses from ~50 to ~0 within ~1M steps and stays
  flat. The log-partition regulariser is doing its job.
- `logits_pos` ≈ −1 to −3. Diagonal logits are bounded.
- `logits_neg` ≈ −100 to −200, drifting up over training. Off-diagonal
  logits being highly negative is the InfoNCE signature of a
  well-separated critic.
- `env_steps` is a linear ramp 0 → 8M, confirming this snapshot is one
  full task (`continual_cfg.steps_per_task = 8M`).
- `steps_per_second` ~150–300 with high variance, occasional dips to
  ~50. That's the per-call wall-clock of `_update_step`; the dips are
  likely XLA recompilation from variable-shape data along the actor
  path. Not a correctness issue.

### CKA alpha diagnostics (image 3)

This is the row that exposes the regression.

- `actor_alpha_scale` flat at 1.0 across all three runs.
- `actor_alpha_max` flat at 0.0 across all three runs.
- `actor_alpha_entropy` flat at 0.0 across all three runs.
- `critic_alpha_scale` flat at 1.0 (cyan run only — the only run with
  `critic_mode=cka`).
- `critic_alpha_max` flat at 0.0.
- `critic_alpha_entropy` flat at 0.0.

Walking the masked-softmax forward:

```
masked_logits[j] = alpha_logits[j] * alpha_scale  if mask[j]
                   -inf                            otherwise
softmax(masked_logits) = uniform-zero  if all mask are False
                         else proper distribution
```

With a single active pool slot the softmax is `[1, 0, ...]`, so
`alpha_max` would be 1.0. Observed value 0.0 is therefore the
diagnostic signature of `mask = [False, False, ...]` — i.e., the
runtime pool is empty at task 1.

`alpha_scale` stuck at 1.0 corroborates: with an empty pool, the
contribution `Σ α_j v_j` evaluates to a zeros pytree regardless of
`alpha_logits` or `alpha_scale`, so neither receives any gradient and
both remain at their init values forever.

## Where the empty pool comes from

`run_continual_contrastive.py` lines 800–814. After task 0:

```python
if task_id == 0:
    out_theta_base = jax.tree_util.tree_map(
        lambda b, v: b + v, learner.theta_base, v_k)
    # Fix C: do NOT seed the pool with a zero vector...
    # (no append at all)
```

The April 26 audit named two failure modes for the task-0 → task-1
hand-off:

- legacy: `pool.append(zero_vector_like(theta_base))`. Wrong — the
  zero vector permanently dilutes alpha mass without contributing
  anything (Bug 4).
- post-Fix-C (current): no append at all. Also wrong — the pool is
  genuinely empty at task 1, so alpha never sees gradient.

The right behaviour is the original CKA-RL formulation: keep
`theta_base = init` (the random init from task 0), and append
`v_0 = trained_task0 - init` to the pool. Then at task 1 the
composition is `theta_base + α_1·v_0 + v_1` and `α_1` has a real
quantity to weight. Equivalent variants:

- `theta_base = trained_task0`, `pool = {v_0}`. At task 1 the
  composition is `theta_base + α_1·v_0 + v_1`, which double-counts v_0
  by `1 + α_1` rather than `1`. Non-clean.
- `theta_base = init`, `pool = {v_0}` (CKA-RL canonical). Composition
  is `init + α_1·v_0 + v_1`. Clean.

## What the metrics would look like once fixed

At task 1, with the canonical pool seed:

- `actor_alpha_max` near 1.0 at start (single active slot ⇒
  softmax = `[1, 0, ...]`), drifting toward whatever value the
  optimiser settles on.
- `actor_alpha_entropy` rises from 0 once the pool grows past one
  slot.
- `actor_alpha_scale` drifts away from 1.0; magnitude depends on how
  much the loss prefers a sharp vs. flat distribution.
- `critic_alpha_*` analogously, but only when `critic_mode=cka`.

Then the canonical comparison (cka vs persistent vs reset on each
side) becomes meaningful, since cka actually exercises a different
forward pass.

## Recommended fix

In `run_continual_contrastive.py`, replace the no-op task-0 branch
with the canonical CKA-RL hand-off:

```python
if task_id == 0:
    # Keep theta_base at the random init from task 0 and put the
    # task-0 delta into the pool. CKA-RL canonical form (Hu et al.,
    # 2025).
    pool.append(v_k)
    # out_theta_base = theta_base unchanged (it is the random init).
    out_theta_base = learner.theta_base
elif adapt_heads_only:
    ...
```

This restores non-empty pool entry to task 1 without reintroducing
the zero-vector dilution that Fix C set out to remove.
