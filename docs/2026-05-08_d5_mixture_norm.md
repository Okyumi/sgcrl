# D5: per-step `mixture_norm` metric in the CKA inner loop

Date: 2026-05-08

What ships:

- `contrastive/knowledge_pool.py`: new helper `mixture_to_vk_ratio`
  computing `|| sum_j alpha_j v_j || / || v_k ||` (L2 over flat).
- `contrastive/continual_learning.py`: when `actor_cka_path` and
  `critic_cka_path` are active and `continual_config.log_mixture_norm`
  is `True`, emits `cka/actor_mixture_norm` and
  `cka/critic_mixture_norm` per inner step.
- `contrastive/continual_config.py`: new flag
  `log_mixture_norm: bool = False` (default off, existing runs
  unchanged).

## Why this metric

Plan section 3.2:

> Hypothesis: even when `alpha` is technically trainable (task `k >= 3`
> under canonical CKA-RL hand-off), most of the actor update is being
> absorbed by `v_k` rather than the mixture `sum_j alpha_j v_j`,
> because the mixture term is approximately constant in that regime.
>
> If this is small (< 0.1) throughout training, the mixture term is
> effectively a small bias that the actor learns to ignore. The paper
> can then say that CKA degrades to per-task residual finetuning, which
> is what we observe.

Combined with the pool-cosine logging shipped earlier (D1-D4), this is
the second empirical pillar of the negative-result figure. Both
together support the conclusion: pool entries collapse to be nearly
parallel and the mixture they produce is too small relative to `v_k`
for the actor to use.

## Implementation choice: per inner step, not per task boundary

Plan section 5.3 listed `cka/mixture_norm` under "per-step metrics".
Computing it inside the inner JIT loop costs:

- `compute_contribution` (a softmax + a `tree_map` of weighted sum).
- Two `linalg.norm` calls (one over `contribution`, one over `v_k`).

That cost is dominated by the existing inner-loop work (Adam + loss
backward). Logging gates on a Python-level `if log_mixture_norm:` so
the flag-off path is bit-identical to the prior code.

## Math equivalence to plan formula

`compute_contribution(pool, alpha_logits, alpha_scale)` returns a pytree
shaped like one knowledge vector, holding `sum_j alpha_j v_j` where
`alpha_j = softmax(alpha_logits * alpha_scale)_j` masked over the
active pool slots. `mixture_to_vk_ratio(contribution, v_k)` flattens
both to 1D and computes `|| contribution || / (|| v_k || + eps)`. The
epsilon (1e-12) is a numerical guard for task 0 / very early steps when
`v_k` is still close to zero; in steady state it does not affect the
metric (1e-12 is much smaller than any realistic `||v_k||`).

## Smoke (local)

Three sanity cases verified by direct unit test (no JIT, no acme):

| `||contribution||` | `||v_k||` | expected ratio | observed ratio |
|---|---|---|---|
| 1.0 | 1.0 | 1.0 | 1.0000 |
| 1.0 | 10.0 | 0.1 | 0.1000 |
| 1.0 | 0.0 | ~1e12 (eps-guarded) | 9.9999...e11 |

Built a 4-capacity pool with two active slots holding orthogonal
vectors v1=[1,1,0,0,0,0] and v2=[0,0,1,1,0,0]; uniform alpha; verified
`compute_contribution` returns 0.5*(v1+v2) before testing
`mixture_to_vk_ratio`.

## How to use

```bash
python run_continual_contrastive.py \
    --critic_mode=cka --actor_mode=cka \
    [other defaults preserved]
# In your run script or config override:
#   continual_config.log_mixture_norm = True
```

W&B keys to monitor:

- `cka/actor_mixture_norm` (when `actor_mode='cka'`)
- `cka/critic_mixture_norm` (when `critic_mode='cka'`)

Existing `actor_alpha_max / actor_alpha_entropy / actor_alpha_scale`
keys (and critic equivalents) are untouched and continue to log
exactly as before.

## Expected pattern

If the audit hypothesis is right (pool-cosine off-diagonal close to
1.0):

- Task 1: mixture_norm = 0 (empty pool, contribution = 0 vector).
- Task 2: mixture_norm fluctuates, single active slot.
- Task 3+: mixture_norm settles below 0.1 once the v_k optimisation
  drives `||v_k||` past `||contribution||`. If this does not happen
  (mixture_norm stays > 0.5), the failure is in the training dynamics,
  not in the pool collapsing — a different paper claim.

## Files touched

- `contrastive/knowledge_pool.py` (added `mixture_to_vk_ratio`).
- `contrastive/continual_learning.py` (import + two-branch wiring in
  the inner-loop diagnostics block, gated on `log_mixture_norm`).
- `contrastive/continual_config.py` (`log_mixture_norm: bool = False`).
- `docs/2026-05-08_d5_mixture_norm.md` (this file, new).
- `docs/2026-05-08_implementation_tracking.md` (D5 row).

No public API changes outside the added flag and helper. The default
`log_mixture_norm=False` keeps every existing run path bit-identical.
