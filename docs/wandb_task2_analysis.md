# W&B analysis: actor_mode=cka at task 2

Snapshot of three runs at task 2 (8M env steps), all with
`actor_mode=cka`, varying `critic_mode` over `{cka, reset, persistent}`.
Companion to `docs/wandb_task1_analysis.md`.

## Verdict on the previous alpha issue

Still not resolved. Task 2 surfaces the symptom more clearly than
task 1: the pool now has exactly one entry (the task-1 delta `v_1`),
which is too few slots for the masked softmax to produce any
gradient on `(alpha_logits, alpha_scale)`. The trainable bundle wiring
is correct, but the pool population law makes the bundle structurally
non-trainable until the pool grows to ≥ 2 active slots.

## Per-metric reading

### CKA alpha diagnostics (image 1)

This is now the most informative panel.

- `actor_alpha_max` flat at 1.0 across all three runs.
- `actor_alpha_scale` flat at 1.0 across all three runs.
- `actor_alpha_entropy` flat at 0.0 across all three runs.
- `critic_alpha_max` flat at 1.0 (cyan run only — only run with
  `critic_mode=cka`).
- `critic_alpha_scale` flat at 1.0.
- `critic_alpha_entropy` flat at 0.0.

`alpha_max = 1.0` is the diagnostic signature of a **single active
pool slot**: with `mask = [True, False, False, False, False, False]`,
the masked softmax is `[1, 0, 0, 0, 0, 0]` regardless of
`alpha_logits` and `alpha_scale`. Maximum is 1, entropy is 0.

`alpha_scale = 1.0` confirms the bundle never moves. Walking the
gradient: the contribution `Σ α_j v_j` collapses to `α_0 · v_0` with
α_0 ≡ 1 (forced by the mask, not by the softmax temperature). So
`d(loss)/d(alpha_scale) = 0` and `d(loss)/d(alpha_logits) = 0`. Adam
on a zero gradient leaves the parameter untouched.

### Critic-side (image 3 top + image 2)

- `critic_loss` mid-task plateau ~1.0–1.5. Cyan (`critic_mode=cka`)
  sits slightly higher than green/orange around 4M. Within noise band
  for one task.
- `categorical_accuracy` ~0.55–0.7. Cyan dips lower than the other
  two between 2M and 5M, recovers by 6M.
- `binary_accuracy` ~0.997 across all three. Healthy.
- `logsumexp` collapses to ~2 quickly and stays. Healthy regulariser.
- `logits_pos` ~−1.5. Cyan a bit lower than the other two between
  2M and 5M (matches the `categorical_accuracy` dip).
- `logits_neg` ~−25 to −30 stable. The early spike to −150 is the
  warm-up; the recovery to ~−25 means off-diagonals are still well
  separated. Less aggressive than at task 1, which is expected as the
  critic refines after task transitions.

### Actor / SAC entropy (image 3 bottom + image 2 right)

- `actor_loss` warmup ~70 → ~10–20 plateau. Cyan and green plateau
  lower than orange (persistent).
- `alpha_loss` ≈ 0 throughout. Dual entropy on target.
- `alpha` (SAC entropy coefficient) settles ~0.05–0.1. Standard.
- `entropy_mean` ~−4. Within healthy band.
- `steps_per_second` ~150–300, with cyan slightly slower than green
  and orange (an extra optimiser pass for the critic CKA bundle).

## Side-by-side: task 1 vs task 2

| Metric | Task 1 | Task 2 |
|---|---|---|
| `actor_alpha_max` | 0.0 | 1.0 |
| `actor_alpha_scale` | 1.0 | 1.0 |
| `actor_alpha_entropy` | 0.0 | 0.0 |
| `critic_alpha_max` (cyan) | 0.0 | 1.0 |

The jump in `alpha_max` from 0.0 to 1.0 is exactly what you would see
when the pool transitions from "empty" to "one active slot". Both
states are equally degenerate from the optimiser's point of view
because the masked softmax has no free degrees of freedom in either.

A non-degenerate gradient on `(alpha_logits, alpha_scale)` requires
**two or more active pool slots**. Below that threshold:

- 0 active slots: `Σ α_j v_j = 0`, the bundle is invisible to the
  forward pass.
- 1 active slot: `Σ α_j v_j = α_0 · v_0` and α_0 ≡ 1 by mask, so the
  bundle is again invisible.
- ≥ 2 active slots: softmax becomes a real distribution that depends
  on `alpha_logits` and `alpha_scale`; gradient flows.

## Cause

The April 26 audit's Fix C dropped the zero-vector pool seed at task
0 to remove dilution. The runner's task-0 branch
(`run_continual_contrastive.py:803`) now does *no* append at all.
Combined with the mask law:

| End of task k | Pool size at start of task k+1 | Active slots |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| 2 | 2 | 2 |
| 3 | 3 | 3 |

The bundle is degenerate at task 1 and task 2 (two of the four most
informative tasks in a typical 10-task ablation), and only starts
training meaningfully at task 3. Three out of every nine ablation
cells (the entire `actor_mode=cka` row) get only seven non-degenerate
training tasks, which heavily contaminates the headline result.

The original CKA-RL formulation puts `v_0` into the pool at the end
of task 0 and keeps `theta_base` at the random init. That gives
**one active slot at task 1**, **two at task 2**, **three at task 3**,
and `alpha_logits + alpha_scale` start receiving gradient at task 2
instead of task 3.

That still leaves task 1 degenerate (one slot). To fix task 1
properly you need *some* second seed vector in the pool at end of
task 0, or to delay the alpha bundle until task 2. The first option
is cleaner and matches several CKA-RL variants that pre-seed with
`v_0` and a small random perturbation.

## Recommended fix (revised)

Two-step:

1. Restore `pool.append(v_0)` at end of task 0 (canonical CKA-RL).
   At task 1 the pool has 1 slot — still degenerate alpha, but
   `composed = init + v_0 + v_1` instead of `theta_base + v_1`, which
   is closer to the canonical form. This unblocks alpha training at
   task 2 onwards.
2. Optionally also append a small Gaussian perturbation of `v_0` as
   a second slot at end of task 0. With two active slots at task 1
   the bundle has a real gradient signal from task 1 onwards. The
   perturbation magnitude can be tied to `continual_config.beta_init_std`
   so it is calibrated to the same scale the rest of the pipeline
   already uses.

Without (2), the metric `alpha_max` will still read 1.0 at task 1
(single slot) and at task 2 (two slots, but logged value is the max
of the softmax which can still hit 1.0 after a sharp distribution
forms). The cleaner test for "is alpha actually training?" is
**`alpha_scale != 1.0` and `alpha_entropy != 0`** — if either
deviates from the init values by mid-task, the bundle is moving.

## What to expect once fixed

After step (1) alone:

- Task 1: `alpha_max = 1.0`, `alpha_scale = 1.0`, `alpha_entropy = 0`
  (degenerate; same as observed). Composed actor matches CKA-RL form
  but bundle does not train.
- Task 2: `alpha_max` between 0.5 and 1.0 depending on softmax
  sharpness, `alpha_scale` drifts away from 1.0, `alpha_entropy > 0`.

After step (1) + step (2):

- Task 1: `alpha_max` between 0.5 and 1.0, `alpha_scale` drifts,
  `alpha_entropy > 0` from the start of task 1.
