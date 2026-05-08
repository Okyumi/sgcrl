# Decomposed-critic verification against SGCRL conventions

Date: 2026-05-08

Purpose: verify the just-shipped decomposed-critic implementation
(`contrastive/decomposed_networks.py`,
`contrastive/continual_learning_decomposed.py`) matches the existing
SGCRL conventions before continuing with the runner glue (N4b).

## Findings and fixes

### 1. Score function — fixed

- Existing convention: `energy_fn` defaults to `'inner_product'`
  (`contrastive/config.py:76`). `_combine_repr` in
  `contrastive/networks.py:258-266` returns
  `jnp.einsum('ik,jk->ij', sa, g)` for inner product, falling back to
  negative-L2 only when `energy_fn='l2'`.
- Original implementation: `apply_score` in `decomposed_networks.py`
  unconditionally returned `-||sa - g||_2`.
- Fix: `apply_score` now branches on the `energy_fn` argument and
  defaults to inner product. Added an optional `repr_norm` flag that
  L2-normalises sa and g before scoring, mirroring the
  `repr_norm=True` branch of `make_networks`.

### 2. Adaptive entropy detection — fixed

- Existing convention (`continual_learning.py:170`):
  `adaptive_entropy = config.entropy_coefficient is None`.
  `ContrastiveConfig` has no `adaptive_entropy` field.
- Original implementation: read `config.adaptive_entropy`, which would
  have raised `AttributeError` at construction.
- Fix: detect with the same `entropy_coefficient is None` rule. Error
  message updated to point users at `entropy_coefficient` rather than a
  non-existent `adaptive_entropy` flag.

### 3. InfoNCE form — fixed

- Existing convention (`continual_learning.py:364-369`): `use_cpc=False`
  default uses `optax.sigmoid_binary_cross_entropy(logits, I)`;
  `use_cpc=True` uses `softmax_cross_entropy + logsumexp_penalty * lse**2`.
- Original implementation: a hand-rolled
  `-mean(diag(score) - logsumexp(score, axis=1))` (softmax-CE form),
  which silently switched the loss family.
- Fix: branch on `config.use_cpc`. Default path is sigmoid-BCE.
  Diagnostics (`binary_accuracy`, `categorical_accuracy`, `logits_pos`,
  `logits_neg`, `logsumexp`) now match the existing learner's keys.

### 4. Actor loss — fixed

- Existing convention (`continual_learning.py:408-438`):
  - Roll goals according to `config.random_goals` (0.0 / 0.5 / 1.0).
  - Apply policy on the (possibly rolled) `new_obs`, sample, take
    `q_action = q_network.apply(...)[0]`.
  - `actor_loss = -jnp.diag(q_action)`; if `config.use_action_entropy`,
    `actor_loss -= alpha * (-log_prob)`.
- Original implementation: no goal-rolling; entropy was always added
  whenever `adaptive_entropy=True` (different flag and different
  semantics). `score` was wrapped in `stop_gradient` for the critic
  params, which is harmless but unnecessary because critic params are
  not in the `value_and_grad` argnums.
- Fix: replicated the goal-rolling block, replaced the
  `adaptive_entropy` gate on the entropy term with
  `config.use_action_entropy`, removed the now-unused
  `stop_gradient` on critic params. Sign is exactly
  `-= alpha * (-log_prob)` to match line 435.

### 5. Hyperparameter audit — clean

| Field | SGCRL config default | Decomposed learner | Status |
|---|---|---|---|
| `learning_rate` | 3e-4 | 3e-4 (Adam) | match |
| `actor_learning_rate` | 3e-4 | 3e-4 | match |
| `batch_size` | 256 | from runner | match |
| `num_sgd_steps_per_step` | 64 | scanned in `scan_step` | match |
| `repr_dim` | 64 | 64 | match |
| `energy_fn` | `inner_product` | now configurable, default `inner_product` | match |
| `repr_norm` | False | now configurable, default False | match |
| `logsumexp_penalty` | 0.01 | honoured under `use_cpc=True` | match |
| `use_cpc` | False | branch added | match |
| `use_action_entropy` | True | branch added | match |
| `random_goals` | 0.5 | branch added | match |
| `use_td` | False | hard-rejected (raises) | intentional |
| `twin_q` | False | hard-rejected (raises) | intentional |
| `network_width` | 256 (config); 1024 (run config) | builder default 1024; runner threads `config.network_width` | match |
| `critic_depth` | 4 | 4 | match |
| `actor_depth` | 4 | from runner / `make_networks` | match |
| `target_entropy` | 0.0 (config); −2.0 (run config) | reads `config.target_entropy` | match |

## Local smoke

`python` with stubbed `acme.jax.utils` (production env runs JAX 0.4.10
on the cluster; local has incompatible JAX). Confirmed:

- `apply_score` returns `(B, B)` for `inner_product`, `l2`, and
  `repr_norm=True` paths.
- `inner_product` produces a non-trivial score matrix
  (`std≈16` at random init, batch=8).
- `l2` produces all `≤ 0` entries.
- `repr_norm=True` clamps every entry to `[-1, 1]`.
- Gradient isolation is preserved under the inner-product score:
  - InfoNCE-only loss → `∇ h_dyn = 0`.
  - Dyn-only loss → `∇ h_phi = ∇ phi_task = ∇ psi = 0`.

## Files touched

- `contrastive/decomposed_networks.py`
  (added `energy_fn`, `repr_norm` builder kwargs; rewrote
  `apply_score`).
- `contrastive/continual_learning_decomposed.py`
  (`adaptive_entropy` detection; rewrote `critic_loss_fn` to branch on
  `use_cpc`; rewrote `actor_loss_fn` to roll goals and gate entropy on
  `use_action_entropy`).
- `docs/2026-05-08_how_to_run_decomposed.md`
  (Step-0 patch now passes `config.energy_fn` and `config.repr_norm`
  through to `make_decomposed_networks`).
- `docs/2026-05-08_decomposed_critic_implementation.md`
  (same Step-0 patch update).

No public API changes: kwargs are additive with sane defaults.

## Next

N4b: apply the (now correct) Step-0 / Step-1 / Step-2 patch in
`run_continual_contrastive.py`, then run the one-task baseline-match
smoke (item 6 on the todo list).
