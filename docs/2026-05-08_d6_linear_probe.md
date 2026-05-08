# D6: linear-probe task classifier (`eval_linear_probe.py`)

Date: 2026-05-08

What ships:

- `run_continual_contrastive.py`: per-task `(obs, action)` sample
  dumped to `probe_data_task{k}_seed{s}.npz` next to the checkpoint
  when `continual_config.log_probe_data=True`. Off by default.
- `contrastive/continual_config.py`: new flag
  `log_probe_data: bool = False`.
- `eval_linear_probe.py`: new top-level script. Loads the per-task
  probe data + final checkpoint, builds the trained `b_shared`
  (decomposed) or `q_network` sa-encoder hidden output (fallback),
  fits a closed-form softmax probe with ridge least-squares, reports
  overall and per-task accuracy plus a row-normalised confusion
  matrix.

## Why this metric

Plan section 3.4:

> Healthy `b_shared` should produce features that do **not** easily
> classify task identity. Failure mode: `b_shared` absorbs task
> identity when exposed to mixed-task gradients.
>
> Eval-time-only diagnostic (no training cost): freeze `b_shared`, fit
> a linear classifier on `b_shared(s, a)` to predict task index,
> report test accuracy. Target is near-chance (1/N for N tasks); a
> probe accuracy above ~50% in the 10-task setting flags the body
> absorbing task identity.

For the decomposed column this is the third empirical pillar (after
pool-cosine D1-D4 and mixture_norm D5) of the negative-result figure.

## Design choices

- **Sample size = `config.batch_size`** (256 default). User-confirmed.
  One `next(iterator)` at end-of-task, sliced to the first
  `batch_size` rows. Replay is HER-relabeled at this point so the obs
  layout matches what the learner trained on.
- **Closed-form softmax via JAX least-squares.** Solve
  `(Phi^T Phi + ridge * I) W = Phi^T Y` where `Phi` is the feature
  matrix with a bias column appended and `Y = one_hot(y)`. Argmax of
  `Phi @ W` is the prediction. Ridge default `1e-4` keeps the system
  well-conditioned even when `hidden_dim > N_train`. No sklearn
  dependency.
- **Decomposed-only PLUS a fallback.** Plan §3.4 is decomposed-only,
  but the user opted into a fallback so we can compare the decomposed
  body's probe accuracy against the existing critic's sa-encoder
  hidden output. The `make_networks` function already exposes
  `critic_hidden_repr_fn(q_params, obs, action) -> (sa_hidden,
  g_hidden)` which is the natural analog of `b_shared`'s output and
  shares the same `ResidualMLP` body shape.

## Run

End-of-training (decomposed column):

```bash
python eval_linear_probe.py \
    --checkpoint_dir=logs/continual_goal_crl \
    --seed=42 \
    --num_tasks=10 \
    --critic_mode=decomposed \
    --actor_mode=reset
```

Comparison baseline on a non-decomposed column:

```bash
python eval_linear_probe.py \
    --checkpoint_dir=logs/continual_goal_crl \
    --seed=42 \
    --num_tasks=10 \
    --critic_mode=cka \
    --actor_mode=cka
```

The script prints overall train / test accuracy, per-task test
accuracy, chance level (1 / num_tasks), and a row-normalised
confusion matrix on the test set. A short interpretation block at the
bottom flags PASS / FAIL / borderline against the plan §3.4 thresholds
(near chance is PASS for the decomposed column; > 0.5 is FAIL).

## Smoke (local)

Two unit tests on the closed-form probe (no JAX runtime issues since
this only uses `jnp.linalg.solve`):

| Scenario | Train acc | Test acc | Chance |
|---|---|---|---|
| One-hot features (trivially separable) | 1.0000 | 1.0000 | 0.10 |
| Pure noise features | 0.198 | 0.104 | 0.10 |
| Separated Gaussians | n/a | 1.0000 | 0.10 |

End-to-end through the decomposed `apply_b_shared` path (stubbed
acme, real haiku, real JAX):

| `b_shared` | Inputs | Probe test acc | Chance | Verdict |
|---|---|---|---|---|
| random init | per-task separable Gaussians | 1.0000 | 0.20 | flags leakage (correct: random body preserves separability) |
| constant-output | per-task separable Gaussians | 0.1875 | 0.20 | near chance (correct: invariant features defeat the probe) |

The same probe code distinguishes the failure case from the
invariant-features case at random init, so on real trained
checkpoints it will distinguish leakage from genuine task-agnostic
features.

## Caveats / what to watch for

- **The script assumes the run used the `ContrastiveConfig` /
  `ContinualConfig` defaults** (network_width, critic_depth,
  phi_task_*, energy_fn, repr_norm, etc.). If a run overrode any of
  these, the script as-is will rebuild the wrong network and the
  loaded params will fail to bind. Future work: persist the relevant
  config fields in the checkpoint dict and read them back here. For
  now, document this as a known limitation.
- **The fallback target (`q_network` sa-encoder hidden) is a
  comparison baseline only.** Plan §3.4 does not set a threshold on
  this column; we do not assert PASS / FAIL there.
- **One sample of 256 transitions per task** may be small for
  fine-grained per-class confusion. If signal/noise is borderline,
  re-run with a larger sample size (raise `config.batch_size`
  temporarily for a probe run, or change the sample size in
  `_dump_probe_data`).

## Files touched

- `run_continual_contrastive.py` (new `_dump_probe_data` helper before
  the post-task extraction; runs once per task, gated on
  `log_probe_data`).
- `contrastive/continual_config.py` (`log_probe_data: bool = False`).
- `eval_linear_probe.py` (this file's main artifact, ~399 lines).
- `docs/2026-05-08_d6_linear_probe.md` (this file, new).
- `docs/2026-05-08_implementation_tracking.md` (D6 row).

No public API changes outside the added flag and the new top-level
script. Default `log_probe_data=False` keeps every existing run path
bit-identical.
