# How to run the decomposed-critic algorithm

Quickstart-style instructions, written for a fresh shell on the HPC
cluster after pulling the latest `section3_done`. Assumes the existing
sgcrl environment is already set up and the existing 9-cell ablation
runs without errors.

> **Status as of 2026-05-08.** The networks, learner, config flags,
> and runner scaffolding are committed. The runner branch that
> dispatches to `ContinualDecomposedLearner` from `train_single_task`
> is **not yet active** — see "Step 0" below for the three-block patch
> that activates it. Once that patch lands, all the steps below run
> as-is.

## TL;DR

```bash
# After the runner-glue patch lands (see Step 0):
python run_continual_contrastive.py \
    --actor_mode=reset \
    --critic_mode=decomposed \
    --dyn_aux_weight=1.0 \
    --num_tasks=10 \
    --seed=42 \
    --use_wandb=True
```

Flags that already exist (`--actor_mode`, `--critic_mode`, `--seed`,
`--use_wandb`, etc.) keep their meaning. The new ones live in the
`ContinualConfig` dataclass and can be set via a custom config or by
reaching into `continual_cfg` from `experiment_configs.py`.

## Step 0 — finish the runner glue

`train_single_task` in `run_continual_contrastive.py` already accepts
the eight new optional kwargs (`prev_b_shared_params`,
`prev_h_phi_params`, etc.) and imports the new modules at the top.
What is left:

1. **Around line 362 (network construction)**, before the existing
   `networks = contrastive.make_networks(...)` call, add:

   ```python
   decomp_nets = None
   if critic_mode == 'decomposed':
       decomp_nets = make_decomposed_networks(
           env_spec, obs_dim=obs_dim,
           repr_dim=config.repr_dim,
           use_residual=config.use_residual,
           network_width=config.network_width,
           critic_depth=config.critic_depth,
           phi_task_width=continual_cfg.phi_task_width,
           phi_task_depth=continual_cfg.phi_task_depth,
       )
   ```

   The existing `make_networks(...)` call still happens; we read its
   policy / sample / log_prob.

2. **Around line 481 (learner construction)**, wrap the existing
   `learner = ContinualContrastiveLearner(...)` in an `if/else`:

   ```python
   if critic_mode == 'decomposed':
       learner = ContinualDecomposedLearner(
           decomp_nets=decomp_nets,
           policy_network=networks.policy_network,
           sample_fn=networks.sample,
           log_prob_fn=networks.log_prob,
           rng=rng,
           iterator=iterator,
           counter=counting.Counter(),
           logger=learner_logger,
           config=config,
           continual_config=continual_cfg,
           task_id=task_id,
           prev_b_shared_params=prev_b_shared_params,
           prev_b_shared_opt_state=prev_b_shared_opt_state,
           prev_h_phi_params=prev_h_phi_params,
           prev_h_phi_opt_state=prev_h_phi_opt_state,
           prev_h_dyn_params=prev_h_dyn_params,
           prev_h_dyn_opt_state=prev_h_dyn_opt_state,
           prev_psi_params=prev_psi_params,
           prev_psi_opt_state=prev_psi_opt_state,
       )
   else:
       learner = ContinualContrastiveLearner(
           # ... existing kwargs unchanged ...
       )
   ```

3. **Around line 800 (post-task state extraction)** and line 889
   (`out_q_params = learner.q_params`), guard the legacy paths and
   add a parallel decomposed extraction. Simplest form:

   ```python
   if critic_mode == 'decomposed':
       # Decomposed path: actor / pool plumbing not used.
       out_theta_base = None
       v_k = None
       out_q_params = None
       out_target_q_params = None
       out_q_optimizer_state = None
       decomposed_state = dict(
           b_shared_params=learner.b_shared_params,
           b_shared_opt_state=learner.b_shared_opt_state,
           h_phi_params=learner.h_phi_params,
           h_phi_opt_state=learner.h_phi_opt_state,
           h_dyn_params=learner.h_dyn_params,
           h_dyn_opt_state=learner.h_dyn_opt_state,
           psi_params=learner.psi_params,
           psi_opt_state=learner.psi_opt_state,
       )
   else:
       # ... existing legacy extraction ...
       decomposed_state = {}
   ```

   Then return `decomposed_state` alongside the other outputs from
   `train_single_task`. The outer loop in `main()` plumbs the
   relevant items into the next call's `prev_*` kwargs.

4. **In `main()`**, where the for-loop over tasks calls
   `train_single_task(...)`, plumb the new kwargs:

   ```python
   prev_decomposed = {}
   for task_id, env_name in enumerate(task_sequence):
       result = train_single_task(
           # ... existing kwargs ...
           prev_b_shared_params=prev_decomposed.get('b_shared_params'),
           prev_b_shared_opt_state=prev_decomposed.get('b_shared_opt_state'),
           prev_h_phi_params=prev_decomposed.get('h_phi_params'),
           prev_h_phi_opt_state=prev_decomposed.get('h_phi_opt_state'),
           prev_h_dyn_params=prev_decomposed.get('h_dyn_params'),
           prev_h_dyn_opt_state=prev_decomposed.get('h_dyn_opt_state'),
           prev_psi_params=prev_decomposed.get('psi_params'),
           prev_psi_opt_state=prev_decomposed.get('psi_opt_state'),
       )
       # ... unpack result, including decomposed_state ...
       if critic_mode == 'decomposed':
           prev_decomposed = decomposed_state
   ```

These three blocks are the entirety of the runner glue. Once
applied, the rest of the runner (replay, env, eval, W&B logging)
works as-is because the decomposed learner exposes a
`policy_params` accessor and the `'policy'` variable client still
serves the actor exactly the same way.

## Step 1 — regression check

Before launching the new mode, verify nothing existing broke:

```bash
python run_continual_contrastive.py \
    --actor_mode=reset \
    --critic_mode=persistent \
    --num_tasks=2 \
    --steps_per_task=200000 \
    --seed=42 \
    --use_wandb=False
```

Compare loss / success curves against the latest known-good
`section3_done` run (same seed, same flags). They should be
bit-for-bit identical (modulo numerical noise from imports). If they
differ, something in the additive imports broke. Roll back to the
previous commit and bisect.

## Step 2 — single-task smoke

Run task 0 only with the decomposed critic, dynamics aux off:

```bash
python run_continual_contrastive.py \
    --actor_mode=reset \
    --critic_mode=decomposed \
    --num_tasks=1 \
    --steps_per_task=200000 \
    --seed=42 \
    --use_wandb=True \
    # --dyn_aux_weight=0.0  # default is 1.0; override here
```

Watch for:

- `learner/critic_loss` decreasing as in the persistent baseline.
- `learner/binary_accuracy` climbing into `> 0.9` within a few
  hundred thousand steps.
- `learner/decomp/L_dyn` decreasing monotonically. The default
  `phi_task_width=256, phi_task_depth=2` is small relative to the
  shared body (1024x4), so the body sees most of the InfoNCE gradient.
- Wall clock per step within ~20% of the persistent baseline. The
  decomposed forward has one extra small encoder; the dyn auxiliary
  is one linear projection plus an MSE.

## Step 3 — full task sequence (single seed)

```bash
python run_continual_contrastive.py \
    --actor_mode=reset \
    --critic_mode=decomposed \
    --dyn_aux_weight=1.0 \
    --num_tasks=10 \
    --seed=42 \
    --use_wandb=True
```

Diagnostics that should already log:

- `learner/decomp/L_dyn` per task. Should drop fast at task 0 (the
  body has nothing to lose), then stay low across task boundaries
  if the body has internalised dynamics.
- standard sgcrl metrics (`critic_loss`, `binary_accuracy`,
  `categorical_accuracy`, `actor_loss`, `entropy_mean`, `alpha`).

## Step 4 — ablation grid

`docs/2026-05-08_plan_proposal1_dyn_aux.md` section 8 lists the
target grid:

| dyn_aux_weight | actor_mode | critic_mode | description           |
|----------------|------------|-------------|-----------------------|
| —              | reset      | persistent  | existing baseline     |
| 0.0            | reset      | decomposed  | decomp split, no dyn  |
| 0.1            | reset      | decomposed  | weak dyn aux          |
| 1.0            | reset      | decomposed  | full dyn aux          |
| 1.0            | reset      | reset       | dyn aux + reset critic|

Five cells, five seeds each, 10 tasks per cell. The existing 9-cell
grid is unchanged; this is an additive column.

## Diagnostics still pending

The plan calls for two diagnostics that are not yet implemented:

- Linear-probe task classifier on `b_shared(s, a)` after each task.
  Trains a logistic regression on a held-out batch and reports test
  accuracy. Target: near `1 / num_tasks` (chance). A run climbing
  significantly above chance flags the body absorbing task identity.
- Per-index masking sensitivity for the contrastive head. Mask one
  state index at a time at evaluation; record the drop in
  within-task accuracy. Object-slot indices should not be the
  dominant features.

Both are eval-time only and do not affect training. Implement after
the first single-cell sanity result.

## Troubleshooting

- **`ValueError: critic_mode=decomposed requires adaptive_entropy=True`**:
  flip `config.adaptive_entropy = True` and set `target_entropy = -2.0`.
- **`ValueError: critic_mode=decomposed does not support config.use_td=True`**:
  set `use_td=False`. The TD path is intentionally out of scope.
- **`ValueError: critic_mode=decomposed does not support twin_q`**:
  set `twin_q=False` for the same reason.
- **Pool-cosine logs filling W&B**: `log_pool_cosine` is independent of
  the decomposed mode. Set it to False when running the decomposed
  cells (it has nothing useful to log there since the pool is empty).
- **Memory blow-up at task 5+**: the only growing state is the
  optimiser running averages on `b_shared` (bounded by parameter
  count, not task count). If you see growth, check that the runner
  does not accidentally accumulate per-task copies of the state in
  the outer loop.

## Where to look in the code

- `contrastive/state_mask.py` — the stable-index mask `M`.
- `contrastive/decomposed_networks.py` — `make_decomposed_networks`,
  `DecomposedCriticNetworks`, `apply_score`, `apply_b_shared`,
  `apply_h_dyn`.
- `contrastive/continual_learning_decomposed.py` —
  `ContinualDecomposedLearner`, `DecomposedTrainingState`. The inner
  JIT step (`update_step`) is the heart of the algorithm.
- `contrastive/continual_config.py` — `dyn_aux_weight`,
  `phi_task_width`, `phi_task_depth`.
- `run_continual_contrastive.py` — imports + signature scaffolding;
  active glue still to land per Step 0 above.
