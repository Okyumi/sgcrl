# Decomposed critic — implementation log (2026-05-08)

Companion to `2026-05-08_plan_proposal1_dyn_aux.md`. This document
describes what shipped, what is verified locally, and what remains
for the runner integration. Code lives on branches `section3_done`
(with docs) and `clean` (code-only).

## What is the algorithm?

A new `critic_mode='decomposed'` that replaces the single
state-action encoder with three separable components plus a goal
encoder:

```
phi(s, a) = h_phi(b_shared([s; a])) + phi_task([s; a])
psi(g)    = psi(g)
score     = - || phi(s, a) - psi(g) ||_2
```

Two heads sit on the shared body:

- `h_phi`: linear projection to the contrastive embedding (R^d).
- `h_dyn`: linear projection to a masked next-state prediction
  (R^{d_M}, with `d_M = 4` selecting end-effector xyz + gripper).

Two losses train the critic. They share gradient through the body:

- `L_InfoNCE`: standard sgcrl InfoNCE on the (B, B) score matrix.
- `L_dyn = || h_dyn(b_shared(s, a)) - select_stable(s') ||_2^2`.

Per-component gradient routing (the structural commitment that makes
the design work):

| component | InfoNCE | dyn aux | actor obj |
|-----------|:-:|:-:|:-:|
| `b_shared`| Y | Y | - |
| `h_phi`   | Y | - | - |
| `h_dyn`   | - | Y | - |
| `phi_task`| Y | - | - |
| `psi`     | Y | - | - |
| actor     | - | - | Y |

At task `k > 0` boundary: `b_shared, h_phi, h_dyn, psi` and their
optimiser states carry forward; `phi_task` and the actor are reset
fresh. The actor is unchanged from the existing `actor_mode='reset'`
path.

The dynamics target uses the cross-task-stable index subset
`(0, 1, 2, 3)` of the unified Sawyer state (end-effector xyz, gripper
distance), per `docs/2026-02-26_STATE_AND_GOAL_INDEX_SEMANTICS.md`.
Object-slot indices are excluded so a single dynamics head does not
have to smear across task-conditional semantics.

## Files added

- `contrastive/state_mask.py` — `STABLE_INDICES = (0, 1, 2, 3)`,
  `stable_state_mask(...)`, `select_stable(...)`. No external
  dependencies beyond `jax.numpy` and `numpy`. Importable without
  the gym / Meta-World stack.
- `contrastive/decomposed_networks.py` — `make_decomposed_networks`
  builds the five Haiku transforms (`b_shared`, `h_phi`, `h_dyn`,
  `phi_task`, `psi`) and returns a `DecomposedCriticNetworks` bundle
  with init / apply functions plus `apply_sa_repr` and `apply_score`
  convenience wrappers. `b_shared` reuses the existing `ResidualMLP`
  body from `contrastive/networks.py` and stops before the final
  output projection.
- `contrastive/continual_learning_decomposed.py` —
  `ContinualDecomposedLearner` (sibling to
  `ContinualContrastiveLearner`) with its own
  `DecomposedTrainingState` carrying the eleven parameter / opt-state
  pairs. The inner JIT step does:
  1. critic loss `L_InfoNCE` with `value_and_grad` over a dict of the
     four contrastive-trained groups (`b_shared`, `h_phi`, `phi_task`,
     `psi`).
  2. dyn loss `L_dyn` with `value_and_grad` over `(b_shared, h_dyn)`.
  3. compose `b_shared` gradient as `grad_NCE + mu * grad_dyn`.
  4. optax updates per group.
  5. actor SAC-style step against the just-updated composed critic.
  6. SAC dual entropy update.
  Wrapped in `lax.scan` for `num_sgd_steps_per_step` inner steps,
  matching the existing learner's pattern.

## Files modified (additive only)

- `contrastive/continual_config.py` — three new fields:
  `dyn_aux_weight: float = 1.0`, `phi_task_width: int = 256`,
  `phi_task_depth: int = 2`. None of the existing `actor_mode` /
  `critic_mode` paths read these.
- `run_continual_contrastive.py` — added imports and eight optional
  `prev_*_params` / `prev_*_opt_state` kwargs to
  `train_single_task`'s signature. **Behaviour unchanged.** The
  runner does not yet branch into `ContinualDecomposedLearner`; that
  is the remaining glue described below.

## What I verified locally

Local Python/Jax import chain is broken because the workspace doesn't
have a working acme install for our Python version. The smoke checks
I ran were:

- AST parse for every modified file: passes.
- `state_mask` standalone smoke (rendered indices, mask shape,
  `select_stable` shape): passes.
- `decomposed_networks` smoke under a stubbed `acme.jax.utils` that
  provides only `zeros_like` and `add_batch_dim`:
  - networks build with the expected param counts;
  - forward shapes are correct: `score (B, B)`, `hidden (B, hidden)`,
    `dyn_pred (B, d_M)`;
  - gradient isolation is correct: an InfoNCE-only loss produces
    zero gradient on `h_dyn`, a dyn-only loss produces zero gradient
    on `h_phi`, `phi_task`, and `psi`;
  - end-to-end `jax.jit(apply_score)` compiles.

## What remains for the runner

The orchestrator's `train_single_task` is heavily entangled with the
legacy `theta_base / v_k / q_params / pool` API. To run the new
critic, three blocks need surgical edits:

1. **Network construction**, around line 362
   (`networks = contrastive.make_networks(...)`):
   ```python
   if critic_mode == 'decomposed':
       decomp_nets = make_decomposed_networks(
           env_spec, obs_dim=obs_dim,
           repr_dim=config.repr_dim,
           use_residual=config.use_residual,
           network_width=config.network_width,
           critic_depth=config.critic_depth,
           phi_task_width=continual_cfg.phi_task_width,
           phi_task_depth=continual_cfg.phi_task_depth,
           energy_fn=config.energy_fn,        # SGCRL default 'inner_product'
           repr_norm=config.repr_norm,
       )
   ```
   The existing `make_networks(...)` is still needed for the actor;
   keep it but read its `policy_network` / `sample` / `log_prob`
   for the decomposed learner's constructor.

2. **Learner construction**, around line 481
   (`learner = ContinualContrastiveLearner(...)`):
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
       learner = ContinualContrastiveLearner(...)  # existing call
   ```

3. **Post-task state extraction**, around line 800 (`v_k = learner.v_k`)
   and line 889 (`out_q_params = learner.q_params`). These need to be
   guarded so they only run for non-`decomposed` modes, and the
   decomposed branch returns the four `(params, opt_state)` pairs in
   place of the existing `q_params / target_q_params /
   q_optimizer_state` triple. The cleanest fix: extend the function's
   return tuple with a trailing dict whose keys are
   `b_shared_params`, `b_shared_opt_state`, etc., and which is empty
   for non-decomposed cells. The orchestrator's outer loop in
   `main()` then plumbs this dict back into the next call's
   `prev_*` kwargs.

These three edits are mechanical but must be done carefully. I left
the import + signature scaffolding in place so the next pass only has
to fill in the branches, not refactor argument lists.

## Smoke test plan after the runner is wired

Before the full ablation:

1. **Regression check**: launch one cell with `critic_mode='persistent'`
   (decomposed code path inactive). The training curve must be
   bit-for-bit identical to a known-good `section3_done` run, modulo
   seed.
2. **Build / run**: launch one task with `critic_mode='decomposed'`,
   `dyn_aux_weight=0.0`. The dynamics auxiliary is disabled but the
   embedding split is active. Loss curves should be in the same
   ballpark as the persistent baseline; if they diverge wildly, the
   shared-body / task-encoder split itself is breaking the InfoNCE
   matching.
3. **Full single cell**: `dyn_aux_weight=1.0` for one task. `decomp/L_dyn`
   should decrease monotonically. Success rate at end of task 0
   should match the persistent baseline within seed noise.
4. **10-task run**: continue from step 3. Watch:
   - `decomp/L_dyn` per task — dynamics is shared, so this should
     drop again at every task boundary or stay low.
   - `binary_accuracy` and `categorical_accuracy` per task.
   - Linear-probe task classifier (planned diagnostic, not yet
     implemented) on `b_shared(s, a)` to verify the body is not
     absorbing task identity.
5. **Ablation grid**: cells from
   `2026-05-08_plan_proposal1_dyn_aux.md` section 8.

## What is deliberately not done in this push

- No actor CKA. `actor_mode='reset'` only.
- No critic CKA. The decomposed path replaces it.
- No twin-Q. The current code raises if the user enables it with
  `critic_mode='decomposed'`.
- No TD path. The learner raises if `config.use_td=True`.
- No negative bank. Out of scope.
- No mixed-task dynamics buffer. Option A (same-task dynamics) is the
  default. We will switch to option B only if the linear probe shows
  body drift.
- No image observations. The reasoning works the same way but the
  shapes change; we have not validated.

## Risk register (recap from plan)

| risk | likelihood | mitigation in this push |
|------|------------|--------------------------|
| `b_shared` collapses | low | dyn aux gives it a real prediction job |
| `phi_task` absorbs everything | medium | `phi_task` is small (256x2) by default; sweep `mu` |
| `b_shared` drifts task-specific | medium | linear probe (planned diagnostic) |
| dyn easier than InfoNCE | medium | tune `mu`; consider a schedule |
| 4 stable dims is too few | low | 7-DoF Sawyer summarised by 4-dim end-effector + gripper is enough to anchor a useful body |
| code regression | low | runner glue not yet active; existing code paths untouched |

## File checksums (this push)

- `contrastive/state_mask.py`: 77 lines
- `contrastive/decomposed_networks.py`: 263 lines
- `contrastive/continual_learning_decomposed.py`: 561 lines
- `contrastive/continual_config.py`: +3 fields
- `run_continual_contrastive.py`: +3 imports, +8 kwargs
