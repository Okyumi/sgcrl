# N4b runner glue: `critic_mode='decomposed'` end-to-end

Date: 2026-05-08

What ships in this commit:

- `train_single_task` in `run_continual_contrastive.py` now dispatches
  the decomposed-critic algorithm end-to-end when
  `--critic_mode=decomposed`. The CKA / persistent / reset code paths
  are untouched.
- `main` threads the eight decomposed `prev_*` kwargs across tasks and
  persists them in the per-task checkpoint.

## Project context (re-confirmed before editing)

- 10-task Sawyer Meta-World (`sawyer_box` excluded). All tasks padded
  to `STATE_DIM_UNIFIED = GOAL_DIM_UNIFIED = 11` by `env_utils.py` so a
  single policy / critic works across tasks. Goal is index-aligned with
  state: `goal[i]` is the desired value for `state[i]`.
- `TaskIDGymWrapper` appends a one-hot per task id to BOTH state and
  goal halves: layout `[state(11), one_hot(num_tasks), goal(11),
  one_hot(num_tasks)]`. `obs_dim = 11 + num_tasks`. The critic sees the
  task id in both `phi(s,a)` and `psi(g)`.
- `flatten_fn` builds HER-relabeled transitions: `obs = [state ||
  future_state_as_goal]`, `next_obs = [next_state ||
  future_state_as_goal]`. The decomposed critic's `obs[:, :obs_dim]` /
  `obs[:, obs_dim:]` slices read state and goal halves correctly.
- `STABLE_INDICES = (0, 1, 2, 3)` (EE xyz + gripper) live at indices
  0..3 of the state half. The task one-hot occupies indices 11..11+T-1.
  The dyn target `next_state[:, stable_idx]` therefore picks up exactly
  EE+gripper and ignores the one-hot, as intended.

## Three blocks landed in `train_single_task`

1. **Early FLAG-side guards.** `critic_mode='decomposed'` rejects
   `use_td`, `twin_q`, `use_image_obs`, `entropy_coefficient is not
   None`, `neg_bank_mode != 'off'`, `actor_mode='cka'`, and
   `k_sample_k>0` before the replay server boots. The decomposed
   learner also raises on `use_td` / `twin_q` internally; fail-fast at
   the runner edge keeps the error attributable to one source.

2. **Network construction.** After the existing
   `networks = contrastive.make_networks(...)` call (which the actor
   still uses), build:
   ```python
   decomp_nets = make_decomposed_networks(
       env_spec, obs_dim=obs_dim,
       repr_dim=config.repr_dim,
       use_residual=config.use_residual,
       network_width=config.network_width,
       critic_depth=config.critic_depth,
       phi_task_width=continual_cfg.phi_task_width,
       phi_task_depth=continual_cfg.phi_task_depth,
       energy_fn=config.energy_fn,
       repr_norm=config.repr_norm,
   )
   ```
   The actor still flows through `make_networks`; `policy_network /
   sample / log_prob` are read from there and handed to the
   decomposed learner.

3. **Learner construction.** Wrapped in an `if/else`:
   - `critic_mode=='decomposed'` → `ContinualDecomposedLearner` with
     all eight carry-forward kwargs (`prev_b_shared_*`,
     `prev_h_phi_*`, `prev_h_dyn_*`, `prev_psi_*`).
   - else → existing `ContinualContrastiveLearner` call, untouched.

4. **Post-task extraction.** When `critic_mode=='decomposed'` we
   short-circuit the actor-pool / critic-q extraction (no `v_k`, no
   `theta_base` fold, no critic CKA pool) and instead pull the four
   shared critic groups + opt states from the learner. The
   non-decomposed branch is unchanged. Both branches now return a
   17-tuple: existing 9 fields + 8 decomposed slots (None for
   non-decomposed paths).

## Three blocks landed in `main`

- Eight `prev_*` decomposed variables initialised to `None` at the
  top of `main`.
- Auto-resume now reads `decomposed_*` keys from the checkpoint when
  `--critic_mode=decomposed`.
- The `train_single_task(...)` call site unpacks the 17-tuple and
  passes the eight `prev_*` decomposed kwargs forward.
- The per-task checkpoint dict gets eight `decomposed_*` entries when
  `--critic_mode=decomposed` is active.

## Smoke

Local stubbed-acme run (production env runs JAX 0.4.10 on the
cluster; local has incompatible JAX) verified:

- AST parse of `run_continual_contrastive.py` clean (1479 lines).
- `train_single_task` signature has all 26 expected arguments
  (existing 18 + eight decomposed `prev_*`).
- `make_decomposed_networks` produces a valid (B, B) inner-product
  score matrix at the realistic 10-task shape (`obs_dim = 21`,
  full obs `42`, action dim `4`).
- Dyn target slicing: `next_state[:, (0,1,2,3)]` returns shape
  `(B, 4)` (matches `d_M`); the task one-hot at indices 11..20 is
  not part of the target.
- Gradient isolation between InfoNCE and dyn losses preserved (from
  the earlier verification smoke).

A one-task end-to-end smoke (item N5) is the next blocker; it must
run on the cluster.

## How to run (end-to-end)

```bash
python run_continual_contrastive.py \
    --critic_mode=decomposed \
    --actor_mode=reset \
    --use_task_id=True \
    --seed=42 --num_tasks=10 --steps_per_task=8000000 \
    [other defaults preserved]
```

Adjustable knobs (continual_config):

- `dyn_aux_weight` (default 1.0). Set 0.0 for the regression-check
  cell (decomposed structure with no dynamics signal).
- `phi_task_width` (default 256), `phi_task_depth` (default 2). The
  per-task additive embedding's capacity.

The Step-0 / Step-1 / Step-2 patch in
`2026-05-08_how_to_run_decomposed.md` is now resolved by N4b — the
runner does this automatically when `--critic_mode=decomposed` is
passed.

## Files touched

- `run_continual_contrastive.py` (network build, learner build,
  post-task extraction, return signature, main-loop carry, ckpt
  extension, auto-resume read).
- `docs/2026-05-08_implementation_tracking.md` (N4b row).
- `docs/2026-05-08_decomposed_critic_n4b.md` (this file, new).

No public API changes outside `run_continual_contrastive.py`'s
`train_single_task` return signature (which only the in-file
`main` loop unpacks).
